# spiral_vispy.py
# --------------------------------------------------------------
# GPU-accelerated version using VisPy (hardware acceleration via OpenGL)
# Install with: pip install vispy pyqt6 (or pyside6)
# This builds the spiral cumulatively (no clearing), adds geometries as they emerge
# Geometries use full edge connections for low dims (<=5) to form "actual" shapes
# Higher dims use cycle + center connections for visibility
# Adjust speed_factor for faster emergence of higher-D (e.g., 3.0 to see 8D quicker)
# --------------------------------------------------------------

import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- YOUR FRAMEWORK DATA ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)  # 137.507°

# dim, note, color_str, name, α, vertex-count (reduced for visibility)
geometries = [
    (1, 'C', 'red',          'Point',        0.015269, 1),
    (2, 'D', 'green',        'Line',         0.008262, 2),
    (3, 'E', 'violet',       'Triangle',     0.110649, 3),
    (4, 'F', 'mediumpurple', 'Tetrahedron', -0.083485, 4),
    (5, 'G', 'blue',         'Pentachoron',  0.025847, 5),
    (6, 'A', 'indigo',       'Hexacross',   -0.045123, 12),
    (7, 'B', 'purple',       'Heptacube',    0.067891, 14),  # reduced from 128
    (8, 'C', 'white',        'Octacube',     0.012345, 16),  # reduced from 256
]

# Cumulative golden-angle rotations
angles = [i * golden_angle_deg for i in range(8)]

# Spiral period
period = 13.057
t_max = period * 8  # 8 full turns

# Adjust this to speed up emergence (higher = higher-D appear sooner)
speed_factor = 2.0  # Start with 2.0; set to 1.0 for slower build-up

# ---------- VISPY SETUP (GPU-accelerated) ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1000, 800), bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = 'turntable'  # Interactive 3D camera

# Initial spiral line
spiral_pos = np.zeros((1, 3))  # Start at origin
spiral_line = Line(pos=spiral_pos, color=(1, 1, 1, 0.6), width=1.5, parent=view.scene)

# Track current max dimension shown
current_dim = -1

frame = 0

def update(ev):
    global frame, current_dim

    frame += 1

    # Generate spiral up to current frame (fewer points for speed)
    num_points = 500
    t = np.linspace(0, (frame / 360.0) * t_max, num_points)

    x = np.zeros(num_points)
    y = np.zeros(num_points)
    z = np.zeros(num_points)

    for i, tt in enumerate(t):
        dim_idx = min(int((tt * speed_factor) // period), 7)
        _, _, color_str, _, alpha, _ = geometries[dim_idx]

        r = np.exp(alpha * (tt % period))
        theta = tt * 2 * np.pi / period

        ang_rad = np.radians(angles[dim_idx])
        x[i] = r * np.cos(theta) * np.cos(ang_rad) - r * np.sin(theta) * np.sin(ang_rad)
        y[i] = r * np.sin(theta) * np.cos(ang_rad) + r * np.cos(theta) * np.sin(ang_rad)
        z[i] = (tt / period - dim_idx) * 0.8  # Helical rise

    # Update spiral
    spiral_pos = np.column_stack((x, y, z))
    spiral_line.set_data(pos=spiral_pos)

    # Check if new dimension has emerged
    max_tt = t[-1]
    new_dim_idx = min(int((max_tt * speed_factor) // period), 7)
    if new_dim_idx > current_dim:
        # Emerge the new geometry from the latest spiral segment
        dim, note, color_str, name, alpha, verts = geometries[new_dim_idx]

        # Extract points from the current dimension's spiral segment (approx verts points)
        segment_start = int((new_dim_idx / 8.0) * num_points)
        segment_points = spiral_pos[segment_start:segment_start + verts * 2: int(num_points / (8 * verts))][:verts]
        if len(segment_points) < verts:
            segment_points = spiral_pos[-verts:]  # Fallback to last verts points

        pts = segment_points

        # Color
        color_rgba = Color(color_str).rgba
        edge_color = list(color_rgba[:3]) + [0.9]  # Slightly transparent edges

        # Markers (glow dots)
        markers = Markers(pos=pts, face_color=color_rgba, edge_color='white', size=10,
                          edge_width=1.5, parent=view.scene, symbol='disc')

        # Edges: Full connections for low dims (<=5) to form actual geometry
        # For higher, cycle + connections to "center" for structure
        edge_segments = []
        if verts <= 5:  # Complete graph (e.g., tetra has all 6 edges)
            for j in range(verts):
                for k in range(j + 1, verts):
                    edge_segments.append(pts[j])
                    edge_segments.append(pts[k])
        else:  # Cycle + to mean (star-like for visibility)
            center = pts.mean(axis=0)
            for j in range(verts):
                # Cycle
                edge_segments.append(pts[j])
                edge_segments.append(pts[(j + 1) % verts])
                # To center
                edge_segments.append(pts[j])
                edge_segments.append(center)

        if edge_segments:
            edges_pos = np.array(edge_segments)
            edges = Line(pos=edges_pos, color=edge_color, width=2, connect='segments', parent=view.scene)

        # Label
        cx, cy, cz = pts.mean(axis=0)
        label = Text(f"{dim}D: {name}\n{note}", pos=[cx, cy, cz + 0.3], color=color_rgba,
                     font_size=12, bold=True, anchor_x='center', parent=view.scene)

        current_dim = new_dim_idx

    # Auto-rotate camera
    view.camera.azimuth = frame * 0.5
    view.camera.elevation = 20

    canvas.update()

# Timer for animation (interval=0.02s ~50fps; smaller for faster playback)
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    app.run()