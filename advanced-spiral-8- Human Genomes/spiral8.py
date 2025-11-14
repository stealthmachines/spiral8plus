# dna_closed_interact.py
# --------------------------------------------------------------
# GPU-accelerated double φ-spiral: DNA-like with closed geometries + inter-shape interactions
# - Closes forms with full lattice (low-D: edges/faces; high-D: grid)
# - Echoes: Faint scaled repetitions
# - Shape-to-shape: Connect consecutive rungs with φ-scaled edges
# - Faster emergence: All 8 by ~frame 200 (adjust speed_factor)
# Install: pip install vispy pyqt6
# --------------------------------------------------------------

import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- FRAMEWORK DATA ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)  # 137.507°

geometries = [
    (1, 'C', 'red',          'Point',        0.015269, 1),
    (2, 'D', 'green',        'Line',         0.008262, 2),
    (3, 'E', 'violet',       'Triangle',     0.110649, 3),
    (4, 'F', 'mediumpurple', 'Tetrahedron', -0.083485, 4),
    (5, 'G', 'blue',         'Pentachoron',  0.025847, 5),
    (6, 'A', 'indigo',       'Hexacross',   -0.045123, 12),
    (7, 'B', 'purple',       'Heptacube',    0.067891, 14),
    (8, 'C', 'white',        'Octacube',     0.012345, 16),
]

angles = [i * golden_angle_deg for i in range(8)]

period = 13.057
t_max = period * 8
speed_factor = 5.0  # Higher for faster full emergence (all 8 quick)

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Strands
strand1_pos = np.zeros((1, 3))
strand2_pos = np.zeros((1, 3))
strand1_line = Line(pos=strand1_pos, color=(1,1,1,0.7), width=2, parent=view.scene)
strand2_line = Line(pos=strand2_pos, color=(1,1,1,0.7), width=2, parent=view.scene)

# Collections
base_pairs = []
echoes = []
inter_links = []  # Shape-to-shape connections
labels = []
rung_centers = []  # Track centers for inter-links
emerged_dims = []

frame = 0

def update(ev):
    global frame, base_pairs, echoes, inter_links, labels, rung_centers, emerged_dims
    frame += 1

    num_points = 600
    t = np.linspace(0, min((frame / 360.0) * t_max * speed_factor, t_max), num_points)  # Cap at t_max

    s1_x, s1_y, s1_z = [], [], []
    s2_x, s2_y, s2_z = [], [], []

    for i, tt in enumerate(t):
        dim_idx = min(int((tt * speed_factor) // period), 7)
        _, _, color_str, _, alpha, _ = geometries[dim_idx]

        r = np.exp(alpha * (tt % period))
        theta = tt * 2 * np.pi / period

        ang1 = np.radians(angles[dim_idx])
        s1_x.append(r * np.cos(theta) * np.cos(ang1) - r * np.sin(theta) * np.sin(ang1))
        s1_y.append(r * np.sin(theta) * np.cos(ang1) + r * np.cos(theta) * np.sin(ang1))
        s1_z.append((tt / period) * 0.8)

        ang2 = np.radians(-angles[dim_idx])
        s2_x.append(r * np.cos(theta) * np.cos(ang2) - r * np.sin(theta) * np.sin(ang2))
        s2_y.append(r * np.sin(theta) * np.cos(ang2) + r * np.cos(theta) * np.sin(ang2))
        s2_z.append((tt / period) * 0.8)

    strand1_line.set_data(pos=np.column_stack((s1_x, s1_y, s1_z)))
    strand2_line.set_data(pos=np.column_stack((s2_x, s2_y, s2_z)))

    # Emerge new rung with closed lattice
    max_tt = t[-1]
    new_dim = min(int((max_tt * speed_factor) // period), 7)
    if frame % 20 == 0 and new_dim > len(base_pairs) - 1:  # Faster trigger
        dim, note, color_str, name, alpha, verts = geometries[new_dim]
        emerged_dims.append(new_dim)

        start = int((new_dim / 8.0) * num_points)
        step = max(1, num_points // (8 * verts))
        idxs = np.arange(start, start + verts * step, step)[:verts]
        idxs = idxs[idxs < len(s1_x)]

        pts1 = np.column_stack((np.array(s1_x)[idxs], np.array(s1_y)[idxs], np.array(s1_z)[idxs]))
        pts2 = np.column_stack((np.array(s2_x)[idxs], np.array(s2_y)[idxs], np.array(s2_z)[idxs]))

        color_rgba = Color(color_str).rgba
        edge_color = list(color_rgba[:3]) + [0.9]

        # Closed lattice
        edge_segments = []
        if verts <= 4:  # Low-D: complete + cross-faces
            for pts in [pts1, pts2]:
                for j in range(verts):
                    for k in range(j + 1, verts):
                        edge_segments.append(pts[j])
                        edge_segments.append(pts[k])
            for j in range(verts):
                edge_segments.append(pts1[j])
                edge_segments.append(pts2[j])
                if verts > 2:
                    for k in range(1, verts):
                        edge_segments.append(pts1[j])
                        edge_segments.append(pts1[(j + k) % verts])
                        edge_segments.append(pts2[j])
                        edge_segments.append(pts2[(j + k) % verts])
        else:  # High-D: grid lattice
            grid_size = int(np.ceil(np.sqrt(verts)))
            for i in range(grid_size):
                for j in range(grid_size):
                    idx = min(i * grid_size + j, verts - 1)
                    if j + 1 < grid_size and idx + 1 < verts:
                        edge_segments.append(pts1[idx])
                        edge_segments.append(pts1[idx + 1])
                        edge_segments.append(pts2[idx])
                        edge_segments.append(pts2[idx + 1])
                    if i + 1 < grid_size and idx + grid_size < verts:
                        edge_segments.append(pts1[idx])
                        edge_segments.append(pts1[idx + grid_size])
                        edge_segments.append(pts2[idx])
                        edge_segments.append(pts2[idx + grid_size])
            for idx in range(verts):
                edge_segments.append(pts1[idx])
                edge_segments.append(pts2[idx])

        if edge_segments:
            edges = Line(pos=np.array(edge_segments), color=edge_color, width=2, connect='segments', parent=view.scene)
            base_pairs.append(edges)

        all_pts = np.vstack((pts1, pts2))
        markers = Markers(pos=all_pts, face_color=color_rgba, edge_color='white', size=8, parent=view.scene)
        base_pairs.append(markers)

        center = all_pts.mean(axis=0)
        rung_centers.append(center)

        label = Text(f"{dim}D: {name}\n{note}", pos=center + [0, 0, 0.3], color=color_str,
                     font_size=10, bold=True, anchor_x='center', parent=view.scene)
        labels.append(label)

        # Echo: faint previous
        if len(emerged_dims) > 1:
            prev_dim = emerged_dims[-2]
            prev_color = list(Color(geometries[prev_dim][2]).rgba)
            prev_color[3] = 0.3
            echo_markers = Markers(pos=all_pts * 0.8, face_color=prev_color, size=5, parent=view.scene)
            echoes.append(echo_markers)

        # Shape-to-shape interactions: connect to previous rung
        if len(rung_centers) > 1:
            prev_center = rung_centers[-2]
            link_segments = []
            for i in range(min(5, len(all_pts))):  # Sample connections
                link_segments.append(prev_center)
                link_segments.append(center)
                link_segments.append(rung_centers[-2] + np.random.normal(0, 0.1, 3))
                link_segments.append(rung_centers[-1] + np.random.normal(0, 0.1, 3))
            inter = Line(pos=np.array(link_segments), color=(0.5,0.5,0.5,0.5), width=1, connect='segments', parent=view.scene)
            inter_links.append(inter)

    # Auto-rotate
    view.camera.azimuth = frame * 0.3
    view.camera.elevation = 15

    canvas.update()

# Start
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    app.run()