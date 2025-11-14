# dna_geometries_fixed.py
# --------------------------------------------------------------
# GPU-accelerated double φ-spiral: DNA-like structure
# Two strands (forward + reverse golden angle) with 8D geometries as base pairs
# Install: pip install vispy pyqt6
# --------------------------------------------------------------

import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Text
from vispy.color import Color

# ---------- FRAMEWORK DATA ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)               # 137.507°

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

# <<< THIS WAS MISSING >>>
angles = [i * golden_angle_deg for i in range(8)]   # cumulative golden angles

period = 13.057
t_max = period * 8
speed_factor = 2.5          # increase to see higher-D faster

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Two spiral strands
strand1_pos = np.zeros((1, 3))
strand2_pos = np.zeros((1, 3))
strand1_line = Line(pos=strand1_pos, color=(1,1,1,0.7), width=2, parent=view.scene)
strand2_line = Line(pos=strand2_pos, color=(1,1,1,0.7), width=2, parent=view.scene)

# Base-pair connections + labels
base_pairs = []
labels = []

frame = 0

def update(ev):
    global frame, base_pairs, labels
    frame += 1

    num_points = 600
    t = np.linspace(0, (frame / 360.0) * t_max, num_points)

    s1_x, s1_y, s1_z = [], [], []
    s2_x, s2_y, s2_z = [], [], []

    for i, tt in enumerate(t):
        dim_idx = min(int((tt * speed_factor) // period), 7)
        _, _, color_str, _, alpha, _ = geometries[dim_idx]

        r = np.exp(alpha * (tt % period))
        theta = tt * 2 * np.pi / period

        # Strand 1: +golden angle
        ang1 = np.radians(angles[dim_idx])
        s1_x.append(r * np.cos(theta) * np.cos(ang1) - r * np.sin(theta) * np.sin(ang1))
        s1_y.append(r * np.sin(theta) * np.cos(ang1) + r * np.cos(theta) * np.sin(ang1))
        s1_z.append((tt / period) * 0.8)

        # Strand 2: -golden angle (counter-rotating)
        ang2 = np.radians(-angles[dim_idx])
        s2_x.append(r * np.cos(theta) * np.cos(ang2) - r * np.sin(theta) * np.sin(ang2))
        s2_y.append(r * np.sin(theta) * np.cos(ang2) + r * np.cos(theta) * np.sin(ang2))
        s2_z.append((tt / period) * 0.8)

    # Update strands
    strand1_line.set_data(pos=np.column_stack((s1_x, s1_y, s1_z)))
    strand2_line.set_data(pos=np.column_stack((s2_x, s2_y, s2_z)))

    # --------------------------------------------------------------
    # Add a new base-pair geometry when a dimension finishes
    # --------------------------------------------------------------
    max_tt = t[-1]
    new_dim = min(int((max_tt * speed_factor) // period), 7)

    if frame % 30 == 0 and new_dim > len(base_pairs) - 1:
        dim, note, color_str, name, alpha, verts = geometries[new_dim]

        # Sample points from both strands in the *current* dimension
        start = int((new_dim / 8.0) * num_points)
        step  = max(1, num_points // (8 * verts))
        idxs  = np.arange(start, start + verts * step, step)[:verts]

        # Guard against out-of-bounds (should never happen, but safe)
        idxs = idxs[idxs < len(s1_x)]

        pts1 = np.column_stack((np.array(s1_x)[idxs],
                                np.array(s1_y)[idxs],
                                np.array(s1_z)[idxs]))
        pts2 = np.column_stack((np.array(s2_x)[idxs],
                                np.array(s2_y)[idxs],
                                np.array(s2_z)[idxs]))

        # Connect corresponding points → base-pair “rungs”
        for p1, p2 in zip(pts1, pts2):
            pair = Line(pos=np.vstack((p1, p2)),
                        color=Color(color_str).rgba,
                        width=3,
                        parent=view.scene)
            base_pairs.append(pair)

        # Label at the centre of the rung set
        centre = (pts1.mean(axis=0) + pts2.mean(axis=0)) / 2
        label = Text(f"{dim}D: {name}\n{note}",
                     pos=centre,
                     color=color_str,
                     font_size=10,
                     bold=True,
                     anchor_x='center',
                     parent=view.scene)
        labels.append(label)

    # Auto-rotate the view
    view.camera.azimuth   = frame * 0.3
    view.camera.elevation = 15

    canvas.update()


# ---------- START ANIMATION ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    app.run()