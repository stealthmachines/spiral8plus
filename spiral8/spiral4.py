# spiral3.py
# --------------------------------------------------------------
# 8-Geometry φ-spiral – all 8 polytopes emerge in sequence
# Optimized for speed: reduced points, faster interval, added speed factor for emergence
# Note: Matplotlib is primarily CPU-based; for true hardware acceleration (GPU),
#       consider migrating to VisPy (pip install vispy) – see comments at bottom.
# --------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# ---------- YOUR FRAMEWORK DATA ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)               # 137.507°

# dim, note, color, name, α, vertex-count (reduced for visibility)
geometries = [
    (1, 'C', 'red',          'Point',        0.015269, 1),
    (2, 'D', 'green',        'Line',         0.008262, 2),
    (3, 'E', 'violet',       'Triangle',     0.110649, 3),
    (4, 'F', 'mediumpurple', 'Tetrahedron', -0.083485, 4),
    (5, 'G', 'blue',         'Pentachoron',  0.025847, 5),
    (6, 'A', 'indigo',       'Hexacross',   -0.045123, 12),
    (7, 'B', 'purple',       'Heptacube',    0.067891, 14),   # reduced from 128
    (8, 'C', 'white',        'Octacube',     0.012345, 16),   # reduced from 256
]

# Cumulative golden-angle rotations for each dimension
angles = [i * golden_angle_deg for i in range(8)]

# Spiral period (your framework)
period = 13.057
t_max = period * 8                     # 8 full turns → one per dimension

# SPEED UP: Factor to accelerate dimension emergence (higher = faster transitions to higher D)
speed_factor = 2.0  # Adjust this (e.g., 3.0 for even faster)

# ---------- MATPLOTLIB SETUP ----------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')
fig.patch.set_facecolor('black')

spiral_line = None
poly_markers = []
labels = []

def init():
    global spiral_line, poly_markers, labels
    ax.cla()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-2, 2)
    ax.set_axis_off()

    spiral_line, = ax.plot([], [], [], color='white', lw=1.5, alpha=0.6)
    poly_markers = []
    labels = []
    return [spiral_line]

def update(frame):
    global spiral_line, poly_markers, labels

    # ---- clear previous polytope graphics ----
    for m in poly_markers:
        m.remove()
    for l in labels:
        l.remove()
    poly_markers = []
    labels = []

    # ---- generate spiral up to current frame ----
    # SPEED UP: Reduced from 1000 to 500 points for faster computation
    t = np.linspace(0, frame * t_max / 360, 500)
    x, y, z = [], [], []

    current_dim = -1
    points_in_dim = []

    for i, tt in enumerate(t):
        # SPEED UP: Accelerate dimension indexing with speed_factor
        dim_idx = min(int((tt * speed_factor) // period), 7)
        dim, note, color, name, alpha, verts = geometries[dim_idx]

        # radius = exp(α · t)  (exponential growth per your α)
        r = np.exp(alpha * (tt % period))
        theta = tt * 2 * np.pi / period

        # rotate by the cumulative golden angle for this dimension
        ang_rad = np.radians(angles[dim_idx])
        xx = r * np.cos(theta) * np.cos(ang_rad) - r * np.sin(theta) * np.sin(ang_rad)
        yy = r * np.sin(theta) * np.cos(ang_rad) + r * np.cos(theta) * np.sin(ang_rad)
        zz = (tt / period - dim_idx) * 0.8          # rise per dimension

        x.append(xx)
        y.append(yy)
        z.append(zz)

        # ---- collect points for the current polytope ----
        if len(points_in_dim) < verts:
            points_in_dim.append([xx, yy, zz])
        elif dim_idx > current_dim:                # dimension just finished
            pts = np.array(points_in_dim)

            # draw edges (connect in order + close)
            for j in range(len(pts)):
                ax.plot(pts[[j, (j+1) % len(pts)], 0],
                        pts[[j, (j+1) % len(pts)], 1],
                        pts[[j, (j+1) % len(pts)], 2],
                        color=color, lw=2, alpha=0.9)

            # glow-dots
            scatter = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                                 c=color, s=80, depthshade=False,
                                 edgecolors='white', linewidths=1.5)
            poly_markers.append(scatter)

            # label
            cx, cy, cz = pts.mean(axis=0)
            txt = ax.text(cx, cy, cz + 0.3,
                          f"{dim}D: {name}\n{note}",
                          color=color, fontsize=9, ha='center', weight='bold')
            labels.append(txt)

            points_in_dim = [[xx, yy, zz]]
            current_dim = dim_idx

    # ---- update the continuous spiral trail ----
    spiral_line.set_data_3d(x, y, z)

    # ---- rotate camera ----
    ax.view_init(elev=20, azim=frame * 0.5)

    return [spiral_line] + poly_markers + labels

# ---------- ANIMATION ----------
# SPEED UP: Reduced interval from 50ms to 20ms for faster playback
ani = FuncAnimation(fig, update, frames=360,
                    init_func=init, interval=20, blit=False)

# Uncomment to save a GIF (requires pillow)
# ani.save('8_geometries_emerging.gif', writer='pillow', fps=60, dpi=100)

plt.show()