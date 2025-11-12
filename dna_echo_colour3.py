# dna_echo_colour_debug.py
# --------------------------------------------------------------
# GPU-accelerated double φ-spiral with explicit per-vertex colors.
# Requires: pip install vispy pyqt6
# --------------------------------------------------------------

import os
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- FRAMEWORK ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)  # ~137.507°

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
speed_factor = 5.0  # fast emergence

# ---------- VISPY SETUP ----------
# Debug: show which backend and GL info we have available
print("VISPY_BACKEND:", app.use_app())  # backend name
# Some GL info comes from canvas.context once created below.

canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800),
                           bgcolor=(0, 0, 0, 1), show=True)

# Attempt to set blending and disable depth test (so colors won't be hidden)
try:
    canvas.context.set_state(blend=True, depth_test=False)
    canvas.context.set_state('translucent', blend=True,
                             blend_func=('src_alpha', 'one_minus_src_alpha'))
except Exception as e:
    print("Warning: couldn't set GL states:", e)

# Print a little info about GL capability (best-effort)
try:
    info = canvas.context.gl_info  # attribute exists in many Vispy versions
    print("GL info:", info)
except Exception:
    try:
        print("GL version:", canvas.context.gl_version)
    except Exception:
        pass

view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 45
view.camera.distance = 8
view.camera.up = '+y'

# ---------- helper: color -> np rgba ----------
def rgba_for(col):
    """Return numpy (N,4) RGBA float32 or single (4,) RGBA"""
    c = Color(col).rgba  # tuple of 4 floats
    return np.array(c, dtype=np.float32)

# ---------- Pre-create visuals ----------
# Strands are drawn as Lines with per-vertex color arrays
strand1 = Line(parent=view.scene, method='gl', connect='strip', antialias=True)
strand2 = Line(parent=view.scene, method='gl', connect='strip', antialias=True)

# collections
rungs = []
echoes = []
links = []
labels = []
centers = []
emerged = []

frame = 0

# ---------- UPDATE LOOP ----------
def update(ev):
    global frame, rungs, echoes, links, labels, centers, emerged
    frame += 1

    # ---- spiral growth ----
    N = 600
    t = np.linspace(0, (frame / 360.0) * t_max * speed_factor, N)
    s1 = np.zeros((N, 3), dtype=np.float32)
    s2 = np.zeros((N, 3), dtype=np.float32)
    # per-vertex colors for strands
    colors_s1 = np.zeros((N, 4), dtype=np.float32)
    colors_s2 = np.zeros((N, 4), dtype=np.float32)

    for i, tt in enumerate(t):
        dim = min(int((tt * speed_factor) // period), 7)
        _, _, col, _, alpha, _ = geometries[dim]
        r = np.exp(alpha * (tt % period))
        theta = tt * 2 * np.pi / period

        a1 = np.radians(angles[dim])
        a2 = np.radians(-angles[dim])
        z = (tt / period) * 0.8

        s1[i, 0] = r * np.cos(theta) * np.cos(a1) - r * np.sin(theta) * np.sin(a1)
        s1[i, 1] = r * np.sin(theta) * np.cos(a1) + r * np.cos(theta) * np.sin(a1)
        s1[i, 2] = z

        s2[i, 0] = r * np.cos(theta) * np.cos(a2) - r * np.sin(theta) * np.sin(a2)
        s2[i, 1] = r * np.sin(theta) * np.cos(a2) + r * np.cos(theta) * np.sin(a2)
        s2[i, 2] = z

        rgba = rgba_for(col)
        # use a smaller alpha for far points to give depth feel
        a = 0.9 if (i > 0.9 * N) else 1.0
        rgba_a = np.array([rgba[0], rgba[1], rgba[2], rgba[3] * a], dtype=np.float32)

        colors_s1[i] = rgba_a
        colors_s2[i] = rgba_a

    # Debug prints: confirm arrays are OK
    if frame % 60 == 0:
        print(f"[debug] strands: s1 shape {s1.shape}, colors shape {colors_s1.shape}, sample RGBA {colors_s1[0]}")

    # Provide per-vertex color arrays to the Line visuals
    try:
        strand1.set_data(pos=s1, color=colors_s1, width=2.0)
        strand2.set_data(pos=s2, color=colors_s2, width=2.0)
    except Exception as e:
        # Some vispy versions require color as (N,3) or (N,4) float32; we already use (N,4)
        print("Error setting strand data with per-vertex colors:", e)
        strand1.set_data(pos=s1, color=(1, 1, 1, 0.8), width=2.0)
        strand2.set_data(pos=s2, color=(1, 1, 1, 0.8), width=2.0)

    # ---- emergence (occasionally add shapes) ----
    cur_dim = min(int((t[-1] * speed_factor) // period), 7)
    # Only create a rung when there's not already one for that dim
    # We'll track by 'emerged' list (store dim indices)
    if frame % 20 == 0 and (len(emerged) == 0 or cur_dim != emerged[-1]):
        dim, note, col, name, alpha, verts = geometries[cur_dim]
        emerged.append(cur_dim)
        print(f"[debug] Emerged dim {cur_dim} color {col}")

        start = int((cur_dim / 8.0) * N)
        step = max(1, N // (8 * verts))
        idx = np.arange(start, start + verts * step, step)[:verts]
        idx = idx[idx < N]

        pts1 = s1[idx]
        pts2 = s2[idx]

        rgba = rgba_for(col)
        edge_rgba = np.array([rgba[0], rgba[1], rgba[2], 0.95], dtype=np.float32)

        # Build segs as sequence of points for the Line (connect='segments')
        segs = []
        if verts <= 4:
            for pts in (pts1, pts2):
                for i in range(verts):
                    for j in range(i + 1, verts):
                        segs += [pts[i], pts[j]]
            for i in range(verts):
                segs += [pts1[i], pts2[i]]
                if verts > 2:
                    for k in range(1, verts):
                        segs += [
                            pts1[i], pts1[(i + k) % verts],
                            pts2[i], pts2[(i + k) % verts]
                        ]
        else:
            g = int(np.ceil(np.sqrt(verts)))
            for i in range(g):
                for j in range(g):
                    n = min(i * g + j, verts - 1)
                    if j + 1 < g and n + 1 < verts:
                        segs += [pts1[n], pts1[n + 1], pts2[n], pts2[n + 1]]
                    if i + 1 < g and n + g < verts:
                        segs += [pts1[n], pts1[n + g], pts2[n], pts2[n + g]]
            for n in range(verts):
                segs += [pts1[n], pts2[n]]

        if segs:
            segs = np.array(segs, dtype=np.float32)
            # Per-vertex colors for the segs: replicate the edge_rgba for each vertex
            colors_segs = np.tile(edge_rgba[None, :], (len(segs), 1))
            try:
                rung_line = Line(pos=segs, color=colors_segs, width=2.0,
                                 connect='segments', method='gl', parent=view.scene)
            except Exception as e:
                print("Warning: could not set per-vertex colors on rung_line:", e)
                rung_line = Line(pos=segs, color=edge_rgba, width=2.0,
                                 connect='segments', method='gl', parent=view.scene)
            rungs.append(rung_line)

        # Markers: per-vertex face colors
        all_pts = np.vstack((pts1, pts2))
        face_colors = np.tile(rgba[None, :], (len(all_pts), 1)).astype(np.float32)
        try:
            mark = Markers(parent=view.scene)
            mark.set_data(pos=all_pts, face_color=face_colors, edge_color=(1.0, 1.0, 1.0, 1.0), size=8)
        except Exception as e:
            print("Warning: markers per-vertex face_color failed:", e)
            mark = Markers(pos=all_pts, face_color=rgba, edge_color='white', size=8, parent=view.scene)
        rungs.append(mark)

        # Label
        cen = all_pts.mean(axis=0)
        centers.append(cen)
        try:
            lbl = Text(f"{dim}D: {name}\n{note}", pos=cen + [0, 0, 0.3],
                       color=Color(col).rgba, font_size=10, bold=True,
                       anchor_x='center', parent=view.scene)
        except Exception as e:
            print("Warning: Text color set failed:", e)
            lbl = Text(f"{dim}D: {name}\n{note}", pos=cen + [0, 0, 0.3],
                       color=(1, 1, 1, 1), font_size=10, bold=True,
                       anchor_x='center', parent=view.scene)
        labels.append(lbl)

        # echo using previous color if available
        if len(emerged) > 1:
            prev = emerged[-2]
            prev_rgba = rgba_for(geometries[prev][2])
            prev_rgba[3] = 0.25
            echo_pts = all_pts * 0.75
            try:
                echo = Markers(parent=view.scene)
                echo.set_data(pos=echo_pts, face_color=np.tile(prev_rgba[None, :], (len(echo_pts), 1)), size=5)
            except Exception as e:
                print("Warning: echo markers failed:", e)
                echo = Markers(pos=echo_pts, face_color=prev_rgba, size=5, parent=view.scene)
            echoes.append(echo)

        # link to previous centre
        if len(centers) > 1:
            prev_c = centers[-2]
            segs_link = []
            for i in range(min(6, len(all_pts))):
                segs_link += [prev_c, cen]
            segs_link = np.array(segs_link, dtype=np.float32)
            colors_link = np.tile(np.array([0.7, 0.7, 0.7, 0.4], dtype=np.float32)[None, :], (len(segs_link), 1))
            try:
                link_line = Line(pos=segs_link, color=colors_link, width=1.0,
                                 connect='segments', method='gl', parent=view.scene)
            except Exception as e:
                print("Warning: link_line per-vertex color failed:", e)
                link_line = Line(pos=segs_link, color=(0.7, 0.7, 0.7, 0.4),
                                 width=1.0, connect='segments', method='gl', parent=view.scene)
            links.append(link_line)

    # auto-rotate
    view.camera.azimuth = frame * 0.3
    view.camera.elevation = 15
    canvas.update()

# ---------- RUN ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == "__main__":
    print("Starting app... If you see white-only output, check the debug prints above.")
    app.run()
