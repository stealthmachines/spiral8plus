# dna_echo_colour_debug_fixed.py
# --------------------------------------------------------------
# GPU-accelerated double φ-spiral with explicit per-vertex colors.
# All subsequent geometries now properly colored.
# Requires: pip install vispy pyqt6
# --------------------------------------------------------------

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
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800),
                           bgcolor=(0, 0, 0, 1), show=True)
canvas.context.set_state('translucent', blend=True,
                         blend_func=('src_alpha', 'one_minus_src_alpha'))

view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 45
view.camera.distance = 8
view.camera.up = '+y'

# ---------- HELPER: color -> numpy rgba ----------
def rgba_for(col):
    return np.array(Color(col).rgba, dtype=np.float32)

# ---------- PRE-CREATE STRANDS ----------
strand1 = Line(parent=view.scene, method='gl', connect='strip', antialias=True)
strand2 = Line(parent=view.scene, method='gl', connect='strip', antialias=True)

# ---------- COLLECTIONS ----------
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

        s1[i] = [r*np.cos(theta)*np.cos(a1) - r*np.sin(theta)*np.sin(a1),
                 r*np.sin(theta)*np.cos(a1) + r*np.cos(theta)*np.sin(a1),
                 z]
        s2[i] = [r*np.cos(theta)*np.cos(a2) - r*np.sin(theta)*np.sin(a2),
                 r*np.sin(theta)*np.cos(a2) + r*np.cos(theta)*np.sin(a2),
                 z]

        rgba = rgba_for(col)
        a = 0.9 if i > 0.9*N else 1.0
        rgba_a = rgba.copy()
        rgba_a[3] *= a
        colors_s1[i] = rgba_a
        colors_s2[i] = rgba_a

    # Update strands
    strand1.set_data(pos=s1, color=colors_s1, width=2.0)
    strand2.set_data(pos=s2, color=colors_s2, width=2.0)

    # ---- emergence of new shapes ----
    cur_dim = min(int((t[-1] * speed_factor) // period), 7)
    if frame % 20 == 0 and (len(emerged) == 0 or cur_dim != emerged[-1]):
        dim, note, col, name, alpha, verts = geometries[cur_dim]
        emerged.append(cur_dim)
        print(f"[debug] Emerged dim {cur_dim} color {col}")

        start = int((cur_dim / 8.0) * N)
        step = max(1, N // (8*verts))
        idx = np.arange(start, start + verts*step, step)[:verts]
        idx = idx[idx < N]

        pts1 = s1[idx]
        pts2 = s2[idx]

        # ----- fresh RGBA arrays per rung -----
        rgba = rgba_for(col)
        edge_rgba = np.array([rgba[0], rgba[1], rgba[2], 0.95], dtype=np.float32)
        face_colors = np.tile(rgba[None,:], (len(pts1)+len(pts2),1)).astype(np.float32)

        # ----- lattice edges -----
        segs = []
        if verts <= 4:
            for pts in (pts1, pts2):
                for i in range(verts):
                    for j in range(i+1, verts):
                        segs += [pts[i], pts[j]]
            for i in range(verts):
                segs += [pts1[i], pts2[i]]
                if verts > 2:
                    for k in range(1, verts):
                        segs += [pts1[i], pts1[(i+k)%verts], pts2[i], pts2[(i+k)%verts]]
        else:
            g = int(np.ceil(np.sqrt(verts)))
            for i in range(g):
                for j in range(g):
                    n = min(i*g + j, verts-1)
                    if j+1 < g and n+1 < verts:
                        segs += [pts1[n], pts1[n+1], pts2[n], pts2[n+1]]
                    if i+1 < g and n+g < verts:
                        segs += [pts1[n], pts1[n+g], pts2[n], pts2[n+g]]
            for n in range(verts):
                segs += [pts1[n], pts2[n]]

        if segs:
            segs = np.array(segs, dtype=np.float32)
            colors_segs = np.tile(edge_rgba[None,:], (len(segs),1))
            rung_line = Line(pos=segs, color=colors_segs, width=2.0,
                             connect='segments', method='gl', parent=view.scene)
            rungs.append(rung_line)

        # ----- vertex markers -----
        all_pts = np.vstack((pts1, pts2))
        mark = Markers(parent=view.scene)
        mark.set_data(pos=all_pts, face_color=face_colors,
                      edge_color=(1.0,1.0,1.0,1.0), size=8)
        rungs.append(mark)

        # ----- label -----
        cen = all_pts.mean(axis=0)
        centers.append(cen)
        lbl = Text(f"{dim}D: {name}\n{note}", pos=cen + [0,0,0.3],
                   color=rgba, font_size=10, bold=True,
                   anchor_x='center', parent=view.scene)
        labels.append(lbl)

        # ----- echo -----
        if len(emerged) > 1:
            prev = emerged[-2]
            prev_rgba = rgba_for(geometries[prev][2])
            prev_rgba[3] = 0.25
            echo_pts = all_pts * 0.75
            echo = Markers(parent=view.scene)
            echo.set_data(pos=echo_pts, face_color=np.tile(prev_rgba[None,:], (len(echo_pts),1)), size=5)
            echoes.append(echo)

        # ----- inter-rung links -----
        if len(centers) > 1:
            prev_c = centers[-2]
            segs_link = []
            for i in range(min(6, len(all_pts))):
                segs_link += [prev_c, cen]
            segs_link = np.array(segs_link, dtype=np.float32)
            colors_link = np.tile(np.array([0.7,0.7,0.7,0.4],dtype=np.float32)[None,:], (len(segs_link),1))
            link_line = Line(pos=segs_link, color=colors_link, width=1.0,
                             connect='segments', method='gl', parent=view.scene)
            links.append(link_line)

    # ---- auto-rotate ----
    view.camera.azimuth = frame * 0.3
    view.camera.elevation = 15
    canvas.update()

# ---------- RUN ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == "__main__":
    print("Starting app with proper color handling...")
    app.run()
