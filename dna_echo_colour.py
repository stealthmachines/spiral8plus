# dna_echo_colour.py
# --------------------------------------------------------------
# GPU-accelerated double φ-spiral with colour, closed lattices,
# inter-shape links and infinite echoing back to the source.
# Requires: pip install vispy pyqt6
# --------------------------------------------------------------

import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- FRAMEWORK ----------
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
speed_factor = 5.0  # fast emergence

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor=(0, 0, 0, 1), show=True)
canvas.context.set_state('translucent', blend=True,
                         blend_func=('src_alpha', 'one_minus_src_alpha'))

view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 45
view.camera.distance = 8

# strands (initialized blank, updated later)
strand1 = Line(parent=view.scene, method='gl')
strand2 = Line(parent=view.scene, method='gl')
strand1.set_data(pos=np.zeros((1, 3)), color=(1, 1, 1, 0.7), width=2)
strand2.set_data(pos=np.zeros((1, 3)), color=(1, 1, 1, 0.7), width=2)

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
    s1, s2 = [], []

    for tt in t:
        dim = min(int((tt * speed_factor) // period), 7)
        _, _, col, _, alpha, _ = geometries[dim]
        r = np.exp(alpha * (tt % period))
        theta = tt * 2 * np.pi / period

        a1 = np.radians(angles[dim])
        a2 = np.radians(-angles[dim])
        z = (tt / period) * 0.8

        s1.append([
            r*np.cos(theta)*np.cos(a1) - r*np.sin(theta)*np.sin(a1),
            r*np.sin(theta)*np.cos(a1) + r*np.cos(theta)*np.sin(a1),
            z
        ])
        s2.append([
            r*np.cos(theta)*np.cos(a2) - r*np.sin(theta)*np.sin(a2),
            r*np.sin(theta)*np.cos(a2) + r*np.cos(theta)*np.sin(a2),
            z
        ])

    strand1.set_data(pos=np.array(s1), color=(1, 1, 1, 0.8))
    strand2.set_data(pos=np.array(s2), color=(1, 1, 1, 0.8))

    # ---- emergence ----
    cur_dim = min(int((t[-1] * speed_factor) // period), 7)
    if frame % 20 == 0 and cur_dim > len(rungs) - 1:
        dim, note, col, name, alpha, verts = geometries[cur_dim]
        emerged.append(cur_dim)

        start = int((cur_dim / 8.0) * N)
        step = max(1, N // (8 * verts))
        idx = np.arange(start, start + verts * step, step)[:verts]
        idx = idx[idx < N]

        pts1 = np.array(s1)[idx]
        pts2 = np.array(s2)[idx]

        rgba = np.array(Color(col).rgba, dtype=np.float32)
        edge_rgba = np.array([*rgba[:3], 0.9], dtype=np.float32)

        # ----- lattice edges -----
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
            line = Line(pos=np.array(segs), color=edge_rgba, width=2,
                        connect='segments', method='gl', parent=view.scene)
            rungs.append(line)

        # ----- vertex markers -----
        all_pts = np.vstack((pts1, pts2))
        mark = Markers(parent=view.scene)
        mark.set_data(pos=all_pts, face_color=rgba, edge_color='white', size=8)
        rungs.append(mark)

        # ----- labels -----
        cen = all_pts.mean(axis=0)
        centers.append(cen)
        lbl = Text(f"{dim}D: {name}\n{note}", pos=cen + [0, 0, 0.3],
                   color=Color(col).rgba, font_size=10, bold=True,
                   anchor_x='center', parent=view.scene)
        labels.append(lbl)

        # ----- faint echo -----
        if len(emerged) > 1:
            prev = emerged[-2]
            prev_rgba = np.array(Color(geometries[prev][2]).rgba, dtype=np.float32)
            prev_rgba[3] = 0.25
            echo_pts = all_pts * 0.75
            echo = Markers(parent=view.scene)
            echo.set_data(pos=echo_pts, face_color=prev_rgba, size=5)
            echoes.append(echo)

        # ----- inter-rung links -----
        if len(centers) > 1:
            prev_c = centers[-2]
            segs = []
            for i in range(min(6, len(all_pts))):
                segs += [prev_c, cen]
            link = Line(pos=np.array(segs),
                        color=(0.7, 0.7, 0.7, 0.4),
                        width=1, connect='segments', method='gl',
                        parent=view.scene)
            links.append(link)

    # ---- auto-rotate ----
    view.camera.azimuth = frame * 0.3
    view.camera.elevation = 15
    canvas.update()


# ---------- RUN ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    app.run()
