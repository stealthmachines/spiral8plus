# dna_echo_colour_persistent_harmonic.py
# --------------------------------------------------------------
# GPU-accelerated double φ-spiral with fully persistent colors
# for strands, lattices, vertices, echoes, labels, and harmonic rainbow.
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
speed_factor = 5.0

# ---------- VISPY ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# ---------- STRANDS ----------
strand1 = Line(pos=np.zeros((1, 3)), color=(1,1,1,0.7), width=2, parent=view.scene)
strand2 = Line(pos=np.zeros((1, 3)), color=(1,1,1,0.7), width=2, parent=view.scene)

# ---------- COLLECTIONS ----------
rungs = []
echoes = []
links = []
labels = []
centers = []
emerged = []

# ---------- FULL STRAND HISTORY ----------
s1_hist = []
s2_hist = []
colors_s1_hist = []
colors_s2_hist = []

frame = 0

# ---------- HELPERS ----------
def rgba_for(col, alpha=1.0):
    c = np.array(Color(col).rgba, dtype=np.float32)
    c[3] *= alpha
    return c

def harmonic_color(t_norm):
    """Cycle through all geometry colors based on normalized t (0..1)"""
    idx = (t_norm * len(geometries)) % len(geometries)
    low = int(np.floor(idx))
    high = (low + 1) % len(geometries)
    mix = idx - low
    c1 = rgba_for(geometries[low][2])
    c2 = rgba_for(geometries[high][2])
    return c1 * (1 - mix) + c2 * mix

# ---------- UPDATE LOOP ----------
def update(ev):
    global frame, rungs, echoes, links, labels, centers, emerged
    global s1_hist, s2_hist, colors_s1_hist, colors_s2_hist
    frame += 1

    # ---- spiral growth ----
    N = 600
    t = np.linspace(0, (frame / 360.0) * t_max * speed_factor, N)
    s1 = np.zeros((N,3), dtype=np.float32)
    s2 = np.zeros((N,3), dtype=np.float32)
    colors_s1 = np.zeros((N,4), dtype=np.float32)
    colors_s2 = np.zeros((N,4), dtype=np.float32)

    for i, tt in enumerate(t):
        dim = min(int((tt * speed_factor) // period), 7)
        _, _, col, _, alpha, _ = geometries[dim]

        r = np.exp(alpha * (tt % period))
        theta = tt * 2 * np.pi / period
        a1 = np.radians(angles[dim])
        a2 = np.radians(-angles[dim])
        z = (tt / period) * 0.8

        s1[i] = [r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                 r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
                 z]
        s2[i] = [r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2),
                 r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2),
                 z]

        # Full harmonic gradient along the strand
        t_norm = tt / t[-1]
        hc = harmonic_color(t_norm)
        colors_s1[i] = hc
        colors_s2[i] = hc

    # Append to full history
    s1_hist.append(s1.copy())
    s2_hist.append(s2.copy())
    colors_s1_hist.append(colors_s1.copy())
    colors_s2_hist.append(colors_s2.copy())

    # Concatenate full history for continuous rendering
    strand1.set_data(pos=np.vstack(s1_hist), color=np.vstack(colors_s1_hist), width=2.0)
    strand2.set_data(pos=np.vstack(s2_hist), color=np.vstack(colors_s2_hist), width=2.0)

    # ---- emerge new rung every 20 frames ----
    cur_dim = min(int((t[-1] * speed_factor) // period), 7)
    if frame % 20 == 0 and (len(emerged) == 0 or cur_dim != emerged[-1]):
        dim, note, col, name, alpha, verts = geometries[cur_dim]
        emerged.append(cur_dim)

        start = int((cur_dim / 8.0) * N)
        step = max(1, N // (8*verts))
        idx = np.arange(start, start + verts*step, step)[:verts]
        idx = idx[idx < N]

        pts1 = s1[idx]
        pts2 = s2[idx]
        rgba = rgba_for(col)
        edge_rgba = rgba.copy()
        edge_rgba[3] = 0.95

        # ----- closed lattice -----
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
            line = Line(pos=segs, color=colors_segs, width=2, connect='segments', method='gl', parent=view.scene)
            rungs.append(line)

        # ----- vertices -----
        all_pts = np.vstack((pts1, pts2))
        mark_colors = np.tile(rgba[None,:], (len(all_pts),1))
        mark = Markers(pos=all_pts, face_color=mark_colors, edge_color=(1,1,1,1), size=8, parent=view.scene)
        rungs.append(mark)

        # ----- center & label -----
        cen = all_pts.mean(axis=0)
        centers.append(cen)
        lbl = Text(f"{dim}D: {name}\n{note}", pos=cen+[0,0,0.3], color=rgba, font_size=10, bold=True, anchor_x='center', parent=view.scene)
        labels.append(lbl)

        # ----- echo back to source -----
        if len(emerged) > 1:
            prev = emerged[-2]
            prev_rgba = rgba_for(geometries[prev][2])
            prev_rgba[3] = 0.25
            echo_pts = all_pts * 0.75
            echo_colors = np.tile(prev_rgba[None,:], (len(echo_pts),1))
            echo = Markers(pos=echo_pts, face_color=echo_colors, size=5, parent=view.scene)
            echoes.append(echo)

        # ----- inter-rung links -----
        if len(centers) > 1:
            prev_c = centers[-2]
            segs_link = []
            for i in range(min(6, len(all_pts))):
                segs_link += [prev_c, cen]
            segs_link = np.array(segs_link, dtype=np.float32)
            colors_link = np.tile(np.array([0.7,0.7,0.7,0.4],dtype=np.float32)[None,:], (len(segs_link),1))
            link = Line(pos=segs_link, color=colors_link, width=1, connect='segments', method='gl', parent=view.scene)
            links.append(link)

    # ---- auto-rotate ----
    view.camera.azimuth = frame * 0.3
    view.camera.elevation = 15
    canvas.update()

# ---------- RUN ----------
timer = app.Timer(interval=0.02, connect=update, start=True)
if __name__ == '__main__':
    canvas.show()
    app.run()
