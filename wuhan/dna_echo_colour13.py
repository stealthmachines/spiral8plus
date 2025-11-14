# dna_echo_colour_harmonic.py
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)

geometries = [
    (1, 'C', 'red', 'Point', 0.015269, 1),
    (2, 'D', 'green', 'Line', 0.008262, 2),
    (3, 'E', 'violet', 'Triangle', 0.110649, 3),
    (4, 'F', 'mediumpurple', 'Tetrahedron', -0.083485, 4),
    (5, 'G', 'blue', 'Pentachoron', 0.025847, 5),
    (6, 'A', 'indigo', 'Hexacross', -0.045123, 12),
    (7, 'B', 'purple', 'Heptacube', 0.067891, 14),
    (8, 'C', 'white', 'Octacube', 0.012345, 16),
]

angles = [i * golden_angle_deg for i in range(8)]
period = 13.057
t_max = period * 8
speed_factor = 5.0

canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

strand1 = Line(pos=np.zeros((1, 3)), color=(1,1,1,0.7), width=2, parent=view.scene)
strand2 = Line(pos=np.zeros((1, 3)), color=(1,1,1,0.7), width=2, parent=view.scene)

rungs, echoes, links, labels, centers, emerged = [], [], [], [], [], []

s1_hist, s2_hist, colors_s1_hist, colors_s2_hist = [], [], [], []

frame = 0

def rgba_for(col, alpha=1.0):
    c = np.array(Color(col).rgba, dtype=np.float32)
    c[3] *= alpha
    return c

def harmonic_color(t_norm, shift=0.0):
    """Return smooth interpolated color across all geometry colors with optional phase shift."""
    n_colors = len(geometries)
    idx = (t_norm * n_colors + shift) % n_colors
    low = int(np.floor(idx))
    high = (low + 1) % n_colors
    mix = idx - low
    c1 = rgba_for(geometries[low][2])
    c2 = rgba_for(geometries[high][2])
    return c1 * (1 - mix) + c2 * mix

def update(ev):
    global frame, s1_hist, s2_hist, colors_s1_hist, colors_s2_hist, emerged, centers, rungs, echoes, links, labels
    frame += 1

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
                 r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1), z]
        s2[i] = [r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2),
                 r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2), z]

        t_norm = tt / t[-1]
        hue_shift = frame * 0.002
        colors_s1[i] = harmonic_color(t_norm, shift=hue_shift)
        colors_s2[i] = harmonic_color(t_norm, shift=hue_shift)

    s1_hist.append(s1.copy())
    s2_hist.append(s2.copy())
    colors_s1_hist.append(colors_s1.copy())
    colors_s2_hist.append(colors_s2.copy())

    strand1.set_data(pos=np.vstack(s1_hist), color=np.vstack(colors_s1_hist), width=2)
    strand2.set_data(pos=np.vstack(s2_hist), color=np.vstack(colors_s2_hist), width=2)

    # ---- rungs, vertices, echoes, links, labels (same as before) ----
    cur_dim = min(int((t[-1] * speed_factor) // period), 7)
    if frame % 20 == 0 and (len(emerged) == 0 or cur_dim != emerged[-1]):
        dim, note, col, name, alpha, verts = geometries[cur_dim]
        emerged.append(cur_dim)

        start = int((cur_dim/8.0)*N)
        step  = max(1, N // (8*verts))
        idx   = np.arange(start, start+verts*step, step)[:verts]
        idx   = idx[idx < N]
        pts1 = np.array(s1)[idx]
        pts2 = np.array(s2)[idx]
        rgba = Color(col).rgba
        edge_rgba = list(rgba[:3]) + [0.9]

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
            line = Line(pos=np.array(segs), color=edge_rgba, width=2,
                        connect='segments', parent=view.scene)
            rungs.append(line)

        all_pts = np.vstack((pts1, pts2))
        mark = Markers(pos=all_pts, face_color=rgba, edge_color='white',
                       size=8, parent=view.scene)
        rungs.append(mark)

        cen = all_pts.mean(axis=0)
        centers.append(cen)
        lbl = Text(f"{dim}D: {name}\n{note}", pos=cen + [0,0,0.3],
                   color=col, font_size=10, bold=True, anchor_x='center', parent=view.scene)
        labels.append(lbl)

        if len(emerged) > 1:
            prev = emerged[-2]
            prev_col = list(Color(geometries[prev][2]).rgba)
            prev_col[3] = 0.25
            echo_pts = all_pts * 0.75
            echo = Markers(pos=echo_pts, face_color=prev_col, size=5, parent=view.scene)
            echoes.append(echo)

        if len(centers) > 1:
            prev_c = centers[-2]
            segs = []
            for i in range(min(6, len(all_pts))):
                segs += [prev_c, cen]
            link = Line(pos=np.array(segs), color=(0.7,0.7,0.7,0.4),
                        width=1, connect='segments', parent=view.scene)
            links.append(link)

    view.camera.azimuth   = frame * 0.3
    view.camera.elevation = 15
    canvas.update()

timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    app.run()
