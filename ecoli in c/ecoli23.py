# 🌌 DNA φ-Harmonic Spiral Split-Test (24 Mappings)
# --------------------------------------------------------------
# Visualize all 24 possible DNA → geometry mappings
# Install: pip install vispy pyqt6
# --------------------------------------------------------------

import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color
import itertools

# ---------- FRAMEWORK ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)  # 137.507°

# ---------- GEOMETRIES ----------
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

# ---------- LOAD GENOME ----------
def load_genome(fasta_file):
    seq = []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                continue
            seq.extend(list(line.strip().upper()))
    return seq

genome_seq = load_genome("ecoli_k12.fasta")
genome_len = len(genome_seq)

# ---------- GENERATE ALL 24 DNA → GEOMETRY MAPPINGS ----------
bases = ['A','T','G','C']
geom_indices = [1,2,4,5]  # Choose your primary geometry indices for 1st 4 positions
# Note: pick indices 1..4 if you want to match original 4-mapping behavior
all_perms = list(itertools.permutations([1,2,3,4]))
mappings = [{b: g for b,g in zip(bases, perm)} for perm in all_perms]

# ---------- CANVASES, VIEWS, STRANDS ----------
canvases = []
views = []
strands = []

for i, mapping in enumerate(mappings):
    canvas = scene.SceneCanvas(keys='interactive', size=(400, 300),
                               bgcolor='black', title=f"Mapping {i+1}")
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    canvases.append(canvas)
    views.append(view)

    strand1 = Line(pos=np.zeros((1,3)), color=(1,1,1,0.7), width=2, parent=view.scene)
    strand2 = Line(pos=np.zeros((1,3)), color=(1,1,1,0.7), width=2, parent=view.scene)
    strands.append((strand1, strand2))

# ---------- EMERGENCE/GRAPH COLLECTIONS ----------
collections = [{'rungs': [], 'echoes': [], 'links': [], 'labels': [], 'centers': [], 'emerged': []} for _ in mappings]

frame = 0
core_radius = 15.0
strand_sep = 0.5
twist_factor = 2*np.pi

# ---------- UPDATE LOOP ----------
def update(ev):
    global frame
    frame += 1
    N = 600
    t = np.linspace(0, frame, N)

    for m_idx, base_map in enumerate(mappings):
        s1, s2 = [], []
        view = views[m_idx]
        strand1, strand2 = strands[m_idx]
        coll = collections[m_idx]

        for tt in t:
            idx = int(tt) % genome_len
            base = genome_seq[idx]
            dim = base_map.get(base, 1) - 1

            _, _, col, _, alpha, verts = geometries[dim]
            r = core_radius * (1 - (tt/genome_len)**1.5)
            r = max(r, 0.5)
            theta = tt * np.radians(golden_angle_deg)
            twist = tt/genome_len * twist_factor
            z = np.sin(tt/genome_len * np.pi * 4) * 2 + (tt/genome_len)*8

            a1 = np.radians(angles[dim])
            a2 = np.radians(-angles[dim])

            s1.append([r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                       r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
                       z])
            s2.append([r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
                       r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
                       z])

        # Update strands
        strand1.set_data(np.array(s1))
        strand2.set_data(np.array(s2))

        # ---------- EMERGE RUNGS ----------
        cur_base = genome_seq[frame % genome_len]
        cur_dim = base_map.get(cur_base, 1) - 1

        if frame % 20 == 0 and cur_dim > len(coll['rungs'])-1:
            dim, note, col, name, alpha, verts = geometries[cur_dim]
            coll['emerged'].append(cur_dim)
            start = int((cur_dim/8.0)*N)
            step = max(1, N // (8*verts))
            idxs = np.arange(start, start+verts*step, step)[:verts]
            idxs = idxs[idxs < N]

            pts1 = np.array(s1)[idxs]
            pts2 = np.array(s2)[idxs]
            rgba = Color(col).rgba
            edge_rgba = list(rgba[:3]) + [0.9]

            # Closed lattice
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
                            segs += [pts1[i], pts1[(i+k)%verts],
                                     pts2[i], pts2[(i+k)%verts]]
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
                coll['rungs'].append(line)

            # Vertices
            all_pts = np.vstack((pts1, pts2))
            mark = Markers(pos=all_pts, face_color=rgba, edge_color='white',
                           size=8, parent=view.scene)
            coll['rungs'].append(mark)

            # Center & label
            cen = all_pts.mean(axis=0)
            coll['centers'].append(cen)
            lbl = Text(f"{cur_base}: {name}", pos=cen + [0,0,0.3],
                       color=col, font_size=10, bold=True,
                       anchor_x='center', parent=view.scene)
            coll['labels'].append(lbl)

            # Echo
            if len(coll['emerged']) > 1:
                prev = coll['emerged'][-2]
                prev_col = list(Color(geometries[prev][2]).rgba)
                prev_col[3] = 0.25
                echo_pts = all_pts * 0.75
                echo = Markers(pos=echo_pts, face_color=prev_col,
                               size=5, parent=view.scene)
                coll['echoes'].append(echo)

            # Inter-rung links
            if len(coll['centers']) > 1:
                prev_c = coll['centers'][-2]
                segs = []
                for i in range(min(6, len(all_pts))):
                    segs += [prev_c, cen]
                link = Line(pos=np.array(segs), color=(0.7,0.7,0.7,0.4),
                            width=1, connect='segments', parent=view.scene)
                coll['links'].append(link)

        # Camera rotation
        view.camera.azimuth = frame * 0.3
        view.camera.elevation = 20 + 5*np.sin(frame*0.005)
        canvases[m_idx].update()

# ---------- TIMER ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    for canvas in canvases:
        canvas.show()
    app.run()
