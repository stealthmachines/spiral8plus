# 🌌 DNA φ-Harmonic Spiral Composite & Split-Test (24 mappings)
# --------------------------------------------------------------
# Full φ-harmonic spiral with rung emergence, echoes, and inter-rung links
# Install: pip install vispy pyqt6
# --------------------------------------------------------------

import numpy as np
import itertools
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- FRAMEWORK ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)  # 137.507°

bases = ['A','T','G','C']
all_mappings = [dict(zip(bases,p)) for p in itertools.permutations([1,2,3,4])]
print(f"Total mappings: {len(all_mappings)}")  # 24

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
core_radius = 15.0
strand_sep = 0.5
twist_factor = 2*np.pi
period = 13.057
N = 600

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

# ---------- COMPOSITE CANVAS ----------
comp_canvas = scene.SceneCanvas(keys='interactive', size=(900,700),
                                bgcolor='black', title="Composite DNA Spiral")
comp_view = comp_canvas.central_widget.add_view()
comp_view.camera = 'turntable'

comp_strand1 = Line(pos=np.zeros((1,3)), color=(0,1,1,0.7), width=2, parent=comp_view.scene)
comp_strand2 = Line(pos=np.zeros((1,3)), color=(1,0.5,0,0.7), width=2, parent=comp_view.scene)

collections = {
    'rungs': [], 'echoes': [], 'links': [], 'labels': [], 'centers': [], 'emerged': []
}

frame = 0

# ---------- UPDATE LOOP ----------
def update(ev):
    global frame, collections
    frame += 1
    t = np.linspace(0, frame, N)

    composite_s1 = np.zeros((N,3))
    composite_s2 = np.zeros((N,3))

    for base_map in all_mappings:
        s1, s2 = [], []
        for tt in t:
            idx = int(tt) % genome_len
            base = genome_seq[idx]
            dim = base_map.get(base, 1) - 1
            _, _, col, _, alpha, verts = geometries[dim]

            r = core_radius * (1 - (tt/genome_len)**1.5)
            r = max(r, 0.5)
            theta = tt * np.radians(golden_angle_deg)
            z = np.sin(tt/genome_len * np.pi * 4) * 2 + (tt/genome_len)*8
            a1 = np.radians(angles[dim])
            a2 = np.radians(-angles[dim])

            s1.append([r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                       r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
                       z])
            s2.append([r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
                       r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
                       z])

        s1_arr = np.array(s1)
        s2_arr = np.array(s2)
        composite_s1 += s1_arr
        composite_s2 += s2_arr

    # Average all 24 mappings
    composite_s1 /= len(all_mappings)
    composite_s2 /= len(all_mappings)

    # Update strands
    comp_strand1.set_data(composite_s1)
    comp_strand2.set_data(composite_s2)

    # ---------- EMERGE RUNGS ----------
    cur_base = genome_seq[frame % genome_len]
    cur_dim = 0
    # Compute a simple consensus dim across all mappings
    dim_counts = [map.get(cur_base,1)-1 for map in all_mappings]
    cur_dim = int(np.round(np.mean(dim_counts)))

    if frame % 20 == 0 and cur_dim > len(collections['rungs'])-1:
        dim, note, col, name, alpha, verts = geometries[cur_dim]
        collections['emerged'].append(cur_dim)

        start = int((cur_dim/8.0)*N)
        step = max(1, N // (8*verts))
        idxs = np.arange(start, start+verts*step, step)[:verts]
        idxs = idxs[idxs < N]

        pts1 = composite_s1[idxs]
        pts2 = composite_s2[idxs]
        rgba = Color(col).rgba
        edge_rgba = list(rgba[:3]) + [0.9]

        # Closed lattice
        segs = []
        for pts in (pts1, pts2):
            for i in range(verts):
                for j in range(i+1, verts):
                    segs += [pts[i], pts[j]]
        for i in range(verts):
            segs += [pts1[i], pts2[i]]

        if segs:
            line = Line(pos=np.array(segs), color=edge_rgba, width=2,
                        connect='segments', parent=comp_view.scene)
            collections['rungs'].append(line)

        all_pts = np.vstack((pts1, pts2))
        mark = Markers(pos=all_pts, face_color=rgba, edge_color='white',
                       size=8, parent=comp_view.scene)
        collections['rungs'].append(mark)

        cen = all_pts.mean(axis=0)
        collections['centers'].append(cen)
        lbl = Text(f"{cur_base}: {name}", pos=cen + [0,0,0.3],
                   color=col, font_size=10, bold=True,
                   anchor_x='center', parent=comp_view.scene)
        collections['labels'].append(lbl)

        # Echo
        if len(collections['emerged']) > 1:
            prev = collections['emerged'][-2]
            prev_col = list(Color(geometries[prev][2]).rgba)
            prev_col[3] = 0.25
            echo_pts = all_pts * 0.75
            echo = Markers(pos=echo_pts, face_color=prev_col,
                           size=5, parent=comp_view.scene)
            collections['echoes'].append(echo)

        # Inter-rung links
        if len(collections['centers']) > 1:
            prev_c = collections['centers'][-2]
            segs = []
            for i in range(min(6,len(all_pts))):
                segs += [prev_c, cen]
            link = Line(pos=np.array(segs), color=(0.7,0.7,0.7,0.4),
                        width=1, connect='segments', parent=comp_view.scene)
            collections['links'].append(link)

    # Camera rotation
    comp_view.camera.azimuth = frame*0.2
    comp_view.camera.elevation = 20 + 5*np.sin(frame*0.005)
    comp_canvas.update()

# ---------- TIMER ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == "__main__":
    comp_canvas.show()
    app.run()
