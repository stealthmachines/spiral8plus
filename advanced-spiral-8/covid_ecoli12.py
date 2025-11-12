# dna_echo_genome_fullcell.py
# --------------------------------------------------------------
# GPU-accelerated double φ-spiral encoding an entire genome
# coloured by nucleotide, tightly packed, with closed lattices,
# inter-links, echoes, final φ-core convergence, and % complete.
# Install: pip install vispy pyqt6
# --------------------------------------------------------------

import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- FRAMEWORK ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)  # 137.507°

# nucleotide → geometry mapping
base_map = {'A': 5, 'T': 2, 'G': 4, 'C': 1}

# geometries: (dim, note, color, name, alpha, vertices)
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

# ---------- AUTO-DETECT AND LOAD GENOME ----------
def find_covid_fasta():
    """Automatically find the COVID-19 FASTA file in the ncbi_dataset"""
    import glob
    possible_paths = [
        r"ncbi_dataset\data\GCF_009858895.2\*.fna",
        r"ncbi_dataset\data\GCA_009858895.3\*.fna",
        r"ncbi_dataset\data\*\*.fna",
    ]

    for pattern in possible_paths:
        files = glob.glob(pattern)
        if files:
            return files[0]

    raise FileNotFoundError("Could not find COVID-19 FASTA file in ncbi_dataset directory")


def load_genome(fasta_file):
    seq = []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                continue
            seq.extend(list(line.strip().upper()))
    return seq

genome_seq = load_genome(find_covid_fasta())
genome_len = len(genome_seq)

# ---------- VISPY ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='#000011')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# strands
strand1 = Line(pos=np.zeros((1,3)), color=(1,1,1,0.7), width=2, parent=view.scene)
strand2 = Line(pos=np.zeros((1,3)), color=(1,1,1,0.7), width=2, parent=view.scene)

# collections
rungs, echoes, links, labels, centers, emerged = [], [], [], [], [], []

# ---------- CELL PARAMETERS ----------
frame = 0
core_radius = 15.0      # central φ-core
strand_sep = 0.5        # small separation between strands
twist_factor = 2*np.pi  # total twist over genome length
membrane_radius = core_radius + 5.0

# ---------- PERCENT COMPLETE TEXT ----------
percent_text = Text("0%", pos=[0, 0, 10], color='white', font_size=16, bold=True,
                    anchor_x='center', parent=view.scene)

# ---------- UPDATE FUNCTION ----------
def update(ev):
    global frame, rungs, echoes, links, labels, centers, emerged

    frame += 1
    N = 600
    t = np.linspace(0, frame, N)

    s1, s2 = [], []

    # ---------- CHROMOSOME PACKING ----------
    for tt in t:
        idx = int(tt) % genome_len
        base = genome_seq[idx]
        dim = base_map.get(base, 1) - 1
        _, _, col, _, alpha, verts = geometries[dim]

        # radial compression toward φ-core
        r = core_radius * (1 - (tt/genome_len)**1.5)
        r = max(r, 0.5)

        # 3D twisting along genome
        theta = tt * np.radians(golden_angle_deg)
        twist = tt / genome_len * twist_factor
        z = np.sin(tt/genome_len * np.pi * 4) * 2 + (tt/genome_len)*8

        # strand 1
        a1 = np.radians(angles[dim])
        pt1 = [
            r*np.cos(theta)*np.cos(a1) - r*np.sin(theta)*np.sin(a1),
            r*np.sin(theta)*np.cos(a1) + r*np.cos(theta)*np.sin(a1),
            z
        ]
        # strand 2
        a2 = np.radians(-angles[dim])
        pt2 = [
            r*np.cos(theta)*np.cos(a2) - r*np.sin(theta)*np.sin(a2) + strand_sep,
            r*np.sin(theta)*np.cos(a2) + r*np.cos(theta)*np.sin(a2) - strand_sep,
            z
        ]

        # enforce membrane boundary
        for pt in (pt1, pt2):
            dist = np.linalg.norm(pt)
            if dist > membrane_radius:
                pt *= (membrane_radius / dist)

        s1.append(pt1)
        s2.append(pt2)

    # ---------- UPDATE STRANDS ----------
    strand1.set_data(np.array(s1))
    strand2.set_data(np.array(s2))

    # ---------- EMERGE RUNGS ----------
    cur_base = genome_seq[frame % genome_len]
    cur_dim = base_map.get(cur_base, 1) - 1
    if frame % 20 == 0 and cur_dim > len(rungs)-1:
        dim, note, col, name, alpha, verts = geometries[cur_dim]
        emerged.append(cur_dim)

        start = int((cur_dim/8.0)*N)
        step = max(1, N // (8*verts))
        idx = np.arange(start, start+verts*step, step)[:verts]
        idx = idx[idx < N]

        pts1 = np.array(s1)[idx]
        pts2 = np.array(s2)[idx]
        rgba = Color(col).rgba
        edge_rgba = list(rgba[:3]) + [0.9]

        # closed lattice
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
            rungs.append(line)

        # vertices
        all_pts = np.vstack((pts1, pts2))
        mark = Markers(pos=all_pts, face_color=rgba, edge_color='white',
                       size=8, parent=view.scene)
        rungs.append(mark)

        # center & label
        cen = all_pts.mean(axis=0)
        centers.append(cen)
        lbl = Text(f"{cur_base}: {name}", pos=cen + [0,0,0.3],
                   color=col, font_size=10, bold=True,
                   anchor_x='center', parent=view.scene)
        labels.append(lbl)

        # echo
        if len(emerged) > 1:
            prev = emerged[-2]
            prev_col = list(Color(geometries[prev][2]).rgba)
            prev_col[3] = 0.25
            echo_pts = all_pts * 0.75
            echo = Markers(pos=echo_pts, face_color=prev_col,
                           size=5, parent=view.scene)
            echoes.append(echo)

        # inter-rung links
        if len(centers) > 1:
            prev_c = centers[-2]
            segs = []
            for i in range(min(6, len(all_pts))):
                segs += [prev_c, cen]
            link = Line(pos=np.array(segs), color=(0.7,0.7,0.7,0.4),
                        width=1, connect='segments', parent=view.scene)
            links.append(link)

    # ---------- PERCENT COMPLETE ----------
    percent = min(frame/genome_len * 100, 100)
    percent_text.text = f"{percent:.1f}%"

    # ---------- CAMERA ----------
    view.camera.azimuth = frame * 0.3
    view.camera.elevation = 20 + 5*np.sin(frame*0.005)
    canvas.update()

# ---------- TIMER ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    app.run()
