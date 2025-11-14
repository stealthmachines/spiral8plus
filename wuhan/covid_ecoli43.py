# dna_echo_single_cell.py
# --------------------------------------------------------------
# GPU-accelerated φ-spiral chromosome for a single SARS-CoV-2-like cell
# Fully FASTA-driven, recursive echoes, holographic lattice, yin/yang backpressure
# Install: pip install vispy pyqt6 numpy
# Run: python dna_echo_single_cell.py
# --------------------------------------------------------------

import os
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- CONSTANTS ----------
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
core_radius = 15.0
strand_sep = 0.5
twist_factor = 2*np.pi

MAX_RUNG_HISTORY = 8000

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

fasta_path = find_covid_fasta()
if not os.path.exists(fasta_path):
    raise FileNotFoundError(f"FASTA not found: {fasta_path}")

genome_seq = load_genome(fasta_path)
genome_len = len(genome_seq)
print(f"Genome length: {genome_len}")

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='#000011',
                           title="Single SARS-CoV-2 φ-Harmonic Cell")
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Strands
strand1 = Line(pos=np.zeros((1,3)), color=(0,1,1,0.9), width=2, parent=view.scene)
strand2 = Line(pos=np.zeros((1,3)), color=(1,0.5,0,0.9), width=2, parent=view.scene)

# Collections
rungs, echoes, links, labels, centers, organelles = [], [], [], [], [], []

frame = 0

# Progress text
progress_text = Text("0%", pos=[0,0,20], color='white', font_size=24,
                     anchor_x='center', parent=view.scene)

# ---------- HELPERS ----------
def spawn_organelle(center, color_rgb, size=0.5, n=12):
    """Analog/noise-driven organelle cluster"""
    pts = center + np.random.normal(scale=0.2, size=(n,3)) * size
    rgba = list(Color(color_rgb).rgba)
    rgba[3] = 1.0
    mark = Markers(pos=pts, face_color=rgba, edge_color=None, size=5, parent=view.scene)
    return {"marker": mark, "positions": pts, "color": rgba, "size": size}

def lattice_push(pt, lattice_nodes, positive=True, strength=0.02):
    """Yin/Yang backpressure: move point toward/away from lattice nodes"""
    if len(lattice_nodes) == 0:
        return pt
    nearest = lattice_nodes[np.random.randint(0,len(lattice_nodes))]
    dir_vec = nearest - pt
    if not positive:
        dir_vec *= -1
    return pt + dir_vec * strength

def consensus_dim_for_base(b):
    """Deterministic digital lattice coordinate"""
    return base_map.get(b,1) - 1

# ---------- UPDATE FUNCTION ----------
def update(ev):
    global frame, rungs, echoes, links, labels, centers, organelles

    frame += 1
    N = 600
    t = np.linspace(0, frame, N)

    s1, s2 = [], []

    for tt in t:
        idx = int(tt) % genome_len
        base = genome_seq[idx]
        dim = consensus_dim_for_base(base)
        _, _, col, _, alpha, verts = geometries[dim]

        # Radial compression
        r = core_radius * (1 - (tt/genome_len)**1.5)
        r = max(r, 0.5)

        # φ-spiral & twist
        theta = np.radians(golden_angle_deg) * tt
        twist = tt/genome_len * twist_factor
        z = np.sin(tt/genome_len * np.pi*4) * 2 + (tt/genome_len)*8

        # Strands
        a1 = np.radians(angles[dim])
        s1.append([r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                   r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
                   z])
        a2 = np.radians(-angles[dim])
        s2.append([r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
                   r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
                   z])

    strand1.set_data(np.array(s1))
    strand2.set_data(np.array(s2))

    # ---------- Emerge rungs and lattice ----------
    cur_base = genome_seq[frame % genome_len]
    dim = consensus_dim_for_base(cur_base)
    _, _, col, name, alpha, verts = geometries[dim]

    if frame % 20 == 0:
        Npts = len(s1)
        step = max(1, Npts // verts)
        idx_pts = np.arange(0, verts*step, step)[:verts]
        pts1 = np.array(s1)[idx_pts]
        pts2 = np.array(s2)[idx_pts]
        all_pts = np.vstack((pts1, pts2))

        # Rungs
        mark = Markers(pos=all_pts, face_color=Color(col).rgba,
                       edge_color='white', size=8, parent=view.scene)
        rungs.append(mark)

        # Centers
        cen = all_pts.mean(axis=0)
        centers.append(cen)

        # Labels
        lbl = Text(f"{cur_base}:{name}", pos=cen + [0,0,0.3], color=col,
                   font_size=10, bold=True, anchor_x='center', parent=view.scene)
        labels.append(lbl)

        # Echo
        if len(centers) > 1:
            prev = centers[-2]
            echo_pts = all_pts * 0.75 + np.random.normal(scale=0.01, size=all_pts.shape)
            echo = Markers(pos=echo_pts, face_color=(1,1,1,0.25),
                           size=5, parent=view.scene)
            echoes.append(echo)

        # Inter-rung links
        if len(centers) > 1:
            prev_c = centers[-2]
            segs = []
            for p in all_pts[:6]:
                segs += [prev_c, p]
            link = Line(pos=np.array(segs), color=(0.7,0.7,0.7,0.4),
                        width=1, connect='segments', parent=view.scene)
            links.append(link)

        # Spawn organelles with backpressure
        spawn_prob = 0.02 + dim*0.02
        cluster_size = 6 + dim
        if np.random.rand() < spawn_prob:
            org = spawn_organelle(cen, col, size=0.3, n=cluster_size)
            organelles.append(org)

    # ---------- Apply lattice backpressure ----------
    lattice_nodes = np.array(centers)
    for org in organelles:
        positive = cur_base in ['A','T']
        new_pts = np.array([lattice_push(p, lattice_nodes, positive=positive, strength=0.02) for p in org['positions']])
        org['positions'] = new_pts
        org['marker'].set_data(pos=new_pts, face_color=org['color'], size=5)

    # ---------- Camera ----------
    view.camera.azimuth = frame * 0.3
    view.camera.elevation = 20 + 5*np.sin(frame*0.005)

    # ---------- Progress ----------
    percent = min(frame/genome_len*100, 100)
    progress_text.text = f"{percent:.2f}%"

    # Limit history
    if len(rungs) > MAX_RUNG_HISTORY:
        drop = len(rungs)//3
        rungs = rungs[drop:]
        echoes = echoes[drop:]
        links = links[drop:]
        labels = labels[drop:]
        centers = centers[drop:]
        organelles = organelles[drop:]

    canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == "__main__":
    canvas.show()
    app.run()
