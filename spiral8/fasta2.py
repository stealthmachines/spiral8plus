# dna_holo_cell.py
# --------------------------------------------------------------
# GPU-accelerated φ-spiral, genome-driven E. coli-like cell
# Holographic lattice, yin-yang dynamics, and genome-triggered division
# Install: pip install vispy pyqt6 numpy
# Run: python dna_holo_cell.py
# --------------------------------------------------------------

import os
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- CONSTANTS ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)  # 137.507°
core_radius = 15.0
strand_sep = 0.5
MAX_POINTS = 8000

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

# ---------- LOAD GENOME ----------
def load_genome(fasta_file):
    seq = []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"): continue
            seq.extend(list(line.strip().upper()))
    return seq

fasta_path = "ecoli_k12.fasta"
if not os.path.exists(fasta_path):
    raise FileNotFoundError(f"FASTA not found: {fasta_path}")
genome_seq = load_genome(fasta_path)
genome_len = len(genome_seq)
print(f"Genome length: {genome_len}")

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800),
                           bgcolor='black', title="Holographic DNA Cell (FASTA-driven)")
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# strands & donut
strand1_vis = Line(pos=np.zeros((1,3)), color=(0,1,1,0.9), width=2, parent=view.scene)
strand2_vis = Line(pos=np.zeros((1,3)), color=(1,0.5,0,0.9), width=2, parent=view.scene)
donut_vis   = Line(pos=np.zeros((1,3)), color=(1,1,0,0.5), width=1.5, parent=view.scene)

# collections
organelles = []
labels = []

accum_s1 = []
accum_s2 = []
accum_donut = []

# division tracking
cells = [{"accum_s1": [], "accum_s2": [], "accum_donut": [], "organelles": [], "labels": [], "genome_start": 0}]
frame = 0

# ---------- HELPERS ----------
def consensus_dim_for_base(b):
    return base_map.get(b,1)-1

def organelle_params_from_base(b):
    d = consensus_dim_for_base(b)
    spawn_prob = 0.02 + (d / 16.0) * 0.12
    cluster_size = 6 + (d % 5) * 3
    radial_bias = 0.2 + (d / 16.0) * 0.9
    noise_scale = 0.08 + (d / 16.0) * 0.4
    return spawn_prob, cluster_size, radial_bias, noise_scale

def spawn_organelle(center, color_rgb, size=0.5, n=12):
    pts = center + np.random.normal(scale=0.2, size=(n,3)) * size
    rgba = list(Color(color_rgb).rgba)
    rgba[3] = 1.0
    mark = Markers(pos=pts, face_color=rgba, edge_color=None, size=5, parent=view.scene)
    return {"marker": mark, "positions": pts, "size": size, "color": rgba}

def lattice_push(pt, lattice_nodes, positive=True, strength=0.02):
    if len(lattice_nodes) == 0:
        return pt
    nearest = lattice_nodes[np.random.randint(0,len(lattice_nodes))]
    dir_vec = nearest - pt
    if not positive:
        dir_vec *= -1
    return pt + dir_vec * strength

def division_motif(seq_window):
    # define a simple motif: consecutive "ATG" triplet triggers division
    return seq_window == ['A','T','G']

# ---------- UPDATE LOOP ----------
def update(ev):
    global frame, cells
    frame += 1
    new_cells = []

    for cell in cells:
        idx = (cell["genome_start"] + frame) % genome_len
        base = genome_seq[idx]

        # lattice computation
        comp_s1 = np.zeros(3)
        comp_s2 = np.zeros(3)
        dim = consensus_dim_for_base(base)
        r = core_radius * (1 - ((idx)/genome_len)**1.4)
        r = max(r, 0.5)
        theta = np.radians(golden_angle_deg) * idx
        z = np.sin(idx/genome_len * np.pi*4) * 1.6 + (idx/genome_len)*6.0
        a1 = np.radians(angles[dim])
        a2 = np.radians(-angles[dim])
        p1 = np.array([r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                       r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
                       z])
        p2 = np.array([r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
                       r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
                       z])
        comp_s1 += p1
        comp_s2 += p2

        # drift/noise
        comp_s1[0] += 0.02 * np.cos(frame*0.005)
        comp_s2[0] -= 0.02 * np.cos(frame*0.005)
        comp_s1 += np.random.normal(scale=0.003, size=3)
        comp_s2 += np.random.normal(scale=0.003, size=3)
        comp_s1[2] += np.sin(frame*0.003)*0.5
        comp_s2[2] -= np.sin(frame*0.003)*0.3

        cell["accum_s1"].append(comp_s1.copy())
        cell["accum_s2"].append(comp_s2.copy())
        donut_center = (comp_s1 + comp_s2)/2.0
        cell["accum_donut"].append(donut_center.copy())

        # spawn organelles
        spawn_prob, cluster_size, radial_bias, noise_scale = organelle_params_from_base(base)
        if np.random.rand() < spawn_prob:
            color_rgb = geometries[min(dim, len(geometries)-1)][2]
            org = spawn_organelle(donut_center, color_rgb, size=0.8+radial_bias, n=cluster_size)
            cell["organelles"].append(org)

        # lattice push
        lattice_nodes = np.array(cell["accum_s1"] + cell["accum_s2"])
        for org in cell["organelles"]:
            positive = base in ['A','T']
            new_pts = np.array([lattice_push(p, lattice_nodes, positive=positive, strength=0.02) for p in org['positions']])
            org['positions'] = new_pts
            org['marker'].set_data(pos=new_pts, face_color=org['color'], size=5)

        # update visuals
        strand1_vis.set_data(np.array(cell["accum_s1"]))
        strand2_vis.set_data(np.array(cell["accum_s2"]))
        if len(cell["accum_donut"]) > 1:
            donut_vis.set_data(np.array(cell["accum_donut"]))

        # genome-driven division
        window = genome_seq[idx:idx+3]
        if len(window)==3 and division_motif(window):
            # split cell
            daughter = {
                "accum_s1": cell["accum_s1"][-10:].copy(),  # start from last few points
                "accum_s2": cell["accum_s2"][-10:].copy(),
                "accum_donut": cell["accum_donut"][-10:].copy(),
                "organelles": [],
                "labels": [],
                "genome_start": idx
            }
            new_cells.append(daughter)

    cells.extend(new_cells)

    # rotate camera
    view.camera.azimuth = frame*0.12
    view.camera.elevation = 15 + 8*np.sin(frame*0.002)
    canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)
if __name__ == "__main__":
    canvas.show()
    app.run()
