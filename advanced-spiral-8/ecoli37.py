"""
Generative DNA φ-Harmonic Spiral — volumetric, genome-driven cells & lattice movement
Requirements: pip install vispy pyqt6
Run: python ecoli37.py
"""

import os
import numpy as np
import itertools
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- CONFIG ----------
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
strand_sep = 0.6

MAX_POINTS = 8000
MAX_ORGANELLES = 500

# ---------- FASTA LOADER ----------
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
print("Genome length:", genome_len)

# ---------- VISPY SCENE ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200,800),
                           bgcolor='black', title="Generative DNA Cells (FASTA-driven)")
view = canvas.central_widget.add_view()
view.camera = 'turntable'

strand1_vis = Line(pos=np.zeros((1,3)), color=(0,1,1,0.9), width=2, parent=view.scene)
strand2_vis = Line(pos=np.zeros((1,3)), color=(1,0.5,0,0.9), width=2, parent=view.scene)
donut_vis   = Line(pos=np.zeros((1,3)), color=(1,1,0,0.5), width=1.5, parent=view.scene)

organelles = []
labels = []

accum_s1 = []
accum_s2 = []
accum_donut = []

frame = 0

# ---------- HELPERS ----------
def consensus_dim_for_base(b):
    return int(np.round(np.mean([m.get(b,1)-1 for m in all_mappings])))

def organelle_params_from_base(b):
    d = consensus_dim_for_base(b)
    spawn_prob = 0.02 + (d / 16.0) * 0.12
    cluster_size = 6 + (d % 5) * 3
    radial_bias = 0.2 + (d / 16.0) * 0.9
    noise_scale = 0.08 + (d / 16.0) * 0.4
    return spawn_prob, cluster_size, radial_bias, noise_scale

def spawn_organelle(center, color_rgb, size=0.5, n=12):
    pts = np.tile(center, (n,1))
    rgba = list(Color(color_rgb).rgba)
    rgba[3] = 1.0
    mark = Markers(pos=pts, face_color=rgba, edge_color=None, size=5, parent=view.scene)
    return {"marker": mark, "positions": pts, "size": size, "color": rgba}

# Lattice push: FASTA yin/yang positive/negative
def lattice_push(pt, lattice_nodes, positive=True, strength=0.02):
    if len(lattice_nodes) == 0:
        return pt
    nearest = lattice_nodes[np.random.randint(0,len(lattice_nodes))]
    dir_vec = nearest - pt
    if not positive:
        dir_vec *= -1
    return pt + dir_vec * strength

# ---------- UPDATE LOOP ----------
def update(ev):
    global frame, accum_s1, accum_s2, accum_donut, organelles

    frame += 1
    idx = frame % genome_len
    base = genome_seq[idx]

    # Compute genomic-driven points
    comp_s1 = np.zeros(3)
    comp_s2 = np.zeros(3)
    for base_map in all_mappings:
        dim = base_map.get(base,1)-1
        r = core_radius * (1 - (idx/genome_len)**1.4)
        r = max(r,0.5)
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

    comp_s1 /= len(all_mappings)
    comp_s2 /= len(all_mappings)

    # Drift + anticorrelated noise (structural + harmonic)
    progress = idx/genome_len
    drift_amp = 0.02 + 0.8*progress
    z_bias = np.sin(frame*0.003 + progress*10.0)*0.5
    comp_s1[0] += drift_amp * np.cos(frame*0.005)
    comp_s2[0] -= drift_amp * np.cos(frame*0.005)
    comp_s1[2] += z_bias
    comp_s2[2] -= z_bias*0.6

    accum_s1.append(comp_s1.copy())
    accum_s2.append(comp_s2.copy())

    donut_center = (comp_s1 + comp_s2)/2.0
    accum_donut.append(donut_center.copy())

    if len(accum_s1) > MAX_POINTS:
        drop = len(accum_s1)//3
        accum_s1 = accum_s1[drop:]
        accum_s2 = accum_s2[drop:]
        accum_donut = accum_donut[drop:]

    # Spawn organelles based on FASTA
    spawn_prob, cluster_size, radial_bias, noise_scale = organelle_params_from_base(base)
    if np.random.rand() < spawn_prob:
        color_rgb = geometries[min(consensus_dim_for_base(base), len(geometries)-1)][2]
        org = spawn_organelle(donut_center, color_rgb, size=0.8+radial_bias, n=cluster_size)
        organelles.append(org)

    # Push organelles along lattice (FASTA yin/yang)
    lattice_nodes = np.array(accum_s1 + accum_s2)
    for org in organelles:
        positive = base in ['A','T']  # positive/negative polarity
        new_pts = np.array([lattice_push(p, lattice_nodes, positive=positive, strength=0.02) for p in org['positions']])
        org['positions'] = new_pts
        org['marker'].set_data(pos=new_pts, face_color=org['color'], size=5)

    # Update strands & donut
    strand1_vis.set_data(np.array(accum_s1))
    strand2_vis.set_data(np.array(accum_s2))
    if len(accum_donut)>1:
        donut_vis.set_data(np.array(accum_donut))

    # Labels
    if frame % 160 == 0:
        dim = consensus_dim_for_base(base)
        col = geometries[min(dim, len(geometries)-1)][2]
        lbl = Text(f"{base}:{geometries[min(dim,7)][3]}", pos=donut_center+[0,0,0.25],
                   color=col, font_size=9, bold=True, parent=view.scene)
        labels.append(lbl)

    # Rotate camera
    view.camera.azimuth = frame*0.12
    view.camera.elevation = 15 + 8*np.sin(frame*0.002)
    canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == "__main__":
    canvas.show()
    app.run()
