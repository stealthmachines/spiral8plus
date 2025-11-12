"""
Generative Multi-SARS-CoV-2 DNA φ-Harmonic Spiral — volumetric, genome-driven cells & division
Requirements: pip install vispy pyqt6
Run: python ecoli_multi.py
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
MAX_ORGANELLES = 200

# ---------- FASTA LOADER ----------

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
            if line.startswith(">"): continue
            seq.extend(list(line.strip().upper()))
    return seq

fasta_path = find_covid_fasta()
if not os.path.exists(fasta_path):
    raise FileNotFoundError(f"FASTA not found: {fasta_path}")
genome_seq = load_genome(fasta_path)
genome_len = len(genome_seq)
print("Genome length:", genome_len)

# ---------- VISPY SCENE ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1400,900),
                           bgcolor='#000011', title="Multi-SARS-CoV-2 Generative Spiral")
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# ---------- Helper Functions ----------
def consensus_dim_for_base(b):
    dim_counts = [m.get(b,1)-1 for m in all_mappings]
    return int(np.round(np.mean(dim_counts)))

def organelle_params_from_base(b):
    d = consensus_dim_for_base(b)
    spawn_prob = 0.02 + (d / 16.0) * 0.12
    cluster_size = 6 + (d % 5) * 3
    radial_bias = 0.2 + (d / 16.0) * 0.9
    noise_scale = 0.08 + (d / 16.0) * 0.4
    return spawn_prob, cluster_size, radial_bias, noise_scale

def spawn_organelle(center, color_rgb, size, n=12):
    pts = center + np.random.normal(scale=0.3, size=(n,3)) * size
    rgba = list(Color(color_rgb).rgba)
    rgba[3] = 0.95
    mark = Markers(pos=pts, face_color=rgba, edge_color='white', size=6, parent=view.scene)
    return {"marker": mark, "age": 0, "life": 1e9}  # persistence effectively infinite

def make_echo_shell(center, scale, color_rgb, alpha=0.12):
    pts = center + (np.random.normal(scale=0.4, size=(max(8, int(12*scale)),3)) * scale)
    rgba = list(Color(color_rgb).rgba)
    rgba[3] = alpha
    shell = Markers(pos=pts, face_color=rgba, edge_color=None, size=3, parent=view.scene)
    return {"marker": shell, "age": 0, "life": 1e9}

# ---------- SARS-CoV-2 Class ----------
class EColi:
    def __init__(self, genome_seq, offset=np.zeros(3), color=(0.5,1.0,0.5)):
        self.genome = genome_seq
        self.offset = np.array(offset)
        self.color = color
        self.frame = 0
        self.accum_s1 = []
        self.accum_s2 = []
        self.organelles = []
        self.echo_shells = []
        self.strand_vis_1 = Line(pos=np.zeros((1,3)), color=color, width=2, parent=view.scene)
        self.strand_vis_2 = Line(pos=np.zeros((1,3)), color=color, width=2, parent=view.scene)

    def update(self, other_lattices=[]):
        self.frame += 1
        idx = self.frame % len(self.genome)
        base = self.genome[idx]
        comp_s1 = np.zeros(3)
        comp_s2 = np.zeros(3)

        # Compute consensus lattice points
        for base_map in all_mappings:
            dim = base_map.get(base, 1) - 1
            r = core_radius * (1 - (idx / len(self.genome))**1.4)
            r = max(r, 0.5)
            theta = idx * np.radians(golden_angle_deg)
            z = np.sin(idx / len(self.genome) * np.pi * 4) * 1.6 + (idx / len(self.genome)) * 6.0
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

        # Lattice push from other bacteria
        for other in other_lattices:
            other_nodes = np.array(other.accum_s1 + other.accum_s2)
            if other_nodes.size > 0:
                delta = np.mean(other_nodes, axis=0) - ((comp_s1 + comp_s2)/2)
                comp_s1 += 0.02 * delta
                comp_s2 += 0.02 * delta

        # Random lattice jitter
        comp_s1 += np.random.normal(scale=0.02, size=3)
        comp_s2 += np.random.normal(scale=0.02, size=3)

        # Division drift
        progress = idx / max(1,len(self.genome))
        drift_amp = 0.02 + 0.8*progress
        z_bias = np.sin(self.frame * 0.003 + progress * 10.0) * 0.5
        comp_s1[0] += drift_amp*np.cos(self.frame*0.005)
        comp_s2[0] -= drift_amp*np.cos(self.frame*0.005)
        comp_s1[2] += z_bias
        comp_s2[2] -= z_bias*0.6

        # Store lattice
        self.accum_s1.append(comp_s1 + self.offset)
        self.accum_s2.append(comp_s2 + self.offset)

        # Spawn organelles
        spawn_prob, cluster_size, radial_bias, noise_scale = organelle_params_from_base(base)
        if np.random.rand() < spawn_prob:
            center = (comp_s1 + comp_s2)/2 + self.offset
            dim = consensus_dim_for_base(base)
            color_rgb = geometries[min(dim,len(geometries)-1)][2]
            self.organelles.append(spawn_organelle(center, color_rgb, size=0.8+radial_bias))
            self.echo_shells.append(make_echo_shell(center, scale=0.6+radial_bias, color_rgb=color_rgb))

        # Update visuals
        self.strand_vis_1.set_data(np.array(self.accum_s1))
        self.strand_vis_2.set_data(np.array(self.accum_s2))

# ---------- MULTI-E.COLI ENVIRONMENT ----------
num_bacteria = 3
offsets = [np.array([i*30.0,0,0]) for i in range(num_bacteria)]
bacteria = [EColi(genome_seq, offset=o) for o in offsets]

def update(ev):
    for i, bac in enumerate(bacteria):
        others = [b for j,b in enumerate(bacteria) if j!=i]
        bac.update(other_lattices=others)
    # Slowly rotate camera
    view.camera.azimuth += 0.12
    view.camera.elevation = 15 + 8*np.sin(bacteria[0].frame*0.002)
    canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == "__main__":
    canvas.show()
    app.run()
