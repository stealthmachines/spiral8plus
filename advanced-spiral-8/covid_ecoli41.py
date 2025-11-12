"""
Generative DNA φ-Harmonic Spiral — holographic, genome-driven cells & lattice movement
Requirements: pip install vispy pyqt6
Run: python ecoli_holographic.py
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

core_radius = 15.0
strand_sep = 0.6
MAX_POINTS = 8000
MAX_ORGANELLES = 500
NUM_BACTERIA = 12  # multiple holographic spirals

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
                           bgcolor='#000011', title="Holographic DNA Spirals (FASTA-driven)")
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# ---------- BACTERIA CLASS ----------
class EColi:
    def __init__(self, genome_seq, offset=np.zeros(3)):
        self.genome_seq = genome_seq
        self.offset = np.array(offset)
        self.frame = 0
        self.accum_s1 = []
        self.accum_s2 = []
        self.accum_donut = []
        self.organelles = []

        self.strand1_vis = Line(pos=np.zeros((1,3)), color=(0,1,1,0.7), width=2, parent=view.scene)
        self.strand2_vis = Line(pos=np.zeros((1,3)), color=(1,0.5,0,0.7), width=2, parent=view.scene)
        self.donut_vis   = Line(pos=np.zeros((1,3)), color=(1,1,0,0.4), width=1.5, parent=view.scene)

    def consensus_dim_for_base(self, b):
        return int(np.round(np.mean([m.get(b,1)-1 for m in all_mappings])))

    def organelle_params_from_base(self, b):
        d = self.consensus_dim_for_base(b)
        spawn_prob = 0.02 + (d / 16.0) * 0.12
        cluster_size = 6 + (d % 5) * 3
        radial_bias = 0.2 + (d / 16.0) * 0.9
        noise_scale = 0.08 + (d / 16.0) * 0.4
        return spawn_prob, cluster_size, radial_bias, noise_scale

    def spawn_organelle(self, center, color_rgb, size=0.5, n=12):
        pts = center + np.random.normal(scale=0.2, size=(n,3)) * size
        rgba = list(Color(color_rgb).rgba)
        rgba[3] = 1.0
        mark = Markers(pos=pts, face_color=rgba, edge_color=None, size=5, parent=view.scene)
        return {"marker": mark, "positions": pts, "size": size, "color": rgba}

    def lattice_push(self, pt, lattice_nodes, positive=True, strength=0.02):
        if len(lattice_nodes) == 0:
            return pt
        nearest = lattice_nodes[np.random.randint(0,len(lattice_nodes))]
        dir_vec = nearest - pt
        if not positive:
            dir_vec *= -1
        return pt + dir_vec * strength

    def update(self, others=[]):
        self.frame += 1
        idx = self.frame % len(self.genome_seq)
        base = self.genome_seq[idx]

        # Holographic DNA: every point carries full genome signal
        comp_s1 = np.zeros(3)
        comp_s2 = np.zeros(3)
        for base_map in all_mappings:
            dim = base_map.get(base,1)-1
            r = core_radius * (1 - (idx/len(self.genome_seq))**1.4)
            r = max(r,0.5)
            theta = np.radians(golden_angle_deg) * idx
            z = np.sin(idx/len(self.genome_seq) * np.pi*4) * 1.6 + (idx/len(self.genome_seq))*6.0
            a1 = np.radians(dim * golden_angle_deg)
            a2 = np.radians(-dim * golden_angle_deg)

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
        comp_s1 += self.offset
        comp_s2 += self.offset

        # Yin/Yang holographic drift
        progress = idx/len(self.genome_seq)
        drift_amp = 0.02 + 0.8*progress
        z_bias = np.sin(self.frame*0.003 + progress*10.0)*0.5
        comp_s1[0] += drift_amp * np.cos(self.frame*0.005)
        comp_s2[0] -= drift_amp * np.cos(self.frame*0.005)
        comp_s1 += np.random.normal(scale=0.003, size=3)
        comp_s2 += np.random.normal(scale=0.003, size=3)
        comp_s1[2] += z_bias
        comp_s2[2] -= z_bias*0.6

        self.accum_s1.append(comp_s1.copy())
        self.accum_s2.append(comp_s2.copy())
        donut_center = (comp_s1 + comp_s2)/2.0
        self.accum_donut.append(donut_center.copy())

        if len(self.accum_s1) > MAX_POINTS:
            drop = len(self.accum_s1)//3
            self.accum_s1 = self.accum_s1[drop:]
            self.accum_s2 = self.accum_s2[drop:]
            self.accum_donut = self.accum_donut[drop:]

        # Spawn organelles
        spawn_prob, cluster_size, radial_bias, noise_scale = self.organelle_params_from_base(base)
        if np.random.rand() < spawn_prob:
            color_rgb = geometries[min(self.consensus_dim_for_base(base), len(geometries)-1)][2]
            org = self.spawn_organelle(donut_center, color_rgb, size=0.8+radial_bias, n=cluster_size)
            self.organelles.append(org)

        # Lattice push (including other bacteria)
        lattice_nodes = np.array(self.accum_s1 + self.accum_s2)
        for other in others:
            lattice_nodes = np.vstack([lattice_nodes, np.array(other.accum_s1 + other.accum_s2)]) if other.accum_s1 else lattice_nodes

        for org in self.organelles:
            positive = base in ['A','T']
            new_pts = np.array([self.lattice_push(p, lattice_nodes, positive=positive, strength=0.02) for p in org['positions']])
            org['positions'] = new_pts
            org['marker'].set_data(pos=new_pts, face_color=org['color'], size=5)

        # Update visuals
        self.strand1_vis.set_data(np.array(self.accum_s1))
        self.strand2_vis.set_data(np.array(self.accum_s2))
        if len(self.accum_donut)>1:
            self.donut_vis.set_data(np.array(self.accum_donut))


# ---------- INIT BACTERIA ----------
bacteria = [EColi(genome_seq, offset=np.random.uniform(-5,5,3)) for _ in range(NUM_BACTERIA)]

# ---------- UPDATE LOOP ----------
def update(ev):
    for bac in bacteria:
        others = [b for b in bacteria if b is not bac]
        bac.update(others=others)
    t = app.get_time()
    view.camera.azimuth = t*12
    view.camera.elevation = 15 + 8*np.sin(t*0.2)
    canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == "__main__":
    canvas.show()
    app.run()
