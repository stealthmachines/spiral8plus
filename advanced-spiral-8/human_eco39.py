"""
Generative DNA φ-Harmonic Spiral — volumetric, genome-driven multiple cells
Requirements: pip install vispy pyqt6
Run: python eco_multi.py
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
NUM_BACTERIA = 12  # dozens, adjust as desired

# ---------- FASTA LOADER ----------

# ---------- AUTO-DETECT AND LOAD GENOME ----------
def find_human_fasta():
    """Automatically find the Human Genome FASTA file in the ncbi_dataset"""
    import glob
    possible_paths = [
        r"ncbi_dataset\ncbi_dataset\data\GCF_000001405.40\*.fna",
        r"ncbi_dataset\ncbi_dataset\data\GCA_000001405.29\*.fna",
        r"ncbi_dataset\ncbi_dataset\data\*\*.fna",
    ]

    for pattern in possible_paths:
        files = glob.glob(pattern)
        if files:
            return files[0]

    raise FileNotFoundError("Could not find Human Genome FASTA file in ncbi_dataset directory")


def load_genome(fasta_file, max_nucleotides=None, chromosome=None, start_position=0):
    """
    Load genome sequence from FASTA file with environment variable support

    Args:
        fasta_file: Path to FASTA file
        max_nucleotides: Maximum nucleotides to load (None = use GENOME_LIMIT env var, default 100000)
        chromosome: Specific chromosome to load (None = use GENOME_CHROMOSOME env var)
        start_position: Starting position in sequence (default 0, or GENOME_START env var)

    Returns:
        str: Genome sequence
    """
    import os

    # Get from environment if not specified
    if max_nucleotides is None:
        env_limit = os.environ.get('GENOME_LIMIT', '100000')
        if env_limit == 'all':
            max_nucleotides = None  # Load full genome
        else:
            try:
                max_nucleotides = int(env_limit)
            except ValueError:
                max_nucleotides = 100000

    if chromosome is None:
        chromosome = os.environ.get('GENOME_CHROMOSOME', None)

    if start_position == 0:
        env_start = os.environ.get('GENOME_START', '0')
        try:
            start_position = int(env_start)
        except ValueError:
            start_position = 0

    sequence = ""
    current_chromosome = None
    nucleotide_count = 0
    position_in_chromosome = 0
    skip_until_start = start_position > 0

    print(f"Loading genome from {fasta_file}...")
    if chromosome:
        print(f"  Filtering: Chromosome {chromosome}")
    if start_position > 0:
        print(f"  Starting at position: {start_position:,}")
    print(f"  Limit: {max_nucleotides:,} nucleotides")

    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith(">"):
                # New chromosome header
                header = line.strip()[1:].split()[0]
                current_chromosome = header
                position_in_chromosome = 0

                # If we're filtering by chromosome and this isn't it, skip
                if chromosome and current_chromosome != chromosome:
                    continue

                # Reset skip flag for new chromosome
                if chromosome and current_chromosome == chromosome:
                    skip_until_start = start_position > 0
                    print(f"Loading from {current_chromosome}...")
            else:
                # If filtering by chromosome and this isn't it, skip
                if chromosome and current_chromosome != chromosome:
                    continue

                bases = line.strip()

                # Handle start position skipping
                if skip_until_start:
                    if position_in_chromosome + len(bases) <= start_position:
                        position_in_chromosome += len(bases)
                        continue
                    else:
                        # Start is within this line
                        offset = start_position - position_in_chromosome
                        bases = bases[offset:]
                        position_in_chromosome = start_position
                        skip_until_start = False
                        print(f"  Started at position {start_position:,}")

                position_in_chromosome += len(bases)

                # Add nucleotides up to limit
                remaining = max_nucleotides - nucleotide_count

                if remaining <= 0:
                    break

                sequence += bases[:remaining]
                nucleotide_count += len(bases[:remaining])

                if nucleotide_count >= max_nucleotides:
                    break

            if nucleotide_count >= max_nucleotides:
                break

    print(f"  Loaded {nucleotide_count:,} nucleotides")
    return sequence


# Load genome
fasta_file = find_human_fasta()
genome_seq = load_genome(fasta_file)
genome_len = len(genome_seq)
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

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='#000011')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# ---------- Human Genome Class ----------
class Ecoli:
    def __init__(self, genome_seq, pos_offset=np.zeros(3)):
        self.genome = genome_seq
        self.genome_len = len(genome_seq)
        self.frame = 0
        self.accum_s1 = []
        self.accum_s2 = []
        self.accum_donut = []
        self.organelles = []
        self.pos_offset = pos_offset

        # Visuals
        self.strand1_vis = Line(pos=np.zeros((1,3)), color=(0,1,1,0.9), width=2, parent=view.scene)
        self.strand2_vis = Line(pos=np.zeros((1,3)), color=(1,0.5,0,0.9), width=2, parent=view.scene)
        self.donut_vis   = Line(pos=np.zeros((1,3)), color=(1,1,0,0.5), width=1.5, parent=view.scene)
        self.labels = []

    def update(self):
        self.frame += 1
        idx = self.frame % self.genome_len
        base = self.genome[idx]

        # Lattice computation (positive/deterministic)
        comp_s1 = np.zeros(3)
        comp_s2 = np.zeros(3)
        for base_map in all_mappings:
            dim = base_map.get(base,1)-1
            r = core_radius * (1 - (idx/self.genome_len)**1.4)
            r = max(r,0.5)
            theta = np.radians(golden_angle_deg) * idx
            z = np.sin(idx/self.genome_len * np.pi*4) * 1.6 + (idx/self.genome_len)*6.0
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

        # Yin/Yang drift + noise
        progress = idx/self.genome_len
        drift_amp = 0.02 + 0.8*progress
        z_bias = np.sin(self.frame*0.003 + progress*10.0)*0.5
        comp_s1[0] += drift_amp * np.cos(self.frame*0.005)
        comp_s2[0] -= drift_amp * np.cos(self.frame*0.005)
        comp_s1 += np.random.normal(scale=0.003, size=3)
        comp_s2 += np.random.normal(scale=0.003, size=3)
        comp_s1[2] += z_bias
        comp_s2[2] -= z_bias*0.6

        comp_s1 += self.pos_offset
        comp_s2 += self.pos_offset

        self.accum_s1.append(comp_s1.copy())
        self.accum_s2.append(comp_s2.copy())

        donut_center = (comp_s1 + comp_s2)/2.0
        self.accum_donut.append(donut_center.copy())

        if len(self.accum_s1) > MAX_POINTS:
            drop = len(self.accum_s1)//3
            self.accum_s1 = self.accum_s1[drop:]
            self.accum_s2 = self.accum_s2[drop:]
            self.accum_donut = self.accum_donut[drop:]

        # Organelles
        spawn_prob, cluster_size, radial_bias, noise_scale = organelle_params_from_base(base)
        if np.random.rand() < spawn_prob:
            color_rgb = geometries[min(consensus_dim_for_base(base), len(geometries)-1)][2]
            org = spawn_organelle(donut_center, color_rgb, size=0.8+radial_bias, n=cluster_size)
            self.organelles.append(org)

        # Lattice push
        lattice_nodes = np.array(self.accum_s1 + self.accum_s2)
        for org in self.organelles:
            positive = base in ['A','T']
            new_pts = np.array([lattice_push(p, lattice_nodes, positive=positive, strength=0.02) for p in org['positions']])
            org['positions'] = new_pts
            org['marker'].set_data(pos=new_pts, face_color=org['color'], size=5)

        # Update visuals
        self.strand1_vis.set_data(np.array(self.accum_s1))
        self.strand2_vis.set_data(np.array(self.accum_s2))
        if len(self.accum_donut) > 1:
            self.donut_vis.set_data(np.array(self.accum_donut))

        # Labels
        if self.frame % 160 == 0:
            dim = consensus_dim_for_base(base)
            col = geometries[min(dim, len(geometries)-1)][2]
            lbl = Text(f"{base}:{geometries[min(dim,7)][3]}", pos=donut_center+[0,0,0.25],
                       color=col, font_size=9, bold=True, parent=view.scene)
            self.labels.append(lbl)

# ---------- CREATE BACTERIA ----------
bacteria = []
offsets = np.random.uniform(-12,12,(NUM_BACTERIA,3))
for off in offsets:
    bacteria.append(Ecoli(genome_seq, pos_offset=off))

# ---------- UPDATE LOOP ----------
def update(ev):
    for bac in bacteria:
        bac.update()
    # Rotate camera globally
    view.camera.azimuth += 0.12
    # Use time.perf_counter() for a monotonic timer
    import time
    t = time.perf_counter()
    view.camera.elevation = 15 + 8*np.sin(t*0.002)
    canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == "__main__":
    canvas.show()
    app.run()
