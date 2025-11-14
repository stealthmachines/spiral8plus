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

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='#000011')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Initialize visual objects

strand1_vis = Line(pos=np.zeros((100, 3)), color='cyan', width=2, parent=view.scene)
strand2_vis = Line(pos=np.zeros((100, 3)), color='cyan', width=2, parent=view.scene)
donut_vis = Line(pos=np.zeros((1, 3)), color='yellow', width=1, parent=view.scene)

# Variables
core_radius = 15.0
strand_sep = 0.5
frame = 0

labels, organelles = [], []
def consensus_dim_for_base(b):
    """Digital positive structure (deterministic lattice coordinate selection)"""
    return int(np.round(np.mean([m.get(b,1)-1 for m in all_mappings])))

def organelle_params_from_base(b):
    """Analog negative structure (noise-driven, stochastic)"""
    d = consensus_dim_for_base(b)
    spawn_prob = 0.02 + (d / 16.0) * 0.12
    cluster_size = 6 + (d % 5) * 3
    radial_bias = 0.2 + (d / 16.0) * 0.9
    noise_scale = 0.08 + (d / 16.0) * 0.4
    return spawn_prob, cluster_size, radial_bias, noise_scale

def spawn_organelle(center, color_rgb, size=0.5, n=12):
    """Create organelle markers with stochastic jitter"""
    pts = center + np.random.normal(scale=0.2, size=(n,3)) * size
    rgba = list(Color(color_rgb).rgba)
    rgba[3] = 1.0
    mark = Markers(pos=pts, face_color=rgba, edge_color=None, size=5, parent=view.scene)
    return {"marker": mark, "positions": pts, "size": size, "color": rgba}

def lattice_push(pt, lattice_nodes, positive=True, strength=0.02):
    """Move point toward/away lattice nodes (yin/yang)"""
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
    # Ensure accum_s1 and accum_s2 are initialized
    if globals().get('accum_s1', None) is None:
        accum_s1 = []
    if globals().get('accum_s2', None) is None:
        accum_s2 = []
    # Ensure accum_donut is initialized
    if globals().get('accum_donut', None) is None:
        accum_donut = []

    frame += 1
    idx = frame % genome_len
    base = genome_seq[idx]

    # ---------- Genomic lattice computation ----------
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

    # ---------- Yin/Yang drift (internal unpacking + analog noise) ----------
    progress = idx/genome_len
    drift_amp = 0.02 + 0.8*progress
    z_bias = np.sin(frame*0.003 + progress*10.0)*0.5
    # deterministic positive
    comp_s1[0] += drift_amp * np.cos(frame*0.005)
    comp_s2[0] -= drift_amp * np.cos(frame*0.005)
    # analog negative
    comp_s1 += np.random.normal(scale=0.003, size=3)
    comp_s2 += np.random.normal(scale=0.003, size=3)
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

    # ---------- Spawn organelles (FASTA-driven stochasticity) ----------
    spawn_prob, cluster_size, radial_bias, noise_scale = organelle_params_from_base(base)
    if np.random.rand() < spawn_prob:
        color_rgb = geometries[min(consensus_dim_for_base(base), len(geometries)-1)][2]
        org = spawn_organelle(donut_center, color_rgb, size=0.8+radial_bias, n=cluster_size)
        organelles.append(org)

    # ---------- Lattice push (yin/yang backpressure) ----------
    lattice_nodes = np.array(accum_s1 + accum_s2)
    for org in organelles:
        positive = base in ['A','T']  # deterministic vs noise
        new_pts = np.array([lattice_push(p, lattice_nodes, positive=positive, strength=0.02) for p in org['positions']])
        org['positions'] = new_pts
        org['marker'].set_data(pos=new_pts, face_color=org['color'], size=5)

    # ---------- Update visuals ----------
    strand1_vis.set_data(np.array(accum_s1))
    strand2_vis.set_data(np.array(accum_s2))
    if len(accum_donut) > 1:
        donut_vis.set_data(np.array(accum_donut))

    # ---------- Labels ----------
    if frame % 160 == 0:
        dim = consensus_dim_for_base(base)
        col = geometries[min(dim, len(geometries)-1)][2]
        lbl = Text(f"{base}:{geometries[min(dim,7)][3]}", pos=donut_center+[0,0,0.25],
                   color=col, font_size=9, bold=True, parent=view.scene)
        labels.append(lbl)

    # ---------- Camera rotation ----------
    view.camera.azimuth = frame*0.12
    view.camera.elevation = 15 + 8*np.sin(frame*0.002)
    canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == "__main__":
    canvas.show()
    app.run()
