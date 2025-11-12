"""
Generative DNA φ-Harmonic Spiral — volumetric, genome-driven cells & division
Requirements: pip install vispy pyqt6
Run: python generative_spiral_cells_full.py
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
N = 600

MAX_POINTS = 8000
MAX_ORGANELLES = 200

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
comp_canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='#000011')
comp_view = comp_canvas.central_widget.add_view()
comp_view.camera = 'turntable'

# Visual elements
strand_vis_1 = Line(pos=np.zeros((1,3)), color=(0.3,0.9,1,0.8), width=2.5, parent=comp_view.scene)
strand_vis_2 = Line(pos=np.zeros((1,3)), color=(1,0.4,0.2,0.8), width=2.5, parent=comp_view.scene)
donut_trail = Line(pos=np.zeros((1,3)), color=(1,1,1,0.3), width=1, parent=comp_view.scene)

# Variables
core_radius = 15.0
strand_sep = 0.5
frame = 0
accum_s1, accum_s2, accum_donut = [], [], []
echo_shells = []

echoes, labels, organelles = [], [], []
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

def spawn_organelle(center, color_rgb, size, n=12, persistence=180):
    pts = center + np.random.normal(scale=0.3, size=(n,3)) * size
    rgba = list(Color(color_rgb).rgba)
    rgba[3] = 0.95
    mark = Markers(pos=pts, face_color=rgba, edge_color='white', size=6, parent=comp_view.scene)
    return {"marker": mark, "age": 0, "life": persistence}

def make_echo_shell(center, scale, color_rgb, alpha=0.12):
    shell_pts = center + (np.random.normal(scale=0.4, size=(max(8, int(12*scale)),3)) * scale)
    rgba = list(Color(color_rgb).rgba)
    rgba[3] = alpha
    shell = Markers(pos=shell_pts, face_color=rgba, edge_color=None, size=3, parent=comp_view.scene)
    return {"marker": shell, "age": 0, "life": 300}

# ---------- UPDATE LOOP ----------
def update(ev):
    global frame, accum_s1, accum_s2, accum_donut, organelles, echo_shells

    frame += 1
    idx = frame % genome_len
    base = genome_seq[idx]

    composite_s1 = np.zeros(3)
    composite_s2 = np.zeros(3)
    for base_map in all_mappings:
        dim = base_map.get(base, 1) - 1
        _, _, col, _, alpha, verts = geometries[dim]
        r = core_radius * (1 - (idx / max(1, genome_len))**1.4)
        r = max(r, 0.5)
        theta = idx * np.radians(golden_angle_deg)
        z = np.sin(idx / max(1, genome_len) * np.pi * 4) * 1.6 + (idx / max(1, genome_len)) * 6.0
        a1 = np.radians(angles[dim])
        a2 = np.radians(-angles[dim])

        p1 = np.array([r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                       r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
                       z])
        p2 = np.array([r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
                       r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
                       z])
        composite_s1 += p1
        composite_s2 += p2

    composite_s1 /= len(all_mappings)
    composite_s2 /= len(all_mappings)

    spawn_prob, cluster_size, radial_bias, noise_scale = organelle_params_from_base(base)
    def radial_jitter(p, bias=radial_bias, noise=noise_scale):
        dir_xy = np.array([p[0], p[1], 0.0])
        norm = np.linalg.norm(dir_xy[:2]) + 1e-6
        unit = dir_xy / norm
        offset = unit * (bias * np.random.rand()) + np.random.normal(scale=noise, size=3)
        return p + offset

    composite_s1 = radial_jitter(composite_s1)
    composite_s2 = radial_jitter(composite_s2)
    anticor = np.random.normal(scale=0.02, size=3)
    composite_s1 += anticor
    composite_s2 -= anticor

    # Division drift
    progress = idx / max(1, genome_len)
    drift_amp = 0.02 + 0.8 * progress
    z_bias = np.sin(frame * 0.003 + progress * 10.0) * 0.5
    composite_s1[0] += drift_amp * np.cos(frame * 0.005)
    composite_s2[0] -= drift_amp * np.cos(frame * 0.005)
    composite_s1[2] += z_bias
    composite_s2[2] -= z_bias * 0.6

    accum_s1.append(composite_s1.copy())
    accum_s2.append(composite_s2.copy())

    if len(accum_s1) > MAX_POINTS:
        drop = len(accum_s1) // 3
        accum_s1 = accum_s1[drop:]
        accum_s2 = accum_s2[drop:]
        accum_donut = accum_donut[drop:]

    # Donut center
    donut_center = (composite_s1 + composite_s2) / 2.0
    accum_donut.append(donut_center.copy())

    # Optional logging
    if frame % 200 == 0:
        print(f"Frame {frame}: Donut center at {donut_center}")

    # Organelles
    if np.random.rand() < spawn_prob:
        local_center = donut_center
        dim = consensus_dim_for_base(base)
        color_rgb = geometries[min(dim, len(geometries)-1)][2]
        cluster = spawn_organelle(local_center, color_rgb, size=0.8 + radial_bias, n=cluster_size, persistence=260)
        organelles.append(cluster)
        echo = make_echo_shell(local_center, scale=0.6 + radial_bias, color_rgb=color_rgb, alpha=0.08)
        echo_shells.append(echo)

    # Large nucleus-like echo
    if (frame % 1000) == 0:
        big_echo = make_echo_shell(donut_center, scale=1.8, color_rgb='gold', alpha=0.12)
        echo_shells.append(big_echo)

    # Update visuals
    strand_vis_1.set_data(np.array(accum_s1))
    strand_vis_2.set_data(np.array(accum_s2))
    if len(accum_donut) > 1:
        donut_trail.set_data(np.array(accum_donut))

    # Fade organelles
    new_organelles = []
    for ent in organelles:
        ent['age'] += 1
        m = ent['marker']
        base_rgba = ent.get('base_rgba', (1,1,1,1))
        alpha = max(0.02, 1.0 - (ent['age'] / ent['life']))
        rgba = list(base_rgba)
        rgba[3] = alpha
        try: m.set_data(pos=m._pos, face_color=rgba)
        except Exception: pass
        if ent['age'] < ent['life']:
            ent['base_rgba'] = rgba
            new_organelles.append(ent)
        else:
            try: m.parent = None
            except Exception: pass
    organelles = new_organelles[:MAX_ORGANELLES]

    # Fade echoes
    new_echoes = []
    for ent in echo_shells:
        ent['age'] += 1
        m = ent['marker']
        base_rgba = ent.get('base_rgba', (1,1,1,1))
        alpha = max(0.0, 0.12 * (1.0 - ent['age'] / ent['life']))
        rgba = list(base_rgba)
        rgba[3] = alpha
        try: m.set_data(pos=m._pos, face_color=rgba)
        except Exception: pass
        if ent['age'] < ent['life']:
            ent['base_rgba'] = rgba
            new_echoes.append(ent)
        else:
            try: m.parent = None
            except Exception: pass
    echo_shells = new_echoes

    # Labels
    if frame % 160 == 0:
        dim = consensus_dim_for_base(base)
        col = geometries[min(dim, len(geometries)-1)][2]
        lbl = Text(f"{base}:{geometries[min(dim,7)][3]}", pos=donut_center + [0,0,0.25],
                   color=col, font_size=9, bold=True, parent=comp_view.scene)
        labels.append(lbl)

    # Rotate camera
    comp_view.camera.azimuth = frame * 0.12
    comp_view.camera.elevation = 15 + 8 * np.sin(frame * 0.002)

    comp_canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == "__main__":
    comp_canvas.show()
    app.run()
