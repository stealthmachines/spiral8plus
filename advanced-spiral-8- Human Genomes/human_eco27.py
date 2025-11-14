# 🌌 DNA φ-Harmonic Spiral Composite & Generative Stack (24 mappings)
# ------------------------------------------------------------------
# Each frame adds a new "slice" of the spiral in 3D space — like printing layer-by-layer.
# ------------------------------------------------------------------

import numpy as np
import itertools
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)

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
N = 600


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

# Strands
comp_strand1 = Line(pos=np.zeros((1, 3)), color=(1, 1, 1, 0.7), width=2, parent=comp_view.scene)
comp_strand2 = Line(pos=np.zeros((1, 3)), color=(1, 1, 1, 0.7), width=2, parent=comp_view.scene)

# Variables
core_radius = 15.0
strand_sep = 0.5
frame = 0
points_s1, points_s2 = [], []
collections = {'rungs': [], 'labels': []}

rungs, labels = [], []
def update(ev):
    global frame, points_s1, points_s2
    frame += 1

    idx = frame % genome_len
    base = genome_seq[idx]

    # Generate just ONE new point slice (layer)
    composite_s1, composite_s2 = np.zeros(3), np.zeros(3)

    for base_map in all_mappings:
        dim = base_map.get(base, 1) - 1
        _, _, col, _, alpha, verts = geometries[dim]

        r = core_radius * (1 - (idx/genome_len)**1.5)
        r = max(r, 0.5)
        theta = idx * np.radians(golden_angle_deg)
        z = np.sin(idx/genome_len * np.pi * 4) * 2 + (idx/genome_len)*8
        a1 = np.radians(angles[dim])
        a2 = np.radians(-angles[dim])

        composite_s1 += np.array([
            r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
            r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
            z
        ])
        composite_s2 += np.array([
            r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
            r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
            z
        ])

    composite_s1 /= len(all_mappings)
    composite_s2 /= len(all_mappings)

    # Append new layer
    points_s1.append(composite_s1)
    points_s2.append(composite_s2)

    # Update geometry incrementally
    comp_strand1.set_data(np.array(points_s1))
    comp_strand2.set_data(np.array(points_s2))

    # Build lattice occasionally
    if frame % 80 == 0:
        dim_counts = [m.get(base,1)-1 for m in all_mappings]
        cur_dim = int(np.round(np.mean(dim_counts)))
        dim, note, col, name, alpha, verts = geometries[cur_dim]

        rgba = Color(col).rgba
        cen = (composite_s1 + composite_s2) / 2
        lbl = Text(f"{base}-{name}", pos=cen + [0,0,0.3],
                   color=col, font_size=10, bold=True, parent=comp_view.scene)
        collections['labels'].append(lbl)

        mark = Markers(pos=np.vstack([composite_s1, composite_s2]),
                       face_color=rgba, edge_color='white', size=8, parent=comp_view.scene)
        collections['rungs'].append(mark)

    # Camera rotation
    comp_view.camera.azimuth = frame*0.2
    comp_view.camera.elevation = 20 + 5*np.sin(frame*0.005)
    comp_canvas.update()

timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == "__main__":
    comp_canvas.show()
    app.run()
