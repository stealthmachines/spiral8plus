# dna_mapping_tester.py
# --------------------------------------------------------------
# Automated mapping test for DNA φ-spiral lattice
# --------------------------------------------------------------

import numpy as np
from itertools import permutations
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color
import os

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

# ---------- CONFIG ----------
fasta_file = find_human_fasta()
output_dir = "mapping_results"
os.makedirs(output_dir, exist_ok=True)

# Geometries (simplified for split-test)
geometries = [
    ("Point", 1, "red"),
    ("Line", 2, "green"),
    ("Triangle", 3, "blue"),
    ("Tetrahedron", 4, "yellow")
]

nucleotides = ["A", "T", "G", "C"]  # Include N for unknown/ambiguous bases
all_mappings = list(permutations(range(len(geometries))))  # 24 permutations

# ---------- LOAD GENOME ----------

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
def fitness_metric(centers):
    """Simple structural metric: mean distance between rung centers"""
    if len(centers) < 2:
        return 0
    dists = [np.linalg.norm(centers[i] - centers[i-1]) for i in range(1, len(centers))]
    return -np.std(dists)  # more uniform spacing = higher fitness

# ---------- RUN φ-SPIRAL SIMULATION ----------
def run_spiral(mapping):
    # VisPy setup
    canvas = scene.SceneCanvas(keys='interactive', size=(800,600), bgcolor='#000011', show=False)
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'

    # φ constants
    phi = (1 + np.sqrt(5))/2
    golden_angle_deg = 360 / (phi**2)

    frame = 0
    centers = []

    for base_idx, base in enumerate(genome_seq):
        base = base.upper()  # Ensure uppercase for matching
        # Handle unknown nucleotide 'N' by mapping to first geometry
        if base == 'N':
            geom_idx = 0
        else:
            geom_idx = mapping[nucleotides.index(base)]
        name, verts, col = geometries[geom_idx]

        # Simple spiral position (2D for demo)
        theta = frame * np.radians(golden_angle_deg)
        r = 5 + frame * 0.01
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = frame * 0.02
        centers.append(np.array([x,y,z]))

        # Optionally, draw a marker (minimal for speed)
        # mark = Markers(pos=[[x,y,z]], face_color=col, size=5, parent=view.scene)

        # Track progress
        percent_complete = (frame+1)/genome_len*100
        if frame % 10000 == 0:
            print(f"Mapping {mapping} | {percent_complete:.2f}% complete")
        frame += 1

    # Compute fitness
    fit = fitness_metric(centers)
    # Note: render_to_file not supported in vispy SceneCanvas - removed
    canvas.close()
    return fit

# ---------- TEST ALL MAPPINGS ----------
results = []
for i, mapping in enumerate(all_mappings):
    print(f"Testing mapping {i+1}/{len(all_mappings)}: {mapping}")
    fit = run_spiral(mapping)
    results.append((mapping, fit))

# ---------- FIND BEST MAPPING ----------
best_mapping, best_fit = max(results, key=lambda x: x[1])
print("\n[OK] Best mapping found:", best_mapping, "with fitness:", best_fit)
