import numpy as np
from itertools import permutations
from concurrent.futures import ThreadPoolExecutor

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

geometries = [("Point",1), ("Line",2), ("Triangle",3), ("Tetrahedron",4)]
nucleotides = ["A","T","G","C"]  # Only the 4 bases that have mappings
all_mappings = list(permutations(range(4)))  # 24 possible mappings

max_frames = 10000
eval_interval = 100  # check fitness every N frames

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
def fitness_metric_rolling(frame_num, coords_buffer):
    """Rolling std dev of distances between consecutive centers"""
    if len(coords_buffer) < 2:
        return 0.0
    diffs = np.diff(coords_buffer, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return -np.std(dists)

# ---------- RUN SINGLE MAPPING ----------
def run_spiral(mapping):
    phi = (1 + np.sqrt(5)) / 2
    golden_angle_deg = 360 / (phi**2)

    best_fit = -np.inf
    convergence_frame = None

    # rolling buffer for last few coordinates
    coords_buffer = []

    for frame in range(max_frames):
        base = genome_seq[frame % genome_len]
        base_normalized = "A" if base == "N" else base
        geom_idx = mapping[nucleotides.index(base_normalized)]
        _, verts = geometries[geom_idx]

        # Spiral coordinates (simplified)
        theta = frame * np.radians(golden_angle_deg)
        r = 5 + frame * 0.01
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = frame * 0.02
        coords_buffer.append(np.array([x, y, z]))

        # Keep buffer small (sliding window)
        if len(coords_buffer) > 1000:
            coords_buffer.pop(0)

        # Evaluate fitness periodically
        if frame % eval_interval == 0 and len(coords_buffer) > 1:
            fit = fitness_metric_rolling(frame, np.array(coords_buffer))
            if fit > best_fit:
                best_fit = fit
                convergence_frame = frame

        # Optional: percent complete
        if frame % 2000 == 0:
            percent_complete = frame / max_frames * 100
            print(f"Mapping {mapping} | {percent_complete:.2f}% complete")

    return mapping, best_fit, convergence_frame

# ---------- RUN ALL MAPPINGS IN PARALLEL ----------
def run_all_mappings_parallel():
    results = []
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [executor.submit(run_spiral, mapping) for mapping in all_mappings]
        for future in futures:
            results.append(future.result())
    return results

# ---------- MAIN ----------
if __name__ == "__main__":
    print("Starting parallel split test of 24 mappings...")
    results = run_all_mappings_parallel()
    best_mapping = max(results, key=lambda x: x[2])  # earliest convergence
    print("\n[OK] Best mapping (fastest convergence):", best_mapping)
