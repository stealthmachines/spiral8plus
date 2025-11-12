# dna_full_genome.py
# --------------------------------------------------------------
# Full-genome φ-spiral, GPU-accelerated, color-coded A/C/G/T.
# Install: pip install vispy pyqt6
# --------------------------------------------------------------

import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Markers
from vispy.color import Color


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

def load_genome(fasta_file, max_nucleotides=None, chromosome=None):
    """
    Load genome sequence from FASTA file with environment variable support

    Args:
        fasta_file: Path to FASTA file
        max_nucleotides: Maximum nucleotides to load (None = use GENOME_LIMIT env var, default 100000)
        chromosome: Specific chromosome to load (None = use GENOME_CHROMOSOME env var)

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

    sequence = ""
    nucleotide_count = 0

    print(f"Loading genome from {fasta_file}...")
    if chromosome:
        print(f"  Filtering: Chromosome {chromosome}")
    print(f"  Limit: {max_nucleotides:,} nucleotides")

    # Parse all records (human genome has multiple chromosomes)
    for record in SeqIO.parse(fasta_file, "fasta"):
        # Check if this is the chromosome we want
        if chromosome and record.id != chromosome:
            continue

        if chromosome:
            print(f"  Loading from {record.id}...")

        # Add sequence from this chromosome
        seq_str = str(record.seq)
        remaining = max_nucleotides - nucleotide_count

        if remaining <= 0:
            break

        sequence += seq_str[:remaining]
        nucleotide_count += len(seq_str[:remaining])

        if nucleotide_count >= max_nucleotides:
            break

        # If no chromosome filter, just take from first chromosome
        if not chromosome:
            break

    print(f"  Loaded {nucleotide_count:,} nucleotides")
    return sequence


# ---------- PARAMETERS ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle = 2 * np.pi / (phi**2)   # radians
rise_per_base = 0.005                 # vertical separation per base
scale = 0.2                            # radial scale

# Base-to-color mapping
base_colors = {
    'A': (1, 0, 0, 1),   # Red
    'C': (0, 1, 0, 1),   # Green
    'G': (0, 0, 1, 1),   # Blue
    'T': (1, 1, 0, 1),   # Yellow
    'N': (0.5, 0.5, 0.5, 1),  # Gray (unknown)
}

# ---------- LOAD GENOME ----------
# genome_seq should be a string of 'A','C','G','T'
# Example: genome_seq = "ATGCGTAC..."
from Bio import SeqIO
import os
genome_seq = load_genome(find_human_fasta())

N = len(genome_seq)

# ---------- COMPUTE SPIRAL POSITIONS ----------
theta = np.arange(N) * golden_angle
r = scale * np.sqrt(np.arange(N))         # optional: sqrt spacing
z = np.arange(N) * rise_per_base

x = r * np.cos(theta)
y = r * np.sin(theta)

positions = np.column_stack((x, y, z))

# ---------- MAP COLORS ----------
colors = np.array([base_colors.get(b, (0.5, 0.5, 0.5, 1)) for b in genome_seq], dtype=np.float32)

# ---------- VISPY ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='#000011')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# markers for all bases
markers = Markers(pos=positions, face_color=colors, size=3, parent=view.scene)

# ---------- ROTATION ----------
frame = 0
def update(ev):
    global frame
    frame += 1
    view.camera.azimuth = frame * 0.05
    view.camera.elevation = 20
    canvas.update()

timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    app.run()
