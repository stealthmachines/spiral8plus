# dna_full_genome_batched.py
# --------------------------------------------------------------
# Full-genome φ-spiral, every base pair rung rendered.
# GPU-accelerated, batched lines for efficiency.
# --------------------------------------------------------------

import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Markers, Line
from vispy.color import Color
from Bio import SeqIO
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
golden_angle = 2 * np.pi / (phi**2)
rise_per_base = 0.005
scale = 0.2
helix_radius = 0.05
echo_scale = 0.7

# Base-to-color mapping
base_colors = {'A': (1,0,0,1), 'C': (0,1,0,1), 'G': (0,0,1,1), 'T': (1,1,0,1),
    'N': (0.5, 0.5, 0.5, 1)}  # Gray (unknown)

# ---------- LOAD GENOME ----------
genome_seq = load_genome(find_human_fasta())
N = len(genome_seq)

# ---------- SPIRAL POSITIONS ----------
theta = np.arange(N) * golden_angle
r = scale * np.sqrt(np.arange(N))
z = np.arange(N) * rise_per_base

x1 = r * np.cos(theta) + helix_radius * np.cos(theta*2)
y1 = r * np.sin(theta) + helix_radius * np.sin(theta*2)

x2 = r * np.cos(theta) - helix_radius * np.cos(theta*2)
y2 = r * np.sin(theta) - helix_radius * np.sin(theta*2)

pos1 = np.column_stack((x1, y1, z))
pos2 = np.column_stack((x2, y2, z))

colors1 = np.array([base_colors.get(b, (0.5, 0.5, 0.5, 1)) for b in genome_seq], dtype=np.float32)
colors2 = colors1.copy()

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='#000011')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# MARKERS
strand1 = Markers(pos=pos1, face_color=colors1, size=2, parent=view.scene)
strand2 = Markers(pos=pos2, face_color=colors2, size=2, parent=view.scene)

# RUNG CONNECTIONS - BATCHED
# RUNGS: Create batched multi-segment lines
batch_size = 10000  # lines per batch for GPU efficiency
lines = []

for i in range(0, N, batch_size):
    end = min(i+batch_size, N)
    pts = np.empty(((end-i)*2,3), dtype=np.float32)
    pts[0::2] = pos1[i:end]
    pts[1::2] = pos2[i:end]
    # Create line outside the tight loop to reduce initialization overhead
    lines.append(pts)

# Create Line objects after collecting all data
for pts in lines:
    Line(pos=pts, color=(0.7,0.7,0.7,0.5), width=1, connect='segments', parent=view.scene)

# ECHO COPIES
echo1 = Markers(pos=pos1*echo_scale, face_color=np.array(colors1)*0.25, size=1.5, parent=view.scene)
echo2 = Markers(pos=pos2*echo_scale, face_color=np.array(colors2)*0.25, size=1.5, parent=view.scene)

# ---------- ROTATION ----------
frame = 0
def update(ev):
    global frame
    frame += 1
    view.camera.azimuth = frame * 0.03
    view.camera.elevation = 20
    canvas.update()

timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    app.run()
