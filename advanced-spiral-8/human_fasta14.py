"""
FASTA Universe: GPU-accelerated holographic DNA cell simulation
Everything is FASTA. All positions, spirals, organelles arise from the genome.
Install: pip install vispy pyqt6 numpy
Run: python fasta_universe.py
"""

import os
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- CONSTANTS ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)

# Genome mapping
bases = ['A','T','G','C']
base_map = {b:i for i,b in enumerate(bases)}  # A=0, T=1, G=2, C=3

# Geometry properties per base (color + structure)
geom_colors = ['red','green','blue','violet']
core_radius = 15.0
strand_sep = 0.5

MAX_POINTS = 200_000  # preallocate GPU buffers
BATCH_STEPS = 1000    # fast-forward genome per frame

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

# ---------- LOAD GENOME ----------
fasta_file = find_human_fasta()
genome_seq = load_genome(fasta_file)
genome_len = len(genome_seq)
print(f"Genome loaded: {genome_len:,} nucleotides")

# ---------- VISUAL SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1400, 1000), bgcolor='#000011')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

lattice_vis = Line(pos=np.zeros((2, 3)), color=(1,1,1,0.8), width=2, parent=view.scene)
progress_text = Text("", pos=(10, 10), color='white', font_size=14, parent=canvas.scene)

# ---------- STATE ----------
lattice = []
organelles = []
frame = 0

# Pre-allocated arrays for performance
MAX_POINTS = 10000
MAX_ORG_POINTS = 5000
lattice_positions = np.zeros((MAX_POINTS, 3), dtype=np.float32)
organelles_positions = np.zeros((MAX_ORG_POINTS, 3), dtype=np.float32)
lattice_count = 0
organelles_count = 0

# Markers
lattice_marker = Markers(pos=lattice_positions[:0], face_color=(1,1,1,0.7), size=2, parent=view.scene)
organelles_marker = Markers(pos=organelles_positions[:0], face_color=(0,1,1,0.7), size=3, parent=view.scene)

def genome_to_vec(idx):
    """Map genome base to 3D φ-spiral position."""
    base = genome_seq[idx]
    dim = base_map.get(base, 0)
    theta = idx * np.radians(golden_angle_deg)
    r = core_radius * (1 - (idx/genome_len)**1.4)
    z = np.sin(idx/genome_len * np.pi*4)*2 + (idx/genome_len)*8
    a = np.radians(dim * 90)
    x = r*np.cos(theta)*np.cos(a) - r*np.sin(theta)*np.sin(a)
    y = r*np.sin(theta)*np.cos(a) + r*np.cos(theta)*np.sin(a)
    return np.array([x,y,z], dtype=np.float32)

# ---------- CELL STATE ----------
cell_state = {"frame":0}

# ---------- UPDATE FUNCTION ----------
def update(ev):
    global lattice_count, organelles_count

    # --- Batch simulation of genome steps ---
    for _ in range(BATCH_STEPS):
        idx = cell_state["frame"] % genome_len
        pos = genome_to_vec(idx)
        lattice_positions[lattice_count % MAX_POINTS] = pos
        lattice_count += 1

        # Spawn organelles stochastically
        if np.random.rand() < 0.02:
            n = 5 + np.random.randint(5)
            organelle_pts = pos + np.random.normal(scale=0.2, size=(n,3))
            organelles_positions[organelles_count:organelles_count+n] = organelle_pts
            organelles_count += n

        cell_state["frame"] += 1

    # --- Update GPU visuals ---
    lattice_marker.set_data(lattice_positions[:lattice_count])
    organelles_marker.set_data(organelles_positions[:organelles_count])

    # --- Camera rotation ---
    view.camera.azimuth = cell_state["frame"] * 0.15
    view.camera.elevation = 15 + 5*np.sin(cell_state["frame"]*0.002)

    # --- Progress display ---
    progress = min(cell_state["frame"]/genome_len*100, 100)
    progress_text.text = f"{progress:.1f}%"

    canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Human Genome GRCh38.p14 Visualization")
    print("="*60)
    if 'genome_len' in dir():
        print(f"Genome length: {genome_len:,} nucleotides")
    print("="*60)

    canvas.show()
    app.run()
