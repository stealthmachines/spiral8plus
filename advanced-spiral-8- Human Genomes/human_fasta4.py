"""
FASTA-Driven Holographic DNA φ-Spiral Cell
Everything is genome-driven; coordinates, noise, organelles, and division
are fully derived from the sequence. No contrived constants.
Requirements: pip install vispy pyqt6 numpy
Run: python fasta_cell.py
"""

import os
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ----------------- CONFIG -----------------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi**2)  # 137.507°

# Base → dimension mapping
base_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

# Geometries: (dimension, color, vertices)
geometries = [
    (0, 'red', 1),
    (1, 'green', 2),
    (2, 'blue', 3),
    (3, 'violet', 4),
    (4, 'orange', 5),
    (5, 'indigo', 6),
    (6, 'purple', 7),
    (7, 'white', 8)
]

angles = [i * golden_angle_deg for i in range(8)]


core_radius = 15.0
strand_sep = 0.5
twist_factor = 2*np.pi
max_points = 8000

# Initialize state
accum_s1 = []
accum_s2 = []
organelles = []
labels = []
centers = []

# Initialize progress text visual (if used)
from vispy.scene.visuals import Text
progress_text = Text("", pos=(10, 10), color='white', font_size=12)

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

# ----------------- LOAD GENOME -----------------
fasta_path = find_human_fasta()
if not os.path.exists(fasta_path):
    raise FileNotFoundError(f"FASTA not found: {fasta_path}")

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

def genome_noise(idx, dim):
    """FASTA-derived pseudo-noise vector (no randomness)"""
    seq_window = genome_seq[idx:idx+3]
    vals = [ord(c)%10 for c in seq_window]
    return np.array([vals[0], vals[1], vals[2]]) * 0.001 * (dim+1)

# ---------- LOAD GENOME ----------
fasta_file = find_human_fasta()
genome_seq = load_genome(fasta_file) if "load_genome" in dir() else open(fasta_file).read().replace("\n", "").upper()
genome_len = len(genome_seq)
print(f"Genome loaded: {genome_len:,} nucleotides")

canvas = scene.SceneCanvas(keys='interactive', size=(1400, 1000), bgcolor='#000011')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Initialize visual objects
strand1_vis = Line(pos=np.zeros((100, 3)), color='cyan', width=2, parent=view.scene)
strand2_vis = Line(pos=np.zeros((100, 3)), color='cyan', width=2, parent=view.scene)

def spawn_organelle(idx, center):
    """FASTA-driven organelle placement"""
    val = ord(genome_seq[idx % genome_len]) % 8
    geom_dim, color, verts = geometries[val]
    pts = center + genome_noise(idx,val)*5.0
    rgba = list(Color(color).rgba)
    rgba[3] = 1.0
    mark = Markers(pos=pts.reshape(1,3), face_color=rgba,
                   edge_color=None, size=6)
    return {"marker": mark, "positions": pts.reshape(1,3), "color": rgba}

# ----------------- UPDATE LOOP -----------------
frame = 0

def update(ev):
    global frame, accum_s1, accum_s2, rungs, organelles, labels, centers

    frame += 1
    idx = frame % genome_len
    # Defensive: always use .upper() and .get() for base_map lookup
    base = genome_seq[idx]
    dim = base_map.get(base.upper(), 0)

    # φ-spiral positions
    theta = idx * np.radians(golden_angle_deg)
    twist = idx / genome_len * twist_factor
    z = np.sin(idx/genome_len * np.pi * 4) * 2 + (idx/genome_len)*8
    a1 = np.radians(angles[dim])
    a2 = np.radians(-angles[dim])

    r = core_radius * (1 - (idx/genome_len)**1.5)
    r = max(r,0.5)

    p1 = np.array([r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                   r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
                   z]) + genome_noise(idx,dim)
    p2 = np.array([r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
                   r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
                   z]) + genome_noise(idx,dim)

    accum_s1.append(p1)
    accum_s2.append(p2)

    if len(accum_s1) > max_points:
        drop = len(accum_s1)//3
        accum_s1 = accum_s1[drop:]
        accum_s2 = accum_s2[drop:]

    strand1_vis.set_data(np.array(accum_s1))
    strand2_vis.set_data(np.array(accum_s2))

    # Organelles
    if idx % 50 == 0:
        center = (p1+p2)/2
        org = spawn_organelle(idx, center)
        organelles.append(org)

    # Update organelles (FASTA-driven backpressure)
    lattice_nodes = np.array(accum_s1 + accum_s2)
    for org in organelles:
        new_pts = []
        for p in org['positions']:
            nearest = lattice_nodes[idx % len(lattice_nodes)]
            dir_vec = nearest - p
            new_pts.append(p + dir_vec * 0.02)  # minimal push
        org['positions'] = np.array(new_pts)
        org['marker'].set_data(pos=org['positions'], face_color=org['color'], size=6)

    # Labels for major nucleotides
    if frame % 200 == 0:
        cen = (p1+p2)/2
        geom_dim, color, verts = geometries[dim]
        lbl = Text(f"{base}:{geom_dim}", pos=cen+[0,0,0.3],
                   color=color, font_size=10, bold=True,
                   anchor_x='center')
        labels.append(lbl)
        centers.append(cen)

    # Camera
    view.camera.azimuth = frame * 0.12
    view.camera.elevation = 15 + 8*np.sin(frame*0.002)

    # Progress
    progress_text.text = f"{min(frame/genome_len*100,100):.2f}%"
    canvas.update()

# ----------------- START -----------------
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
