# dna_fasta_holo_cell.py
# --------------------------------------------------------------
# Full volumetric holographic Human Genome-like cell
# EVERYTHING derived from FASTA
# Install: pip install vispy pyqt6 numpy
# Run: python dna_fasta_holo_cell.py
# --------------------------------------------------------------

import os
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers
from vispy.color import Color

# ---------- LOAD FASTA ----------

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

# ---------- CONSTANTS ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)

# ---------- VISUAL SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 900), bgcolor='#000011')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Visual elements
strand1_vis = Line(pos=np.zeros((1, 3)), color=(0.2, 0.8, 1, 0.9), width=2, parent=view.scene)
strand2_vis = Line(pos=np.zeros((1, 3)), color=(1, 0.2, 0.8, 0.9), width=2, parent=view.scene)
donut_vis = Line(pos=np.zeros((1, 3)), color=(1, 1, 1, 0.5), width=1, parent=view.scene)

# ---------- STATE ----------
accum_s1, accum_s2, accum_donut = [], [], []
organelles = []
frame = 0
division_count = 0

def base_value(b):
    """Numeric mapping fully derived from FASTA."""
    return (ord(b) % 10) + 1

def nucleotide_color(b):
    """Color derived from ASCII code of base."""
    v = (ord(b) % 255)/255.0
    return (v, 1-v, 0.5+0.5*v)

# ---------- HOLOGRAPHIC HELPERS ----------
def lattice_push(pt, lattice_nodes, strength):
    if len(lattice_nodes) == 0: return pt
    nearest = lattice_nodes[np.random.randint(0,len(lattice_nodes))]
    dir_vec = nearest - pt
    return pt + dir_vec * strength

def spawn_organelle(center, base):
    val = base_value(base)
    n = val + 4
    scale = val*0.05 + 0.05
    pts = center + np.random.normal(scale=scale, size=(n,3))
    col = nucleotide_color(base)
    rgba = list(col)+[1.0]
    mark = Markers(pos=pts, face_color=rgba, size=5, parent=view.scene)
    return {"marker": mark, "positions": pts, "color": rgba}

def divide_cell():
    global accum_s1, accum_s2, accum_donut, organelles
    half = len(accum_s1)//2
    accum_s1 = accum_s1[half:]
    accum_s2 = accum_s2[half:]
    accum_donut = accum_donut[half:]
    for org in organelles:
        org['positions'] += np.random.normal(scale=0.3, size=org['positions'].shape)
    print("Cell divided! Total divisions:", division_count)

# ---------- UPDATE LOOP ----------
def update(ev):
    global frame, accum_s1, accum_s2, accum_donut, organelles, division_count

    frame += 1
    idx = frame % genome_len
    base = genome_seq[idx]
    val = base_value(base)

    # ---------- Holographic positions ----------
    r = 10 + val*0.5
    theta = np.radians(frame*golden_angle_deg)
    z = np.sin(frame/genome_len*np.pi*4)*2 + (frame/genome_len)*6

    p1 = np.array([r*np.cos(theta), r*np.sin(theta), z])
    p2 = np.array([r*np.cos(theta)+0.5, r*np.sin(theta)-0.5, z])
    p1 += np.random.normal(scale=0.01*val, size=3)
    p2 += np.random.normal(scale=0.01*val, size=3)

    accum_s1.append(p1)
    accum_s2.append(p2)
    center = (p1+p2)/2
    accum_donut.append(center)

    if len(accum_s1) > 6000:
        drop = len(accum_s1)//3
        accum_s1 = accum_s1[drop:]
        accum_s2 = accum_s2[drop:]
        accum_donut = accum_donut[drop:]

    # ---------- Organelles ----------
    if frame % (20 + val) == 0:
        org = spawn_organelle(center, base)
        organelles.append(org)

    lattice_nodes = np.array(accum_s1 + accum_s2)
    for org in organelles:
        new_pts = np.array([lattice_push(p, lattice_nodes, 0.02) for p in org['positions']])
        org['positions'] = new_pts
        org['marker'].set_data(pos=new_pts, face_color=org['color'], size=5)

    # ---------- Update visuals ----------
    strand1_vis.set_data(np.array(accum_s1))
    strand2_vis.set_data(np.array(accum_s2))
    donut_vis.set_data(np.array(accum_donut))

    # ---------- Camera ----------
    view.camera.azimuth = frame*0.15
    view.camera.elevation = 15 + 8*np.sin(frame*0.002)

    # ---------- Cell division ----------
    if frame % 2000 == 0:
        division_count += 1
        divide_cell()

    canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == "__main__":
    canvas.show()
    app.run()
