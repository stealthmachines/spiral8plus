# dna_echo_genome_fullcell.py
# --------------------------------------------------------------
# GPU-accelerated Human Genome chromosome folding
# Genome-driven, tightly packed, φ-core convergence
# Install: pip install vispy pyqt6
# --------------------------------------------------------------

import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- FRAMEWORK ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)  # 137.507°

# nucleotide → instruction mapping
instruction_map = {
    'A': {'geom': 5, 'twist': +1, 'radial': -0.02, 'echo': True},    # pentachoron
    'T': {'geom': 2, 'twist': -1, 'radial': +0.01, 'echo': False},   # line
    'G': {'geom': 4, 'twist': +2, 'radial': -0.015, 'echo': True},   # tetrahedron
    'C': {'geom': 1, 'twist': 0, 'radial': 0, 'echo': False},        # point
}

# geometries: (dim, note, color, name, alpha, vertices)
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

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='#000011')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Strands
strand1 = Line(pos=np.zeros((1, 3)), color=(1, 1, 1, 0.7), width=2, parent=view.scene)
strand2 = Line(pos=np.zeros((1, 3)), color=(1, 1, 1, 0.7), width=2, parent=view.scene)

# Variables
core_radius = 15.0
strand_sep = 0.5
twist_factor = 2 * np.pi
frame = 0

rungs, echoes, links, labels, centers, emerged = [], [], [], [], [], []
def update(ev):
    global frame, rungs, echoes, links, labels, centers, emerged

    frame += 1
    # map progress to genome
    idx = frame % genome_len
    base = genome_seq[idx]
    instr = instruction_map.get(base, {'geom':1, 'twist':0, 'radial':0, 'echo':False})
    dim = instr['geom'] - 1
    _, _, col, _, alpha, verts = geometries[dim]

    # ---------- radial compression / twist ----------
    r = core_radius * (1 - (frame/genome_len)**1.5)
    r = max(r, 0.5)
    theta = frame * np.radians(golden_angle_deg)
    twist = frame/genome_len * twist_factor
    z = np.sin(frame/genome_len * np.pi * 4) * 2 + (frame/genome_len) * 8

    # strand positions
    a1 = np.radians(angles[dim])
    a2 = np.radians(-angles[dim])

    s1 = np.array([[r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                    r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
                    z]])
    s2 = np.array([[r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
                    r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
                    z]])

    strand1.set_data(np.vstack([strand1.pos, s1]))
    strand2.set_data(np.vstack([strand2.pos, s2]))

    # ---------- EMERGE RUNGS ----------
    if frame % 20 == 0:
        emerged.append(dim)
        all_pts = np.vstack([s1, s2])
        rgba = Color(col).rgba
        mark = Markers(pos=all_pts, face_color=rgba, edge_color='white', size=8, parent=view.scene)
        rungs.append(mark)

        # center & label
        cen = all_pts.mean(axis=0)
        centers.append(cen)
        lbl = Text(f"{base}: {geometries[dim][3]}", pos=cen+[0,0,0.3],
                   color=col, font_size=10, bold=True, anchor_x='center', parent=view.scene)
        labels.append(lbl)

        # echo
        if len(emerged) > 1 and instr['echo']:
            prev = emerged[-2]
            prev_col = list(Color(geometries[prev][2]).rgba)
            prev_col[3] = 0.25
            echo_pts = all_pts * 0.75
            echo = Markers(pos=echo_pts, face_color=prev_col, size=5, parent=view.scene)
            echoes.append(echo)

        # inter-rung links
        if len(centers) > 1:
            prev_c = centers[-2]
            segs = []
            for i in range(min(6, len(all_pts))):
                segs += [prev_c, cen]
            link = Line(pos=np.array(segs), color=(0.7,0.7,0.7,0.4),
                        width=1, connect='segments', parent=view.scene)
            links.append(link)

    # ---------- % COMPLETE ----------
    pct_complete = min(100.0, frame/genome_len*100)
    print(f"\rGenome Progress: {pct_complete:.2f}%", end="")

    # ---------- CAMERA ----------
    view.camera.azimuth = frame * 0.3
    view.camera.elevation = 20 + 5*np.sin(frame*0.005)
    canvas.update()

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
