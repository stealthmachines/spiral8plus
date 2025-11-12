# dna_echo_cell.py
# --------------------------------------------------------------
# GPU-accelerated genome-driven Human Genome cell
# Chromosome packs into φ-core; membrane, organelles, and echoes
# are generated dynamically.
# Install: pip install vispy pyqt6
# --------------------------------------------------------------

import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- FRAMEWORK ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)  # 137.507°
golden_angle_rad = np.radians(golden_angle_deg)

# nucleotide → geometry mapping
base_map = {'A': 5, 'T': 2, 'G': 4, 'C': 1}

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
frame = 0
core_radius = 15.0
nucleoid_radius = 12.0
cell_radius = 25.0
total_twist = 4 * np.pi
strand_sep = 0.5
twist_factor = 2 * np.pi
rungs, membrane, organelles, labels, centers, emerged = [], [], [], [], [], []

def update(ev):
    global frame, rungs, membrane, organelles, labels, centers, emerged

    frame += 1
    N = 600
    t = np.linspace(0, frame, N)

    s1, s2 = [], []
    membrane_pts = []
    organelle_pts = []

    for tt in t:
        idx = int(tt) % genome_len
        base = genome_seq[idx]
        dim = base_map.get(base, 1) - 1
        _, _, col, _, alpha, verts = geometries[dim]

        # ---------- NUCLEOID PACKING ----------
        r = nucleoid_radius * (1 - (tt/genome_len)**1.5)
        r = max(r, 0.5)
        theta = tt * golden_angle_rad
        twist = tt/genome_len * total_twist
        z = np.sin(tt/genome_len * np.pi * 4) * 2 + (tt/genome_len)*8

        # Strand 1
        a1 = np.radians(angles[dim])
        s1.append([r*np.cos(theta+twist)*np.cos(a1) - r*np.sin(theta+twist)*np.sin(a1),
                   r*np.sin(theta+twist)*np.cos(a1) + r*np.cos(theta+twist)*np.sin(a1),
                   z])
        # Strand 2
        a2 = np.radians(-angles[dim])
        s2.append([r*np.cos(theta+twist)*np.cos(a2) - r*np.sin(theta+twist)*np.sin(a2) + strand_sep,
                   r*np.sin(theta+twist)*np.cos(a2) + r*np.cos(theta+twist)*np.sin(a2) - strand_sep,
                   z])

        # ---------- CELL MEMBRANE ----------
        progress = tt/genome_len
        env_radius = nucleoid_radius + (cell_radius - nucleoid_radius) * progress
        env_radius += np.sin(tt*0.05)*0.3  # slight undulation
        membrane_pts.append([env_radius*np.cos(theta),
                             env_radius*np.sin(theta),
                             z*0.5])

        # ---------- ORGANELLES (ribosomes/granules) ----------
        if base in ['G','C']:
            organelle_pts.append([r*np.cos(theta), r*np.sin(theta), z*0.8])

    # ---------- UPDATE VISUALS ----------
    strand1.set_data(np.array(s1))
    strand2.set_data(np.array(s2))

    # Chromosome markers (emerged rungs)
    cur_base = genome_seq[frame % genome_len]
    cur_dim = base_map.get(cur_base, 1) - 1
    if frame % 20 == 0 and cur_dim > len(rungs)-1:
        dim, note, col, name, alpha, verts = geometries[cur_dim]
        emerged.append(cur_dim)

        start = int((cur_dim/8.0)*N)
        step = max(1, N // (8*verts))
        idx = np.arange(start, start+verts*step, step)[:verts]
        idx = idx[idx < N]

        pts1 = np.array(s1)[idx]
        pts2 = np.array(s2)[idx]
        rgba = Color(col).rgba
        edge_rgba = list(rgba[:3]) + [0.9]

        # Closed lattice
        segs = []
        for pts in (pts1, pts2):
            for i in range(verts):
                for j in range(i+1, verts):
                    segs += [pts[i], pts[j]]
        for i in range(verts):
            segs += [pts1[i], pts2[i]]
        if segs:
            line = Line(pos=np.array(segs), color=edge_rgba, width=2,
                        connect='segments', parent=view.scene)
            rungs.append(line)

        all_pts = np.vstack((pts1, pts2))
        mark = Markers(pos=all_pts, face_color=rgba, edge_color='white',
                       size=6, parent=view.scene)
        rungs.append(mark)

        cen = all_pts.mean(axis=0)
        centers.append(cen)
        lbl = Text(f"{cur_base}: {name}", pos=cen + [0,0,0.3],
                   color=col, font_size=10, bold=True,
                   anchor_x='center', parent=view.scene)
        labels.append(lbl)

    # Membrane markers
    if membrane_pts:
        membrane_array = np.array(membrane_pts)
        mark = Markers(pos=membrane_array, face_color=(0.4,0.6,1,0.4),
                       edge_color='white', size=4, parent=view.scene)
        membrane.append(mark)

    # Organelles
    if organelle_pts:
        organelle_array = np.array(organelle_pts)
        mark = Markers(pos=organelle_array, face_color=(1,1,0,0.6),
                       edge_color='white', size=3, parent=view.scene)
        organelles.append(mark)

    # ---------- CAMERA ----------
    view.camera.azimuth = frame * 0.2
    view.camera.elevation = 15 + 5*np.sin(frame*0.003)

    # ---------- % COMPLETE ----------
    pct = frame / genome_len * 100
    if pct > 100: pct = 100
    if not hasattr(update, "pct_text"):
        update.pct_text = Text(f"{pct:.1f}% complete", pos=[0,0,cell_radius+5],
                               color='white', font_size=16, bold=True,
                               anchor_x='center', parent=view.scene)
    else:
        update.pct_text.text = f"{pct:.1f}% complete"

    canvas.update()

# ---------- TIMER ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    app.run()
