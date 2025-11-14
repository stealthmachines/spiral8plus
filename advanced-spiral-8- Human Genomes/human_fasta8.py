"""
FASTA is Life — Full Holographic Genome-Driven Cell with Division
Everything in 3D (strands, organelles, decay, division) is derived from the genome.
Requirements: pip install vispy pyqt6 numpy
Run: python fasta_is_life.py
"""

import os
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ----------------- CONSTANTS -----------------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)

# Base → dimension mapping
base_map = {'A':0, 'T':1, 'G':2, 'C':3}

# Geometries: (dim, color, vertices)
geometries = [
    (0,'red',1), (1,'green',2), (2,'blue',3), (3,'violet',4),
    (4,'orange',5), (5,'indigo',6), (6,'purple',7), (7,'white',8)
]

angles = [i * golden_angle_deg for i in range(8)]
core_radius = 20.0
strand_sep = 1.0
twist_factor = 2*np.pi
max_points = 12000
decay_strength = 0.005
division_interval = 2000  # frames per division

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

def fasta_noise(idx, dim):
    """Genome-driven deterministic noise"""
    window = genome_seq[idx:idx+3]
    vals = [ord(c)%10 for c in window]
    return np.array([vals[0], vals[1], vals[2]])*0.001*(dim+1)

# ---------- LOAD GENOME ----------
fasta_file = find_human_fasta()
genome_seq = load_genome(fasta_file) if "load_genome" in dir() else open(fasta_file).read().replace("\n", "").upper()
genome_len = len(genome_seq)
print(f"Genome loaded: {genome_len:,} nucleotides")

canvas = scene.SceneCanvas(keys='interactive', size=(1400, 1000), bgcolor='#000011')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Initialize visual objects
strand1_vis = Line(pos=np.zeros((100, 3)), color='cyan', width=2)
strand2_vis = Line(pos=np.zeros((100, 3)), color='cyan', width=2)
view.add(strand1_vis)
view.add(strand2_vis)

def spawn_organelle(idx, center):
    val = ord(genome_seq[idx % genome_len]) % 8
    geom_dim, color, verts = geometries[val]
    pts = center + fasta_noise(idx,val)*10.0
    rgba = list(Color(color).rgba)
    rgba[3] = 1.0
    mark = Markers(pos=pts.reshape(1,3), face_color=rgba,
                   edge_color=None, size=6)
    view.add(mark)
    return {"marker": mark, "positions": pts.reshape(1,3), "color": rgba, "age":0}

cells = [{
    "frame": 0,
    "accum_s1": [],
    "accum_s2": [],
    "organelles": []
}]
progress_text = Text("", pos=(10, 10), color='white', font_size=12)
view.add(progress_text)
frame = 0
# ----------------- UPDATE LOOP -----------------
def update(ev):
    global frame
    new_cells = []
    for cell in cells:
        cell["frame"] += 1
        frame = cell["frame"]
        idx = frame % genome_len
        base = genome_seq[idx]
        dim = base_map.get(base,0)
    new_cells = []
    if len(cells) > 0:
        for cell in cells:
            cell["frame"] += 1
            frame = cell["frame"]
            idx = frame % genome_len
            base = genome_seq[idx]
            dim = base_map.get(base,0)

            theta = idx * np.radians(golden_angle_deg)
            twist = idx / genome_len * twist_factor
            z = np.sin(idx/genome_len * np.pi * 6) * 3 + (idx/genome_len)*12
            a1 = np.radians(angles[dim])
            a2 = np.radians(-angles[dim])
            r = core_radius * (1 - (idx/genome_len)**1.5)
            r = max(r,0.5)

            p1 = np.array([r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                           r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
                           z]) + fasta_noise(idx,dim)
            p2 = np.array([r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
                           r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
                           z]) + fasta_noise(idx,dim)

            cell["accum_s1"].append(p1)
            cell["accum_s2"].append(p2)
            if len(cell["accum_s1"]) > max_points:
                drop = len(cell["accum_s1"])/3
                cell["accum_s1"] = cell["accum_s1"][int(drop):]
                cell["accum_s2"] = cell["accum_s2"][int(drop):]

            strand1_vis.set_data(np.array(cell["accum_s1"]))
            strand2_vis.set_data(np.array(cell["accum_s2"]))
        frame = cells[-1]["frame"]
    else:
        frame = 0
    # Organelle update block (fix indentation and undefined variables)
    for cell in cells:
        to_remove = []
        for org in cell.get("organelles", []):
            new_pts = []
            for p in org["positions"]:
                dir_vec = np.random.normal(0, 1, 3)
                decay = 0.01
                new_p = p + dir_vec * 0.015
                def update(ev):
                    global frame
                    new_cells = []
                    if len(cells) > 0:
                        for cell in cells:
                            cell["frame"] += 1
                            frame = cell["frame"]
                            idx = frame % genome_len
                            base = genome_seq[idx]
                            dim = base_map.get(base,0)

                            theta = idx * np.radians(golden_angle_deg)
                            twist = idx / genome_len * twist_factor
                            z = np.sin(idx/genome_len * np.pi * 6) * 3 + (idx/genome_len)*12
                            a1 = np.radians(angles[dim])
                            a2 = np.radians(-angles[dim])
                            r = core_radius * (1 - (idx/genome_len)**1.5)
                            r = max(r,0.5)

                            p1 = np.array([r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                                           r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
                                           z]) + fasta_noise(idx,dim)
                            p2 = np.array([r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
                                           r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
                                           z]) + fasta_noise(idx,dim)

                            cell["accum_s1"].append(p1)
                            cell["accum_s2"].append(p2)
                            if len(cell["accum_s1"]) > max_points:
                                drop = len(cell["accum_s1"])/3
                                cell["accum_s1"] = cell["accum_s1"][int(drop):]
                                cell["accum_s2"] = cell["accum_s2"][int(drop):]

                            strand1_vis.set_data(np.array(cell["accum_s1"]))
                            strand2_vis.set_data(np.array(cell["accum_s2"]))
                        frame = cells[-1]["frame"]
                    else:
                        frame = 0
                    view.camera.azimuth = frame * 0.15
                    view.camera.elevation = 20 + 10*np.sin(frame*0.003)
                    percent = min(sum([c["frame"] for c in cells])/genome_len*100,100) if len(cells) > 0 else 0
                    progress_text.text = f"{percent:.2f}%"
                    canvas.update()

print("Human Genome GRCh38.p14 Visualization")
print("="*60)
if 'genome_len' in dir():
    print(f"Genome length: {genome_len:,} nucleotides")
print("="*60)

# Register update function with timer
timer = app.Timer(interval=0.016, connect=update, start=True)
canvas.show()
app.run()
