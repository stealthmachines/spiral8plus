"""
FASTA Universe — holographic genome-driven multi-cell simulation
Everything emerges from the genome; nothing is contrived.
Supports multiple interacting cells, organelles, lattice backpressure, and division.
"""

import os
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers
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


def load_genome(fasta_file, max_nucleotides=None, chromosome=None, start_position=0):
    """Load genome with GENOME_LIMIT support"""
    import os

    if max_nucleotides is None:
        env_limit = os.environ.get('GENOME_LIMIT', '100000')
        max_nucleotides = None if env_limit == 'all' else int(env_limit)

    if chromosome is None:
        chromosome = os.environ.get('GENOME_CHROMOSOME', None)

    sequence = ""
    current_chr = None
    count = 0

    print(f"Loading genome (limit: {max_nucleotides:,})...")
    if chromosome:
        print(f"  Filtering: {chromosome}")

    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith(">"):
                current_chr = line.strip()[1:].split()[0]
                if chromosome and current_chr != chromosome:
                    continue
                if chromosome:
                    print(f"  Loading from {current_chr}...")
            else:
                if chromosome and current_chr != chromosome:
                    continue

                bases = line.strip().upper()
                remaining = max_nucleotides - count

                if remaining <= 0:
                    break

                sequence += bases[:remaining]
                count += len(bases[:remaining])

                if count >= max_nucleotides:
                    break

    print(f"  Loaded: {count:,} nucleotides")
    return sequence


# ---------- LOAD GENOME ----------
fasta_path = find_human_fasta()
genome_seq = load_genome(fasta_path)
genome_len = len(genome_seq)
print(f"Genome length: {genome_len:,}")

# ---------- VISUAL SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1400,1000), bgcolor='#000011')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# ---------- STATE ----------
phi = (1 + np.sqrt(5))/2
max_points_per_cell = 8000

# Each cell has its own lattice, organelles, and genome position
cells = [{"lattice": [], "organelles": [], "frame": 0}]

# ---------- HELPER FUNCTIONS ----------
def genome_to_vec(idx):
    """Convert genome index to a 3D vector holographically from the sequence"""
    b = genome_seq[idx % genome_len]
    val = ord(b) % 16 + 1
    angle = val/16*2*np.pi
    radius = 5 + val*0.5
    z = np.sin(idx/genome_len * np.pi*4) * val*0.15
    return np.array([radius*np.cos(angle), radius*np.sin(angle), z])

# Create initial visual (will be populated by update loop)
strand_vis = Line(pos=np.zeros((2, 3)), color=(1,1,1,0.7), width=2, parent=view.scene)

def spawn_organelles(center, idx):
    """Organelles emerge from genome-driven randomness"""
    val = (ord(genome_seq[idx % genome_len]) % 8) + 1
    pts = center + np.random.normal(scale=val*0.2, size=(val,3))
    rgba = list(Color('cyan').rgba)
    rgba[3] = 0.9
    mark = Markers(pos=pts, face_color=rgba, size=5, parent=view.scene)
    return {"marker": mark, "positions": pts}

def check_division(cell):
    """Implicit genome-driven cell division"""
    if len(cell["lattice"]) > max_points_per_cell:
        half = len(cell["lattice"])//2
        new_cell = {
            "lattice": cell["lattice"][half:],
            "organelles": [],
            "frame": cell["frame"]
        }
        cell["lattice"] = cell["lattice"][:half]
        return new_cell
    return None

# ---------- UPDATE LOOP ----------
def update(ev):
    global cells, strand_vis

    new_cells = []

    for cell in cells:
        idx = cell["frame"] % genome_len
        pos = genome_to_vec(idx)
        cell["lattice"].append(pos)

        # Maintain lattice size
        if len(cell["lattice"]) > max_points_per_cell:
            cell["lattice"] = cell["lattice"][len(cell["lattice"])//2:]

        # Organelles emerge naturally
        if np.random.rand() < (ord(genome_seq[idx % genome_len]) % 5)/5:
            org = spawn_organelles(pos, idx)
            cell["organelles"].append(org)

        # Holographic backpressure / decay
        for org in cell["organelles"]:
            for i, p in enumerate(org["positions"]):
                if len(cell["lattice"]) > 0:
                    nearest_idx = np.random.randint(0,len(cell["lattice"]))
                    p += (cell["lattice"][nearest_idx]-p)*0.02
                    p *= 0.995
            org["marker"].set_data(pos=org["positions"])

        # Genome-driven cell division
        offspring = check_division(cell)
        if offspring:
            new_cells.append(offspring)

        cell["frame"] += 1

    cells.extend(new_cells)

    # Merge all lattice points holographically for visualization
    if any(len(cell["lattice"]) > 0 for cell in cells):
        all_positions = np.vstack([cell["lattice"] for cell in cells if len(cell["lattice"]) > 0])
        strand_vis.set_data(all_positions)

    # Camera rotation driven holographically by genome frames
    total_frames = sum(cell["frame"] for cell in cells)
    view.camera.azimuth = total_frames * 0.07
    view.camera.elevation = 25 + 10*np.sin(total_frames*0.002)

    canvas.update()

# ---------- RUN ----------
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
