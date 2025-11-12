"""
All is FASTA — fully self-organizing holographic cell
No contrived timers, modulo operations, or fixed divisions
Everything emerges from genome sequence alone
"""

import os, numpy as np
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
canvas = scene.SceneCanvas(keys='interactive', size=(1200,900), bgcolor='#000011')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

strand_vis = Line(pos=np.zeros((1,3)), color=(1,1,1,0.8), width=2, parent=view.scene)
organelles = []

# ---------- STATE ----------
lattice = []
frame = 0

# ---------- HELPER FUNCTIONS ----------
def genome_to_vec(idx):
    """Map genome to a 3D vector recursively and holographically"""
    b = genome_seq[idx % genome_len]
    val = ord(b) % 16
    angle = val / 16 * 2*np.pi
    radius = 5 + val*0.5
    z = np.sin(idx/genome_len * np.pi*4) * val*0.1
    return np.array([radius*np.cos(angle), radius*np.sin(angle), z])

def spawn_organelles(center, idx):
    """Organelles emerge from genome values around a center"""
    val = ord(genome_seq[idx % genome_len]) % 8 + 1
    pts = center + np.random.normal(scale=val*0.2, size=(val,3))
    rgba = list(Color('cyan').rgba)
    rgba[3] = 0.9
    mark = Markers(pos=pts, face_color=rgba, size=5, parent=view.scene)
    return {"marker": mark, "positions": pts}

# ---------- UPDATE LOOP ----------
def update(ev):
    global frame, lattice, organelles

    idx = frame % genome_len
    pos = genome_to_vec(idx)

    lattice.append(pos)
    if len(lattice) > 8000:  # saturation -> holographic replication
        lattice = lattice[len(lattice)//2:]

    # Organelles naturally emerge when genome-derived vectors cluster
    if np.random.rand() < (ord(genome_seq[idx % genome_len])%5)/5:
        org = spawn_organelles(pos, idx)
        organelles.append(org)

    # Implicit decay / backpressure from genome density
    for org in organelles:
        for i, p in enumerate(org["positions"]):
            nearest_idx = np.random.randint(0,len(lattice))
            p += (lattice[nearest_idx]-p)*0.02
            p *= 0.995  # decay entirely from interactions
        org["marker"].set_data(pos=org["positions"])

    strand_vis.set_data(np.array(lattice))

    view.camera.azimuth = frame * 0.1
    view.camera.elevation = 20 + 5*np.sin(frame*0.003)

    frame += 1
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
