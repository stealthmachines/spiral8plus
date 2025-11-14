# dna_holo_infinium.py
# --------------------------------------------------------------
# Fully FASTA-driven holographic φ-spiral cell simulation
# Infinite emergent behavior (division, lattice, organelles)
# Install: pip install vispy pyqt6 numpy
# Run: python dna_holo_infinium.py
# --------------------------------------------------------------

import os
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers
from vispy.color import Color

# ---------- CONSTANTS ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)
core_radius_base = 15.0
strand_sep_base = 0.5

# nucleotide → geometry mapping
base_map = {'A': 5, 'T': 2, 'G': 4, 'C': 1}
geometries = [
    (1,'C','red','Point',1),
    (2,'D','green','Line',2),
    (3,'E','violet','Triangle',3),
    (4,'F','mediumpurple','Tetrahedron',4),
    (5,'G','blue','Pentachoron',5),
    (6,'A','indigo','Hexacross',12),
    (7,'B','purple','Heptacube',14),
    (8,'C','white','Octacube',16),
]
angles = [i*golden_angle_deg for i in range(8)]

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

def consensus_dim(b):
    return base_map.get(b,1)-1

def organelle_params(b):
    d = consensus_dim(b)
    return 0.02 + d/16*0.12, 6 + (d%5)*3, 0.2 + d/16*0.9, 0.08 + d/16*0.4

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
donut_vis = Line(pos=np.zeros((1, 3)), color='yellow', width=1, parent=view.scene)

# Initialize state
cells = [{"lattice": [], "organelles": [], "genome_idx": 0, "genome_start": 0, "accum_s1": [], "accum_s2": [], "accum_donut": []}]

def spawn_organelle(center, color, n=12, size=0.5):
    pts = center + np.random.normal(scale=0.2,size=(n,3))*size
    rgba = list(Color(color).rgba)
    rgba[3]=1.0
    mark = Markers(pos=pts, face_color=rgba, edge_color=None, size=5,parent=view.scene)
    return {"marker": mark, "positions": pts, "color": rgba}

def lattice_push(pt, nodes, positive=True, strength=0.02):
    if len(nodes)==0: return pt
    nearest = nodes[np.random.randint(0,len(nodes))]
    vec = nearest - pt
    if not positive: vec*=-1
    return pt + vec*strength

def division_motif(seq_window):
    return seq_window==['A','T','G']  # simple motif triggers division

# ---------- UPDATE LOOP ----------
frame = 0

def update(ev):
    global frame, cells
    frame += 1
    new_cells=[]
    for cell in cells:
        idx = (cell["genome_start"] + frame) % genome_len
        base = genome_seq[idx]
        dim = consensus_dim(base)

        # φ-spiral positions
        r = core_radius_base*(1-(idx/genome_len)**1.4)
        r=max(r,0.5)
        theta = np.radians(golden_angle_deg)*idx
        z = np.sin(idx/genome_len*np.pi*4)*1.6+(idx/genome_len)*6.0
        a1 = np.radians(angles[dim])
        a2 = np.radians(-angles[dim])
        p1 = np.array([r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                       r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1), z])
        p2 = np.array([r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep_base,
                       r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep_base, z])
        # drift/noise (FASTA-driven)
        p1 += np.random.normal(scale=0.003,size=3)+0.02*np.cos(frame*0.005)
        p2 += np.random.normal(scale=0.003,size=3)-0.02*np.cos(frame*0.005)
        p1[2]+=np.sin(frame*0.003)*0.5
        p2[2]-=np.sin(frame*0.003)*0.3

        cell["accum_s1"].append(p1)
        cell["accum_s2"].append(p2)
        donut = (p1+p2)/2
        cell["accum_donut"].append(donut)

        # organelles
        sp, n, rb, ns = organelle_params(base)
        if np.random.rand()<sp:
            color = geometries[min(dim,len(geometries)-1)][2]
            cell["organelles"].append(spawn_organelle(donut,color,n,0.8+rb))

        # lattice push
        nodes = np.array(cell["accum_s1"]+cell["accum_s2"])
        for org in cell["organelles"]:
            pos = base in ['A','T']
            new_pts=np.array([lattice_push(p,nodes,pos,0.02) for p in org['positions']])
            org['positions']=new_pts
            org['marker'].set_data(pos=new_pts,face_color=org['color'],size=5)

        # update visuals
        strand1_vis.set_data(np.array(cell["accum_s1"]))
        strand2_vis.set_data(np.array(cell["accum_s2"]))
        if len(cell["accum_donut"])>1:
            donut_vis.set_data(np.array(cell["accum_donut"]))

        # division triggered by FASTA motif
        window=genome_seq[idx:idx+3]
        if len(window)==3 and division_motif(window):
            daughter={"accum_s1":cell["accum_s1"][-10:].copy(),
                      "accum_s2":cell["accum_s2"][-10:].copy(),
                      "accum_donut":cell["accum_donut"][-10:].copy(),
                      "organelles":[],"genome_start":idx}
            new_cells.append(daughter)

    cells.extend(new_cells)

    view.camera.azimuth = frame*0.12
    view.camera.elevation = 15+8*np.sin(frame*0.002)
    canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)
if __name__=='__main__':
    canvas.show()
    app.run()
