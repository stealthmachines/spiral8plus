# human_spiral9.py
# --------------------------------------------------------------
# Human genome (GRCh38.p14) visualization
# GPU-accelerated double phi-spiral with colour, closed lattices,
# inter-shape links and infinite echoing back to the source.
# Optimized for viral RNA genome characteristics
# Install: pip install vispy pyqt6 numpy
# --------------------------------------------------------------

import os
import glob
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- FRAMEWORK ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)               # 137.507°

# DNA-optimized geometries with human genome color scheme
geometries = [
    (1, 'C', 'red',          'Point',        0.015269, 1),
    (2, 'D', 'orange',       'Line',         0.008262, 2),
    (3, 'E', 'yellow',       'Triangle',     0.110649, 3),
    (4, 'F', 'lime',         'Tetrahedron', -0.083485, 4),
    (5, 'G', 'cyan',         'Pentachoron',  0.025847, 5),
    (6, 'A', 'blue',         'Hexacross',   -0.045123, 12),
    (7, 'B', 'purple',       'Heptacube',    0.067891, 14),
    (8, 'C', 'magenta',      'Octacube',     0.012345, 16),
]

angles = [i * golden_angle_deg for i in range(8)]

period = 13.057
t_max = period * 8
speed_factor = 6.0          # fast emergence for human genome

# ---------- AUTO-DETECT AND LOAD GENOME ----------
def find_human_fasta():
    """Automatically find the Human Genome FASTA file (GRCh38.p14) in the ncbi_dataset"""
    possible_paths = [
        r"ncbi_dataset\ncbi_dataset\data\GCF_000001405.40\*.fna",
        r"ncbi_dataset\data\GCF_000001405.40\*.fna",
        r"ncbi_dataset\data\*\*.fna",
    ]

    for pattern in possible_paths:
        files = glob.glob(pattern)
        if files:
            return files[0]

    raise FileNotFoundError("Could not find Human Genome FASTA file (GCF_000001405.40) in ncbi_dataset directory")

def load_genome(fasta_file, max_nucleotides=None, chromosome=None):
    """
    Load genome sequence from FASTA file with flexible nucleotide limiting

    Args:
        fasta_file: Path to FASTA file
        max_nucleotides: Maximum nucleotides to load (None = use GENOME_LIMIT env var, default 100000)
        chromosome: Specific chromosome to load (e.g., "chr1", "NC_000001.11")

    Returns:
        tuple: (sequence_list, metadata_dict)
    """
    # Check environment variable for user preference
    env_limit = os.environ.get('GENOME_LIMIT', '100000')

    if max_nucleotides is None:
        if env_limit == 'all':
            max_nucleotides = None  # Load everything
        else:
            try:
                max_nucleotides = int(env_limit)
            except ValueError:
                max_nucleotides = 100000

    seq = []
    current_chr = None
    metadata = {'chromosomes': [], 'total_loaded': 0}

    print(f"Loading Human genome from: {fasta_file}")
    with open(fasta_file, encoding='utf-8') as f:
        for line in f:
            if line.startswith(">"):
                chr_name = line.strip()[1:].split()[0]
                if current_chr is None:
                    current_chr = chr_name
                    metadata['chromosomes'].append(chr_name)
                    print(f"Sequence: {line.strip()}")
                elif chromosome is None or chromosome in chr_name:
                    # Loading multiple chromosomes
                    current_chr = chr_name
                    metadata['chromosomes'].append(chr_name)
                continue

            if chromosome is None or (current_chr and chromosome in current_chr):
                seq.extend(list(line.strip().upper()))

                # Check if we've reached the limit
                if max_nucleotides and len(seq) >= max_nucleotides:
                    seq = seq[:max_nucleotides]
                    metadata['total_loaded'] = len(seq)
                    print(f"  Loaded {max_nucleotides:,} nucleotides from {metadata['chromosomes'][0]}")
                    print(f"  (limited to {max_nucleotides:,}, set GENOME_LIMIT env var to change)")
                    return seq, metadata

    metadata['total_loaded'] = len(seq)
    if max_nucleotides is None:
        print(f"  Loaded {len(seq):,} nucleotides from {len(metadata['chromosomes'])} chromosome(s)")

    return seq, metadata

# Load the human genome
try:
    human_fasta_path = find_human_fasta()
    genome_seq, metadata = load_genome(human_fasta_path)
    genome_len = len(genome_seq)
    print(f"Genome loaded: {genome_len:,} nucleotides")
except Exception as e:
    print(f"Warning: {e}")
    print("Running without genome data (pure mathematical visualization)")
    genome_seq = None
    genome_len = 29903

# ---------- VISPY ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1400, 900), bgcolor='#000011')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Strands - viral RNA color scheme
strand1 = Line(pos=np.zeros((1, 3)), color=(1,0.8,0.2,0.8), width=2.5, parent=view.scene)
strand2 = Line(pos=np.zeros((1, 3)), color=(0.2,0.8,1,0.8), width=2.5, parent=view.scene)

# Title and info
title_text = Text("Human Genome GRCh38 - Echoing phi-Spiral",
                  pos=[0, 0, 8], color='white', font_size=16, bold=True,
                  anchor_x='center', parent=view.scene)
info_text = Text(f"{genome_len:,} nucleotides",
                 pos=[0, 0, 6.5], color='cyan', font_size=12,
                 anchor_x='center', parent=view.scene)

# Collections
rungs      = []   # closed lattice objects
echoes     = []   # faint repetitions (viral protein echoes)
links      = []   # inter-rung connections (RNA-protein interactions)
labels     = []
centers    = []   # rung centres for linking
emerged    = []   # dimensions that have appeared

frame = 0

def update(ev):
    global frame, rungs, echoes, links, labels, centers, emerged

    frame += 1

    # ---- spiral growth (never stops - infinite viral replication) ----
    N = 500  # Optimized for human genome
    t = np.linspace(0, (frame / 360.0) * t_max * speed_factor, N)

    s1, s2 = [], []
    for tt in t:
        dim = min(int((tt * speed_factor) // period), 7)
        _, _, col, _, alpha, _ = geometries[dim]

        r = np.exp(alpha * (tt % period))
        theta = tt * 2 * np.pi / period

        # strand 1 (sense RNA)
        a1 = np.radians(angles[dim])
        s1.append([r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                   r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
                   (tt/period)*0.9])

        # strand 2 (antisense RNA - counter-rotate)
        a2 = np.radians(-angles[dim])
        s2.append([r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2),
                   r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2),
                   (tt/period)*0.9])

    strand1.set_data(pos=np.array(s1))
    strand2.set_data(pos=np.array(s2))

    # ---- emerge new rung every 18 frames (faster for human genome) ----
    cur_dim = min(int((t[-1] * speed_factor) // period), 7)
    if frame % 18 == 0 and cur_dim > len(rungs)-1:
        dim, note, col, name, alpha, verts = geometries[cur_dim]
        emerged.append(cur_dim)

        # sample points from current segment
        start = int((cur_dim/8.0)*N)
        step  = max(1, N // (8*verts))
        idx   = np.arange(start, start+verts*step, step)[:verts]
        idx   = idx[idx < N]

        if len(idx) == 0:
            canvas.update()
            return

        pts1 = np.array(s1)[idx]
        pts2 = np.array(s2)[idx]

        rgba = Color(col).rgba
        edge_rgba = list(rgba[:3]) + [0.9]

        # ----- closed lattice (viral protein complexes) -----
        segs = []
        if verts <= 4:                     # low-D: complete graph + faces
            for pts in (pts1, pts2):
                for i in range(verts):
                    for j in range(i+1, verts):
                        segs += [pts[i], pts[j]]
            for i in range(verts):
                segs += [pts1[i], pts2[i]]
                if verts > 2:
                    for k in range(1, verts):
                        segs += [pts1[i], pts1[(i+k)%verts],
                                 pts2[i], pts2[(i+k)%verts]]
        else:                              # high-D: grid lattice
            g = int(np.ceil(np.sqrt(verts)))
            for i in range(g):
                for j in range(g):
                    n = min(i*g + j, verts-1)
                    if j+1 < g and n+1 < verts:
                        segs += [pts1[n], pts1[n+1], pts2[n], pts2[n+1]]
                    if i+1 < g and n+g < verts:
                        segs += [pts1[n], pts1[n+g], pts2[n], pts2[n+g]]
            for n in range(verts):
                segs += [pts1[n], pts2[n]]

        if segs:
            line = Line(pos=np.array(segs), color=edge_rgba, width=2,
                        connect='segments', parent=view.scene)
            rungs.append(line)

        # vertices (nucleotide positions)
        all_pts = np.vstack((pts1, pts2))
        mark = Markers(pos=all_pts, face_color=rgba, edge_color='white',
                       size=9, parent=view.scene)
        rungs.append(mark)

        # centre & label
        cen = all_pts.mean(axis=0)
        centers.append(cen)
        lbl = Text(f"{dim}D: {name}\n{note}", pos=cen + [0,0,0.3],
                   color=col, font_size=9, bold=True,
                   anchor_x='center', parent=view.scene)
        labels.append(lbl)

        # ----- echo back to source (viral protein assembly echoes) -----
        if len(emerged) > 1:
            prev = emerged[-2]
            prev_col = list(Color(geometries[prev][2]).rgba)
            prev_col[3] = 0.35  # More visible for viral structures
            echo_pts = all_pts * 0.78          # shrink toward origin
            echo = Markers(pos=echo_pts, face_color=prev_col,
                           size=6, parent=view.scene)
            echoes.append(echo)

        # ----- inter-rung links (RNA-protein interactions) -----
        if len(centers) > 1:
            prev_c = centers[-2]
            segs = []
            for i in range(min(8, len(all_pts))):  # More connections for viral proteins
                segs += [prev_c, cen]
            link = Line(pos=np.array(segs), color=(0.8,0.8,0.3,0.6),
                        width=1.5, connect='segments', parent=view.scene)
            links.append(link)

    # ---- auto-rotate with viral dynamics ----
    view.camera.azimuth   = frame * 0.25
    view.camera.elevation = 20 + 5*np.sin(frame*0.003)
    view.camera.distance = 25 + 3*np.sin(frame*0.002)

    # Update info
    progress = (frame % 1000) / 10.0  # Cycling progress for infinite loop
    info_text.text = f"{genome_len:,} nucleotides | Frame: {frame}"

    canvas.update()

timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("Human Genome GRCh38 Echoing phi-Spiral Visualization")
    print("="*70)
    print(f"Genome: {genome_len:,} nucleotides")
    print("Infinite viral replication with echoing protein assemblies")
    print("Double phi-spiral with closed lattices and inter-shape connections")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Mouse wheel: Zoom")
    print("  - ESC: Exit")
    print("="*70)
    canvas.show()
    app.run()
