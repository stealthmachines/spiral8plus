# 🌌 DNA φ-Harmonic Spiral Split-Test (24 Mappings)
# --------------------------------------------------------------
# Visualize all 24 possible DNA → geometry mappings
# Install: pip install vispy pyqt6
# --------------------------------------------------------------

import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color
import itertools

# ---------- FRAMEWORK ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)  # 137.507°

# Generate all 24 permutations of DNA → 4 geometry indices
dna_bases = ['A', 'T', 'G', 'C']
geometry_indices = [1, 2, 4, 5]  # Use first 4 geometries
all_perms = list(itertools.permutations(geometry_indices))
mappings = [dict(zip(dna_bases, perm)) for perm in all_perms]

# ---------- GEOMETRIES ----------
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
period = 13.057
t_max = period * 8
speed_factor = 5.0

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
# Create canvases for all 24 mappings (display first 4 for performance)
canvases = []
views = []
strands = []
collections = []

num_to_display = min(4, len(mappings))  # Display first 4 mappings

for i in range(num_to_display):
    canvas = scene.SceneCanvas(keys='interactive', size=(600, 600), bgcolor='#000011', title=f'Mapping {i+1}')
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'

    # Strands for this canvas
    strand1 = Line(pos=np.zeros((1, 3)), color=(1, 1, 1, 0.7), width=2, parent=view.scene)
    strand2 = Line(pos=np.zeros((1, 3)), color=(1, 1, 1, 0.7), width=2, parent=view.scene)

    # Store
    canvases.append(canvas)
    views.append(view)
    strands.append((strand1, strand2))
    collections.append(Markers(parent=view.scene))

# Variables
core_radius = 15.0
strand_sep = 0.5
twist_factor = 2 * np.pi
frame = 0

rungs, echoes, links, labels, centers, emerged = [], [], [], [], [], []
def update(ev):
    global frame
    frame += 1
    N = 600
    t = np.linspace(0, frame, N)

    # Process only the displayed mappings
    for m_idx in range(len(views)):
        base_map = mappings[m_idx]
        s1, s2 = [], []
        view = views[m_idx]
        strand1, strand2 = strands[m_idx]
        coll = collections[m_idx]

        for tt in t:
            idx = int(tt) % genome_len
            base = genome_seq[idx]
            dim = base_map.get(base, 1) - 1

            _, _, col, _, alpha, verts = geometries[dim]
            r = core_radius * (1 - (tt/genome_len)**1.5)
            r = max(r, 0.5)
            theta = tt * np.radians(golden_angle_deg)
            twist = tt/genome_len * twist_factor
            z = np.sin(tt/genome_len * np.pi * 4) * 2 + (tt/genome_len)*8

            a1 = np.radians(angles[dim])
            a2 = np.radians(-angles[dim])

            s1.append([r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                       r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
                       z])
            s2.append([r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
                       r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
                       z])

        # Update strands
        strand1.set_data(np.array(s1))
        strand2.set_data(np.array(s2))

        # ---------- EMERGE RUNGS ----------
        cur_base = genome_seq[frame % genome_len]
        cur_dim = base_map.get(cur_base, 1) - 1

        if frame % 20 == 0 and cur_dim > len(coll['rungs'])-1:
            dim, note, col, name, alpha, verts = geometries[cur_dim]
            coll['emerged'].append(cur_dim)
            start = int((cur_dim/8.0)*N)
            step = max(1, N // (8*verts))
            idxs = np.arange(start, start+verts*step, step)[:verts]
            idxs = idxs[idxs < N]

            pts1 = np.array(s1)[idxs]
            pts2 = np.array(s2)[idxs]
            rgba = Color(col).rgba
            edge_rgba = list(rgba[:3]) + [0.9]

            # Closed lattice
            segs = []
            if verts <= 4:
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
            else:
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
                coll['rungs'].append(line)

            # Vertices
            all_pts = np.vstack((pts1, pts2))
            mark = Markers(pos=all_pts, face_color=rgba, edge_color='white',
                           size=8, parent=view.scene)
            coll['rungs'].append(mark)

            # Center & label
            cen = all_pts.mean(axis=0)
            coll['centers'].append(cen)
            lbl = Text(f"{cur_base}: {name}", pos=cen + [0,0,0.3],
                       color=col, font_size=10, bold=True,
                       anchor_x='center', parent=view.scene)
            coll['labels'].append(lbl)

            # Echo
            if len(coll['emerged']) > 1:
                prev = coll['emerged'][-2]
                prev_col = list(Color(geometries[prev][2]).rgba)
                prev_col[3] = 0.25
                echo_pts = all_pts * 0.75
                echo = Markers(pos=echo_pts, face_color=prev_col,
                               size=5, parent=view.scene)
                coll['echoes'].append(echo)

            # Inter-rung links
            if len(coll['centers']) > 1:
                prev_c = coll['centers'][-2]
                segs = []
                for i in range(min(6, len(all_pts))):
                    segs += [prev_c, cen]
                link = Line(pos=np.array(segs), color=(0.7,0.7,0.7,0.4),
                            width=1, connect='segments', parent=view.scene)
                coll['links'].append(link)

        # Camera rotation
        view.camera.azimuth = frame * 0.3
        view.camera.elevation = 20 + 5*np.sin(frame*0.005)
        canvases[m_idx].update()

# ---------- TIMER ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    for canvas in canvases:
        canvas.show()
    app.run()
