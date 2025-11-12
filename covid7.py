# covid_echo_genome_packed.py
# --------------------------------------------------------------
# SARS-CoV-2 Wuhan-Hu-1 genome visualization
# GPU-accelerated double φ-spiral encoding the entire viral genome
# coloured by nucleotide, tightly packed, with closed lattices,
# inter-links, echoes, and final φ-core convergence.
# Auto-detects FASTA file from ncbi_dataset
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
golden_angle_deg = 360 / (phi ** 2)  # 137.507°

# nucleotide → geometry mapping (RNA bases for coronavirus)
base_map = {'A': 5, 'U': 2, 'G': 4, 'C': 1, 'T': 2}  # T treated as U for RNA

# geometries: (dim, note, color, name, alpha, vertices)
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
speed_factor = 5.0

# ---------- AUTO-DETECT AND LOAD GENOME ----------
def find_covid_fasta():
    """Automatically find the COVID-19 FASTA file in the ncbi_dataset"""
    possible_paths = [
        r"ncbi_dataset\data\GCF_009858895.2\*.fna",
        r"ncbi_dataset\data\GCA_009858895.3\*.fna",
        r"ncbi_dataset\data\*\*.fna",
    ]

    for pattern in possible_paths:
        files = glob.glob(pattern)
        if files:
            return files[0]

    raise FileNotFoundError("Could not find COVID-19 FASTA file in ncbi_dataset directory")

def load_genome(fasta_file):
    seq = []
    print(f"Loading SARS-CoV-2 genome from: {fasta_file}")
    with open(fasta_file) as f:
        header = ""
        for line in f:
            if line.startswith(">"):
                header = line.strip()
                print(f"Sequence: {header}")
                continue
            seq.extend(list(line.strip().upper()))
    return seq, header

# Load the coronavirus genome
try:
    covid_fasta_path = find_covid_fasta()
    genome_seq, genome_header = load_genome(covid_fasta_path)
    genome_len = len(genome_seq)
    print(f"Genome loaded: {genome_len:,} nucleotides")
except Exception as e:
    print(f"Error loading genome: {e}")
    print("Using mock sequence...")
    genome_seq = list("ATTAAAGGTTTATACCTTCCCAGGTAACAAACCAACCAACTTTCGATCTCTTGTAGATCTGTTCTCTAAACGAACTTTAA" * 100)
    genome_header = ">Mock SARS-CoV-2 sequence"
    genome_len = len(genome_seq)

# ---------- VISPY ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1400, 900), bgcolor='#000011')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# strands - viral RNA strands
strand1 = Line(pos=np.zeros((1,3)), color=(1,0.8,0.2,0.8), width=2.5, parent=view.scene)
strand2 = Line(pos=np.zeros((1,3)), color=(0.2,0.8,1,0.8), width=2.5, parent=view.scene)

# collections
rungs, echoes, links, labels, centers, emerged = [], [], [], [], [], []

# Title and info
title_text = Text("SARS-CoV-2 Wuhan-Hu-1 Genome", pos=[0, 0, 25],
                  color='white', font_size=18, bold=True, anchor_x='center', parent=view.scene)
info_text = Text(f"{genome_len:,} nucleotides", pos=[0, 0, 22],
                 color='cyan', font_size=12, anchor_x='center', parent=view.scene)

frame = 0
core_radius = 5.0  # max distance from φ-core

# ---------- UPDATE ----------
def update(ev):
    global frame, rungs, echoes, links, labels, centers, emerged

    frame += 1
    # Shorter genome means we can visualize more quickly
    N = 400  # Reduced for viral genome
    t = np.linspace(0, frame, N)

    s1, s2 = [], []

    # ---------- VIRAL PACKING PARAMETERS ----------
    core_radius = 12.0       # smaller central φ-core for viral genome
    strand_sep = 0.4         # tighter separation for compact viral RNA
    twist_factor = 3*np.pi   # more twist for compact viral structure

    for tt in t:
        idx = int(tt) % genome_len
        base = genome_seq[idx]
        dim = base_map.get(base, 1) - 1

        _, _, col, _, alpha, verts = geometries[dim]

        # ---------- VIRAL RADIAL COMPRESSION ----------
        # Tighter packing for viral genome
        r = core_radius * (1 - (tt/genome_len)**1.3)
        r = max(r, 0.3)

        # ---------- 3D VIRAL TWISTING ----------
        theta = tt * np.radians(golden_angle_deg)  # φ-golden spiral
        twist = tt/genome_len * twist_factor       # enhanced z-axis rotation
        z = np.sin(tt/genome_len * np.pi * 6) * 1.5 + (tt/genome_len)*10  # tighter wrap

        # Strand 1 (sense strand)
        a1 = np.radians(angles[dim])
        s1.append([
            r*np.cos(theta)*np.cos(a1) - r*np.sin(theta)*np.sin(a1),
            r*np.sin(theta)*np.cos(a1) + r*np.cos(theta)*np.sin(a1),
            z
        ])

        # Strand 2 (antisense strand)
        a2 = np.radians(-angles[dim])
        s2.append([
            r*np.cos(theta)*np.cos(a2) - r*np.sin(theta)*np.sin(a2) + strand_sep,
            r*np.sin(theta)*np.cos(a2) + r*np.cos(theta)*np.sin(a2) - strand_sep,
            z
        ])

    # ---------- UPDATE STRANDS ----------
    strand1.set_data(np.array(s1))
    strand2.set_data(np.array(s2))

    # ---------- EMERGE RUNGS ----------
    cur_base = genome_seq[frame % genome_len]
    cur_dim = base_map.get(cur_base, 1) - 1

    # More frequent emergence for smaller genome
    if frame % 15 == 0 and cur_dim > len(rungs)-1:
        dim, note, col, name, alpha, verts = geometries[cur_dim]
        emerged.append(cur_dim)

        start = int((cur_dim/8.0)*N)
        step = max(1, N // (8*verts))
        idx = np.arange(start, start+verts*step, step)[:verts]
        idx = idx[idx < N]

        if len(idx) == 0:
            return

        pts1 = np.array(s1)[idx]
        pts2 = np.array(s2)[idx]
        rgba = Color(col).rgba
        edge_rgba = list(rgba[:3]) + [0.9]

        # ---------- CLOSED LATTICE ----------
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
            rungs.append(line)

        all_pts = np.vstack((pts1, pts2))
        mark = Markers(pos=all_pts, face_color=rgba, edge_color='white',
                       size=9, parent=view.scene)
        rungs.append(mark)

        cen = all_pts.mean(axis=0)
        centers.append(cen)
        lbl = Text(f"{cur_base}: {name}", pos=cen + [0,0,0.3],
                   color=col, font_size=9, bold=True,
                   anchor_x='center', parent=view.scene)
        labels.append(lbl)

        # ---------- VIRAL ECHO ----------
        if len(emerged) > 1:
            prev = emerged[-2]
            prev_col = list(Color(geometries[prev][2]).rgba)
            prev_col[3] = 0.3  # More visible echoes
            echo_pts = all_pts * 0.8
            echo = Markers(pos=echo_pts, face_color=prev_col,
                           size=6, parent=view.scene)
            echoes.append(echo)

        # ---------- INTER-RUNG LINKS (Protein connections) ----------
        if len(centers) > 1:
            prev_c = centers[-2]
            segs = []
            for i in range(min(8, len(all_pts))):  # More connections
                segs += [prev_c, cen]
            link = Line(pos=np.array(segs), color=(0.8,0.8,0.2,0.5),
                        width=1.5, connect='segments', parent=view.scene)
            links.append(link)

    # ---------- CAMERA ----------
    view.camera.azimuth = frame * 0.25
    view.camera.elevation = 25 + 8*np.sin(frame*0.004)
    view.camera.distance = 40 + 5*np.sin(frame*0.003)

    # Update progress
    progress = (frame % genome_len) / genome_len * 100
    info_text.text = f"{genome_len:,} nucleotides | Progress: {progress:.1f}%"

    canvas.update()

timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SARS-CoV-2 Wuhan-Hu-1 Genome Visualization")
    print("="*60)
    print(f"Genome length: {genome_len:,} nucleotides")
    print("Double φ-spiral encoding with viral RNA characteristics")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Mouse wheel: Zoom")
    print("  - ESC: Exit")
    print("="*60)
    canvas.show()
    app.run()
