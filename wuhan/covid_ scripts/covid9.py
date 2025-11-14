# covid_echo_fullvirion.py
# --------------------------------------------------------------
# SARS-CoV-2 Wuhan-Hu-1 complete viral particle visualization
# GPU-accelerated full virion with tightly packed RNA genome
# Shows % complete and φ-core convergence in viral capsid context
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

base_map = {'A': 5, 'U': 2, 'G': 4, 'C': 1, 'T': 2}  # T treated as U for RNA
geometries = [
    (1,'C','red','Point',0.015269,1),
    (2,'D','orange','Line',0.008262,2),
    (3,'E','yellow','Triangle',0.110649,3),
    (4,'F','lime','Tetrahedron',-0.083485,4),
    (5,'G','cyan','Pentachoron',0.025847,5),
    (6,'A','blue','Hexacross',-0.045123,12),
    (7,'B','purple','Heptacube',0.067891,14),
    (8,'C','magenta','Octacube',0.012345,16),
]
angles = [i*golden_angle_deg for i in range(8)]

# ---------- VIRAL PARTICLE PARAMETERS ----------
virion_length = 15.0     # scaled units for viral capsid length
virion_radius = 4.0      # capsid radius
num_strands = 1          # single-stranded RNA genome
strand_sep = 0.15        # minimal separation for ssRNA

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

# RNA strands (single-stranded RNA, but we can show multiple for visual effect)
strands = [Line(pos=np.zeros((1,3)), color=(1,0.7+i*0.1,0.2,0.8), width=2.5, parent=view.scene) for i in range(num_strands)]
rungs, echoes, links, labels, centers, emerged = [], [], [], [], [], []

# Title and progress
title_text = Text("SARS-CoV-2 Wuhan-Hu-1 Virion", pos=(0, 0, virion_length+3),
                  color='white', font_size=18, bold=True, anchor_x='center', parent=view.scene)
progress_label = Text("0.0%", pos=(0, 0, virion_length+1),
                      color='cyan', font_size=16, bold=True, anchor_x='center', parent=view.scene)
info_text = Text(f"{genome_len:,} nucleotides", pos=(0, 0, virion_length-1),
                 color='lime', font_size=11, anchor_x='center', parent=view.scene)

frame = 0
speed_factor = 5.0

# ---------- UPDATE ----------
def update(ev):
    global frame
    frame += 1
    N = 400  # Reduced for viral genome
    t = np.linspace(0, frame, N)

    # Clear strand data
    s_all = [[] for _ in range(num_strands)]

    for tt in t:
        idx = int(tt) % genome_len
        base = genome_seq[idx]
        dim = base_map.get(base,1)-1
        _,_,col,_,alpha,verts = geometries[dim]

        # Viral radial packing: compress toward core, stay within virion
        r = virion_radius * (1 - (tt/genome_len)**1.4)
        r = max(r, 0.2)

        theta = tt * np.radians(golden_angle_deg)
        twist = tt/genome_len * 5*np.pi  # Enhanced twist for viral RNA
        z = (tt/genome_len)*virion_length

        for strand_idx in range(num_strands):
            offset = strand_idx*strand_sep - (num_strands-1)*strand_sep/2
            a1 = np.radians(angles[dim])
            s_all[strand_idx].append([
                r*np.cos(theta)*np.cos(a1) - r*np.sin(theta)*np.sin(a1) + offset,
                r*np.sin(theta)*np.cos(a1) + r*np.cos(theta)*np.sin(a1) + offset,
                z
            ])

    # update strands
    for i in range(num_strands):
        strands[i].set_data(np.array(s_all[i]))

    # ---------- EMERGE RUNGS ----------
    cur_base = genome_seq[frame % genome_len]
    cur_dim = base_map.get(cur_base,1)-1

    if frame % 15 == 0 and cur_dim > len(rungs)-1:
        dim, note, col, name, alpha, verts = geometries[cur_dim]
        emerged.append(cur_dim)
        start = int((cur_dim/8.0)*N)
        step = max(1,N//(8*verts))
        idx = np.arange(start,start+verts*step,step)[:verts]
        idx = idx[idx<N]

        if len(idx) == 0:
            canvas.update()
            return

        pts1 = np.array(s_all[0])[idx]
        pts2 = np.array(s_all[0])[idx] + 0.1  # slight offset for visual
        rgba = Color(col).rgba
        edge_rgba = list(rgba[:3])+[0.9]

        # closed lattice
        segs=[]
        for pts in (pts1, pts2):
            for i in range(verts):
                for j in range(i+1,verts):
                    segs += [pts[i], pts[j]]
        for i in range(verts):
            segs += [pts1[i], pts2[i]]
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
        lbl = Text(f"{cur_base}:{name}", pos=cen+[0,0,0.3], color=col,
                   font_size=9, bold=True, anchor_x='center', parent=view.scene)
        labels.append(lbl)

        # Echo (viral protein complexes)
        if len(emerged)>1:
            prev = emerged[-2]
            prev_col = list(Color(geometries[prev][2]).rgba)
            prev_col[3]=0.3
            echo_pts = all_pts*0.8
            echo = Markers(pos=echo_pts, face_color=prev_col, size=6, parent=view.scene)
            echoes.append(echo)

        # Inter-rung links (RNA-protein interactions)
        if len(centers)>1:
            prev_c = centers[-2]
            segs=[]
            for i in range(min(8,len(all_pts))):
                segs+=[prev_c,cen]
            link=Line(pos=np.array(segs), color=(0.8,0.8,0.2,0.5),
                      width=1.5, connect='segments', parent=view.scene)
            links.append(link)

    # ---------- CAMERA ----------
    view.camera.azimuth = frame*0.25
    view.camera.elevation = 25+8*np.sin(frame*0.004)
    view.camera.distance = 35 + 5*np.sin(frame*0.003)

    # ---------- % COMPLETE ----------
    percent_complete = min(100, frame/genome_len*100)
    progress_label.text = f"{percent_complete:.1f}%"

    # Change color as it progresses
    if percent_complete < 25:
        progress_label.color = 'cyan'
    elif percent_complete < 50:
        progress_label.color = 'yellow'
    elif percent_complete < 75:
        progress_label.color = 'orange'
    else:
        progress_label.color = 'lime'

    canvas.update()

timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("SARS-CoV-2 Wuhan-Hu-1 Complete Virion Visualization")
    print("="*70)
    print(f"Genome length: {genome_len:,} nucleotides (single-stranded RNA)")
    print(f"Virion dimensions: {virion_length} x {virion_radius*2} units")
    print("Tightly packed viral RNA with φ-golden spiral encoding")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Mouse wheel: Zoom")
    print("  - ESC: Exit")
    print("="*70)
    canvas.show()
    app.run()
