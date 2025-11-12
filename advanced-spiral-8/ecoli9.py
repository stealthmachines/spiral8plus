# dna_echo_fullcell.py
# --------------------------------------------------------------
# GPU-accelerated full E. coli cell visualization
# with tightly packed chromosome(s), % complete, and φ-core convergence
# Install: pip install vispy pyqt6
# --------------------------------------------------------------

import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- FRAMEWORK ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)  # 137.507°

base_map = {'A': 5, 'T': 2, 'G': 4, 'C': 1}
geometries = [
    (1,'C','red','Point',0.015269,1),
    (2,'D','green','Line',0.008262,2),
    (3,'E','violet','Triangle',0.110649,3),
    (4,'F','mediumpurple','Tetrahedron',-0.083485,4),
    (5,'G','blue','Pentachoron',0.025847,5),
    (6,'A','indigo','Hexacross',-0.045123,12),
    (7,'B','purple','Heptacube',0.067891,14),
    (8,'C','white','Octacube',0.012345,16),
]
angles = [i*golden_angle_deg for i in range(8)]

# ---------- CELL PARAMETERS ----------
cell_length = 20.0  # scaled units
cell_radius = 5.0
num_strands = 1      # single chromosome; increase for multiple spirals
strand_sep = 0.2     # separation between strands

# ---------- LOAD GENOME ----------
def load_genome(fasta_file):
    seq = []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"): continue
            seq.extend(list(line.strip().upper()))
    return seq

genome_seq = load_genome("ecoli_k12.fasta")
genome_len = len(genome_seq)

# ---------- VISPY ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# DNA strands
strands = [Line(pos=np.zeros((1,3)), color=(1,1,1,0.7), width=2, parent=view.scene) for _ in range(num_strands)]
rungs, echoes, links, labels, centers, emerged = [], [], [], [], [], []

# % complete label
progress_label = Text("0%", pos=(0,0,cell_length+2), color='white', font_size=20, parent=view.scene)

frame = 0
speed_factor = 5.0

# ---------- UPDATE ----------
def update(ev):
    global frame
    frame += 1
    N = 600
    t = np.linspace(0, frame, N)

    # Clear strand data
    s_all = [[] for _ in range(num_strands)]

    for tt in t:
        idx = int(tt) % genome_len
        base = genome_seq[idx]
        dim = base_map.get(base,1)-1
        _,_,col,_,alpha,verts = geometries[dim]

        # radial packing: compress toward core, stay within cell
        r = cell_radius * (1 - (tt/genome_len)**1.5)
        r = max(r, 0.3)

        theta = tt * np.radians(golden_angle_deg)
        twist = tt/genome_len * 4*np.pi
        z = (tt/genome_len)*cell_length

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
    if frame % 20 == 0 and cur_dim > len(rungs)-1:
        dim, note, col, name, alpha, verts = geometries[cur_dim]
        emerged.append(cur_dim)
        start = int((cur_dim/8.0)*N)
        step = max(1,N//(8*verts))
        idx = np.arange(start,start+verts*step,step)[:verts]
        idx = idx[idx<N]

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
                       size=8, parent=view.scene)
        rungs.append(mark)

        cen = all_pts.mean(axis=0)
        centers.append(cen)
        lbl = Text(f"{cur_base}:{name}", pos=cen+[0,0,0.3], color=col,font_size=10,bold=True, anchor_x='center', parent=view.scene)
        labels.append(lbl)

        if len(emerged)>1:
            prev = emerged[-2]
            prev_col = list(Color(geometries[prev][2]).rgba)
            prev_col[3]=0.25
            echo_pts = all_pts*0.75
            echo = Markers(pos=echo_pts, face_color=prev_col, size=5, parent=view.scene)
            echoes.append(echo)

        if len(centers)>1:
            prev_c = centers[-2]
            segs=[]
            for i in range(min(6,len(all_pts))):
                segs+=[prev_c,cen]
            link=Line(pos=np.array(segs), color=(0.7,0.7,0.7,0.4), width=1, connect='segments', parent=view.scene)
            links.append(link)

    # ---------- CAMERA ----------
    view.camera.azimuth = frame*0.3
    view.camera.elevation = 20+5*np.sin(frame*0.005)

    # ---------- % COMPLETE ----------
    percent_complete = min(100, frame/genome_len*100)
    progress_label.text = f"{percent_complete:.1f}%"

    canvas.update()

timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    app.run()
