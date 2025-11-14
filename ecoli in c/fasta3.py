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
def load_genome(fasta_file):
    seq = []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"): continue
            seq.extend(list(line.strip().upper()))
    return seq

fasta_path = "ecoli_k12.fasta"
if not os.path.exists(fasta_path):
    raise FileNotFoundError(f"FASTA not found: {fasta_path}")
genome_seq = load_genome(fasta_path)
genome_len = len(genome_seq)
print(f"Genome length: {genome_len}")

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200,800), bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

strand1_vis = Line(pos=np.zeros((1,3)), color=(0,1,1,0.9), width=2,parent=view.scene)
strand2_vis = Line(pos=np.zeros((1,3)), color=(1,0.5,0,0.9), width=2,parent=view.scene)
donut_vis   = Line(pos=np.zeros((1,3)), color=(1,1,0,0.5), width=1.5,parent=view.scene)

# ---------- CELL DATA ----------
cells = [{"accum_s1": [], "accum_s2": [], "accum_donut": [],
          "organelles": [], "genome_start": 0}]
frame = 0

# ---------- HELPERS ----------
def consensus_dim(b):
    return base_map.get(b,1)-1

def organelle_params(b):
    d = consensus_dim(b)
    return 0.02 + d/16*0.12, 6 + (d%5)*3, 0.2 + d/16*0.9, 0.08 + d/16*0.4

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
def update(ev):
    global frame, cells
    frame+=1
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
