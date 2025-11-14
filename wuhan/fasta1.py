# dna_fasta_holo_cell.py
# --------------------------------------------------------------
# Full volumetric holographic E. coli-like cell
# EVERYTHING derived from FASTA
# Install: pip install vispy pyqt6 numpy
# Run: python dna_fasta_holo_cell.py
# --------------------------------------------------------------

import os
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers
from vispy.color import Color

# ---------- LOAD FASTA ----------
def load_genome(fasta_file):
    seq = []
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"FASTA not found: "+fasta_file)
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"): continue
            seq.extend(list(line.strip().upper()))
    return seq

genome_seq = load_genome("ecoli_k12.fasta")
genome_len = len(genome_seq)
print(f"Genome length: {genome_len}")

# ---------- VISUAL SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200,800), bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Strands & lattices
strand1_vis = Line(pos=np.zeros((1,3)), color=(0,1,1,0.9), width=2, parent=view.scene)
strand2_vis = Line(pos=np.zeros((1,3)), color=(1,0.5,0,0.9), width=2, parent=view.scene)
donut_vis   = Line(pos=np.zeros((1,3)), color=(1,1,0,0.5), width=1.5, parent=view.scene)

accum_s1, accum_s2, accum_donut = [], [], []
organelles = []
frame = 0
division_count = 0

# ---------- FASTA-DRIVEN PARAMETERS ----------
phi = (1 + np.sqrt(5))/2
golden_angle_deg = 360/phi**2

def base_value(b):
    """Numeric mapping fully derived from FASTA."""
    return (ord(b) % 10) + 1

def nucleotide_color(b):
    """Color derived from ASCII code of base."""
    v = (ord(b) % 255)/255.0
    return (v, 1-v, 0.5+0.5*v)

# ---------- HOLOGRAPHIC HELPERS ----------
def lattice_push(pt, lattice_nodes, strength):
    if len(lattice_nodes) == 0: return pt
    nearest = lattice_nodes[np.random.randint(0,len(lattice_nodes))]
    dir_vec = nearest - pt
    return pt + dir_vec * strength

def spawn_organelle(center, base):
    val = base_value(base)
    n = val + 4
    scale = val*0.05 + 0.05
    pts = center + np.random.normal(scale=scale, size=(n,3))
    col = nucleotide_color(base)
    rgba = list(col)+[1.0]
    mark = Markers(pos=pts, face_color=rgba, size=5, parent=view.scene)
    return {"marker": mark, "positions": pts, "color": rgba}

def divide_cell():
    global accum_s1, accum_s2, accum_donut, organelles
    half = len(accum_s1)//2
    accum_s1 = accum_s1[half:]
    accum_s2 = accum_s2[half:]
    accum_donut = accum_donut[half:]
    for org in organelles:
        org['positions'] += np.random.normal(scale=0.3, size=org['positions'].shape)
    print("Cell divided! Total divisions:", division_count)

# ---------- UPDATE LOOP ----------
def update(ev):
    global frame, accum_s1, accum_s2, accum_donut, organelles, division_count

    frame += 1
    idx = frame % genome_len
    base = genome_seq[idx]
    val = base_value(base)

    # ---------- Holographic positions ----------
    r = 10 + val*0.5
    theta = np.radians(frame*golden_angle_deg)
    z = np.sin(frame/genome_len*np.pi*4)*2 + (frame/genome_len)*6

    p1 = np.array([r*np.cos(theta), r*np.sin(theta), z])
    p2 = np.array([r*np.cos(theta)+0.5, r*np.sin(theta)-0.5, z])
    p1 += np.random.normal(scale=0.01*val, size=3)
    p2 += np.random.normal(scale=0.01*val, size=3)

    accum_s1.append(p1)
    accum_s2.append(p2)
    center = (p1+p2)/2
    accum_donut.append(center)

    if len(accum_s1) > 6000:
        drop = len(accum_s1)//3
        accum_s1 = accum_s1[drop:]
        accum_s2 = accum_s2[drop:]
        accum_donut = accum_donut[drop:]

    # ---------- Organelles ----------
    if frame % (20 + val) == 0:
        org = spawn_organelle(center, base)
        organelles.append(org)

    lattice_nodes = np.array(accum_s1 + accum_s2)
    for org in organelles:
        new_pts = np.array([lattice_push(p, lattice_nodes, 0.02) for p in org['positions']])
        org['positions'] = new_pts
        org['marker'].set_data(pos=new_pts, face_color=org['color'], size=5)

    # ---------- Update visuals ----------
    strand1_vis.set_data(np.array(accum_s1))
    strand2_vis.set_data(np.array(accum_s2))
    donut_vis.set_data(np.array(accum_donut))

    # ---------- Camera ----------
    view.camera.azimuth = frame*0.15
    view.camera.elevation = 15 + 8*np.sin(frame*0.002)

    # ---------- Cell division ----------
    if frame % 2000 == 0:
        division_count += 1
        divide_cell()

    canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == "__main__":
    canvas.show()
    app.run()
