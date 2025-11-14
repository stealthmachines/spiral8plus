"""
FASTA is Life — Full Holographic Genome-Driven Cell with Division
Everything in 3D (strands, organelles, decay, division) is derived from the genome.
Requirements: pip install vispy pyqt6 numpy
Run: python fasta_is_life.py
"""

import os
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ----------------- CONSTANTS -----------------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)

# Base → dimension mapping
base_map = {'A':0, 'T':1, 'G':2, 'C':3}

# Geometries: (dim, color, vertices)
geometries = [
    (0,'red',1), (1,'green',2), (2,'blue',3), (3,'violet',4),
    (4,'orange',5), (5,'indigo',6), (6,'purple',7), (7,'white',8)
]

angles = [i * golden_angle_deg for i in range(8)]
core_radius = 20.0
strand_sep = 1.0
twist_factor = 2*np.pi
max_points = 12000
decay_strength = 0.005
division_interval = 2000  # frames per division

# ----------------- LOAD GENOME -----------------
fasta_path = "ecoli_k12.fasta"
if not os.path.exists(fasta_path):
    raise FileNotFoundError(f"FASTA not found: {fasta_path}")

def load_genome(path):
    seq = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"): continue
            seq.extend(list(line.strip().upper()))
    return seq

genome_seq = load_genome(fasta_path)
genome_len = len(genome_seq)
print(f"Genome length: {genome_len}")

# ----------------- VISPY SETUP -----------------
canvas = scene.SceneCanvas(keys='interactive', size=(1400,900), bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

strand1_vis = Line(pos=np.zeros((1,3)), color=(1,1,1,0.7), width=2, parent=view.scene)
strand2_vis = Line(pos=np.zeros((1,3)), color=(1,1,1,0.7), width=2, parent=view.scene)

progress_text = Text("0%", pos=[0,0,30], color='white', font_size=28,
                     anchor_x='center', parent=view.scene)

# ----------------- STATE -----------------
cells = [{"frame":0, "accum_s1":[], "accum_s2":[], "organelles":[], "labels":[], "centers":[]}]

# ----------------- HELPERS -----------------
def fasta_noise(idx, dim):
    """Genome-driven deterministic noise"""
    window = genome_seq[idx:idx+3]
    vals = [ord(c)%10 for c in window]
    return np.array([vals[0], vals[1], vals[2]])*0.001*(dim+1)

def spawn_organelle(idx, center):
    val = ord(genome_seq[idx % genome_len]) % 8
    geom_dim, color, verts = geometries[val]
    pts = center + fasta_noise(idx,val)*10.0
    rgba = list(Color(color).rgba)
    rgba[3] = 1.0
    mark = Markers(pos=pts.reshape(1,3), face_color=rgba,
                   edge_color=None, size=6, parent=view.scene)
    return {"marker": mark, "positions": pts.reshape(1,3), "color": rgba, "age":0}

# ----------------- UPDATE LOOP -----------------
def update(ev):
    new_cells = []
    for cell in cells:
        cell["frame"] += 1
        frame = cell["frame"]
        idx = frame % genome_len
        base = genome_seq[idx]
        dim = base_map.get(base,0)

        theta = idx * np.radians(golden_angle_deg)
        twist = idx / genome_len * twist_factor
        z = np.sin(idx/genome_len * np.pi * 6) * 3 + (idx/genome_len)*12
        a1 = np.radians(angles[dim])
        a2 = np.radians(-angles[dim])
        r = core_radius * (1 - (idx/genome_len)**1.5)
        r = max(r,0.5)

        p1 = np.array([r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                       r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
                       z]) + fasta_noise(idx,dim)
        p2 = np.array([r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
                       r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
                       z]) + fasta_noise(idx,dim)

        cell["accum_s1"].append(p1)
        cell["accum_s2"].append(p2)
        if len(cell["accum_s1"]) > max_points:
            drop = len(cell["accum_s1"])//3
            cell["accum_s1"] = cell["accum_s1"][drop:]
            cell["accum_s2"] = cell["accum_s2"][drop:]

        strand1_vis.set_data(np.array(cell["accum_s1"]))
        strand2_vis.set_data(np.array(cell["accum_s2"]))

        # Spawn organelles
        if idx % 60 == 0:
            center = (p1+p2)/2
            org = spawn_organelle(idx, center)
            cell["organelles"].append(org)

        # Organelles update with decay
        lattice_nodes = np.array(cell["accum_s1"] + cell["accum_s2"])
        to_remove = []
        for org in cell["organelles"]:
            new_pts = []
            for i, p in enumerate(org['positions']):
                nearest = lattice_nodes[i % len(lattice_nodes)]
                dir_vec = nearest - p
                decay = (ord(base)%5)*0.001 + decay_strength
                org['age'] += decay
                new_p = p + dir_vec * 0.015
                new_p *= (1 - decay)
                new_pts.append(new_p)
            org['positions'] = np.array(new_pts)
            org['marker'].set_data(pos=org['positions'], face_color=org['color'], size=6)
            if org['age'] > 0.95:
                org['marker'].parent = None
                to_remove.append(org)
        for org in to_remove:
            cell["organelles"].remove(org)

        # Labels
        if frame % 300 == 0:
            cen = (p1+p2)/2
            geom_dim, color, verts = geometries[dim]
            lbl = Text(f"{base}:{geom_dim}", pos=cen+[0,0,0.5],
                       color=color, font_size=12, bold=True,
                       anchor_x='center', parent=view.scene)
            cell["labels"].append(lbl)
            cell["centers"].append(cen)

        # Division
        if frame % division_interval == 0:
            # duplicate cell state, slightly offset in space
            new_cell = {
                "frame":0,
                "accum_s1": [s + np.random.normal(0,0.5,3) for s in cell["accum_s1"]],
                "accum_s2": [s + np.random.normal(0,0.5,3) for s in cell["accum_s2"]],
                "organelles": [dict(marker=o["marker"], positions=o["positions"].copy(),
                                    color=o["color"], age=o["age"]) for o in cell["organelles"]],
                "labels": [],
                "centers": cell["centers"].copy()
            }
            new_cells.append(new_cell)

    cells.extend(new_cells)

    # Camera
    view.camera.azimuth = frame * 0.15
    view.camera.elevation = 20 + 10*np.sin(frame*0.003)

    # Progress
    percent = min(sum([c["frame"] for c in cells])/genome_len*100,100)
    progress_text.text = f"{percent:.2f}%"
    canvas.update()

# ----------------- START -----------------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    app.run()
