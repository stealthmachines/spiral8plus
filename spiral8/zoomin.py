"""
FASTA Single Cell Zoom — GPU-accelerated φ-spiral chromosome
Everything is FASTA, fully volumetric, high-detail view of one cell.
Install: pip install vispy pyqt6 numpy
Run: python fasta_single_cell.py
"""

import os
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Markers, Line, Text
from vispy.color import Color

# ---------- CONSTANTS ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)
core_radius = 15.0
strand_sep = 0.3
MAX_POINTS = 500_000
MAX_ORGANELLES = 200_000
BATCH_STEPS = 500

bases = ['A','T','G','C']
base_map = {b:i for i,b in enumerate(bases)}
geom_colors = ['red','green','blue','violet']

# ---------- LOAD GENOME ----------
def load_genome(fasta_file):
    seq = []
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"FASTA not found: {fasta_file}")
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"): continue
            seq.extend(list(line.strip().upper()))
    return seq

genome_seq = load_genome("ecoli_k12.fasta")
genome_len = len(genome_seq)
print(f"Genome length: {genome_len}")

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1600,1200), bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(distance=30.0)

# ---------- SINGLE CELL CLASS ----------
class SingleCell:
    def __init__(self, genome):
        self.genome = genome
        self.frame = 0
        self.lattice_positions = np.zeros((MAX_POINTS,3), dtype=np.float32)
        self.organelles_positions = np.zeros((MAX_ORGANELLES,3), dtype=np.float32)
        self.lattice_count = 0
        self.organelles_count = 0

        self.lattice_marker = Markers(pos=self.lattice_positions[:0], face_color=(1,1,1,0.8), size=1.5, parent=view.scene)
        self.organelles_marker = Markers(pos=self.organelles_positions[:0], face_color=(0,1,1,0.7), size=2.5, parent=view.scene)
        self.strand1 = Line(pos=np.zeros((1,3)), color=(1,0.5,0,0.7), width=2, parent=view.scene)
        self.strand2 = Line(pos=np.zeros((1,3)), color=(0,1,0.5,0.7), width=2, parent=view.scene)
        self.rungs = []

    def genome_to_vec(self, idx):
        base = self.genome[idx]
        dim = base_map.get(base, 0)
        theta = idx * np.radians(golden_angle_deg)
        r = core_radius * (1 - (idx/len(self.genome))**1.4)
        z = np.sin(idx/len(self.genome) * np.pi*4)*2 + (idx/len(self.genome))*8
        a = np.radians(dim*90)
        x = r*np.cos(theta)*np.cos(a) - r*np.sin(theta)*np.sin(a)
        y = r*np.sin(theta)*np.cos(a) + r*np.cos(theta)*np.sin(a)
        return np.array([x,y,z], dtype=np.float32)

    def step(self):
        s1, s2 = [], []

        for _ in range(BATCH_STEPS):
            idx = self.frame % len(self.genome)
            pos = self.genome_to_vec(idx)
            self.lattice_positions[self.lattice_count % MAX_POINTS] = pos
            self.lattice_count += 1

            # Strand offsets
            s1.append(pos)
            s2.append(pos + np.array([strand_sep, strand_sep, 0]))

            # Organelles (stochastic)
            if np.random.rand() < 0.03:
                n = 5 + np.random.randint(6)
                organelle_pts = pos + np.random.normal(scale=0.1, size=(n,3))
                start = self.organelles_count
                end = min(start+n, MAX_ORGANELLES)
                self.organelles_positions[start:end] = organelle_pts[:end-start]
                self.organelles_count += (end-start)

            self.frame += 1

        # Update GPU visuals
        self.lattice_marker.set_data(self.lattice_positions[:self.lattice_count])
        self.organelles_marker.set_data(self.organelles_positions[:self.organelles_count])
        self.strand1.set_data(np.array(s1))
        self.strand2.set_data(np.array(s2))

        # Optional rungs between strands
        for i in range(0, len(s1), 10):
            segs = np.array([s1[i], s2[i]])
            line = Line(pos=segs, color=(1,1,0,0.5), width=1.2, parent=view.scene)
            self.rungs.append(line)
            if len(self.rungs) > 500:
                self.rungs[0].parent = None
                self.rungs.pop(0)

# ---------- INITIALIZE CELL ----------
cell = SingleCell(genome_seq)
progress_text = Text("0%", pos=[0,0,core_radius*3], color='white', font_size=24,
                     anchor_x='center', parent=view.scene)

# ---------- UPDATE FUNCTION ----------
def update(ev):
    cell.step()
    percent = min(cell.frame/len(genome_seq)*100, 100)
    progress_text.text = f"{percent:.1f}%"

    # Camera zoom slowly in/out
    view.camera.distance = 25 + 5*np.sin(cell.frame*0.002)
    view.camera.azimuth = cell.frame*0.08
    view.camera.elevation = 15 + 5*np.sin(cell.frame*0.001)

    canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.005, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    app.run()
