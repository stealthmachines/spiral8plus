"""
FASTA Holographic Single Cell with Infinite Division
GPU-accelerated φ-spiral chromosomes, stochastic organelles, lattice & rungs
Cell divides endlessly, fully FASTA-driven, volumetric, holographic
Install: pip install vispy pyqt6 numpy
Run: python fasta_infinite_division.py
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
view.camera = scene.cameras.TurntableCamera(distance=25.0, azimuth=0, elevation=20.0)

# ---------- SINGLE CELL CLASS ----------
class SingleCell:
    def __init__(self, genome, origin=np.zeros(3), scale=1.0):
        self.genome = genome
        self.frame = 0
        self.origin = origin
        self.scale = scale
        self.lattice_positions = np.zeros((MAX_POINTS,3), dtype=np.float32)
        self.organelles_positions = np.zeros((MAX_ORGANELLES,3), dtype=np.float32)
        self.lattice_count = 0
        self.organelles_count = 0
        self.strand1 = Line(pos=np.zeros((1,3)), color=(1,0.5,0,0.7), width=2, parent=view.scene)
        self.strand2 = Line(pos=np.zeros((1,3)), color=(0,1,0.5,0.7), width=2, parent=view.scene)
        self.rungs = []

    def genome_to_vec(self, idx):
        base = self.genome[idx]
        dim = base_map.get(base, 0)
        theta = idx * np.radians(golden_angle_deg)
        r = core_radius * self.scale * (1 - (idx/len(self.genome))**1.4)
        z = np.sin(idx/len(self.genome) * np.pi*4)*2*self.scale + (idx/len(self.genome))*8*self.scale
        a = np.radians(dim*90)
        x = r*np.cos(theta)*np.cos(a) - r*np.sin(theta)*np.sin(a)
        y = r*np.sin(theta)*np.cos(a) + r*np.cos(theta)*np.sin(a)
        return np.array([x,y,z], dtype=np.float32) + self.origin

    def step(self):
        s1, s2 = [], []

        for _ in range(BATCH_STEPS):
            idx = self.frame % len(self.genome)
            pos = self.genome_to_vec(idx)
            self.lattice_positions[self.lattice_count % MAX_POINTS] = pos
            self.lattice_count += 1

            # Strand offsets
            s1.append(pos)
            s2.append(pos + np.array([strand_sep, strand_sep, 0])*self.scale)

            # Organelles (stochastic, genome-driven)
            if np.random.rand() < 0.03:
                n = 5 + np.random.randint(6)
                organelle_pts = pos + np.random.normal(scale=0.1*self.scale, size=(n,3))
                start = self.organelles_count
                end = min(start+n, MAX_ORGANELLES)
                self.organelles_positions[start:end] = organelle_pts[:end-start]
                self.organelles_count += (end-start)

            self.frame += 1

        # Update visuals
        self.strand1.set_data(np.array(s1))
        self.strand2.set_data(np.array(s2))
        self.lattice_marker.set_data(self.lattice_positions[:self.lattice_count])
        self.organelles_marker.set_data(self.organelles_positions[:self.organelles_count])

        # Rungs
        for i in range(0, len(s1), 10):
            segs = np.array([s1[i], s2[i]])
            line = Line(pos=segs, color=(1,1,0,0.5), width=1.2, parent=view.scene)
            self.rungs.append(line)
            if len(self.rungs) > 500:
                self.rungs[0].parent = None
                self.rungs.pop(0)

    def init_markers(self):
        self.lattice_marker = Markers(pos=self.lattice_positions[:0], face_color=(1,1,1,0.8), size=1.5, parent=view.scene)
        self.organelles_marker = Markers(pos=self.organelles_positions[:0], face_color=(0,1,1,0.7), size=2.5, parent=view.scene)

# ---------- MULTIPLE CELLS WITH DIVISION ----------
cells = []
initial_cell = SingleCell(genome_seq)
initial_cell.init_markers()
cells.append(initial_cell)

division_interval = 2000  # frames before division
cell_scale_decay = 0.5    # scale for daughter cells

progress_text = Text("0%", pos=[0,0,core_radius*3], color='white', font_size=24,
                     anchor_x='center', parent=view.scene)

def update(ev):
    global cells
    new_cells = []
    for cell in cells:
        cell.step()
        # Division
        if cell.frame % division_interval == 0 and cell.frame > 0:
            # Two daughter cells
            for offset in [np.array([5,5,0]), np.array([-5,-5,0])]:
                daughter = SingleCell(cell.genome, origin=cell.origin + offset, scale=cell.scale*cell_scale_decay)
                daughter.init_markers()
                new_cells.append(daughter)
    cells.extend(new_cells)

    # Update progress
    total_frames = sum([c.frame for c in cells])
    percent = min(total_frames / (len(genome_seq)*10) * 100, 100)
    progress_text.text = f"{percent:.1f}%"

    canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.005, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    app.run()
