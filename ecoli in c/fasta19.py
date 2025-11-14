"""
FASTA Universe Ultra — GPU-accelerated holographic genome-driven cells
Everything is FASTA. Multi-cell spirals, organelles, division, lattice interactions.
Batch GPU updates for extreme speed.
Install: pip install vispy pyqt6 numpy
Run: python fasta_universe_ultra.py
"""

import os
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Markers, Text
from vispy.color import Color

# ---------- CONSTANTS ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)
core_radius = 15.0
strand_sep = 0.5
MAX_POINTS = 1_000_000
MAX_ORG_POINTS = 500_000
BATCH_STEPS = 1000
CELL_SPACING = 10.0
DIVISION_INTERVAL = 25_000
MAX_CELLS = 200

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
canvas = scene.SceneCanvas(keys='interactive', size=(1800, 1000), bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# ---------- CELL CLASS ----------
class Cell:
    def __init__(self, genome, origin=np.zeros(3)):
        self.genome = genome
        self.origin = np.array(origin, dtype=np.float32)
        self.frame = 0
        self.lattice_positions = np.zeros((MAX_POINTS,3),dtype=np.float32)
        self.organelles_positions = np.zeros((MAX_ORG_POINTS,3),dtype=np.float32)
        self.lattice_count = 0
        self.organelles_count = 0
        self.lattice_marker = Markers(pos=self.lattice_positions[:0], face_color=(1,1,1,0.7), size=1.5, parent=view.scene)
        self.organelles_marker = Markers(pos=self.organelles_positions[:0], face_color=(0,1,1,0.7), size=2.5, parent=view.scene)

    def genome_to_vec(self, idx):
        base = self.genome[idx]
        dim = base_map.get(base, 0)
        theta = idx * np.radians(golden_angle_deg)
        r = core_radius * (1 - (idx/len(self.genome))**1.4)
        z = np.sin(idx/len(self.genome) * np.pi*4)*2 + (idx/len(self.genome))*8
        a = np.radians(dim*90)
        x = r*np.cos(theta)*np.cos(a) - r*np.sin(theta)*np.sin(a)
        y = r*np.sin(theta)*np.cos(a) + r*np.cos(theta)*np.sin(a)
        return np.array([x,y,z], dtype=np.float32) + self.origin

    def step(self):
        for _ in range(BATCH_STEPS):
            idx = self.frame % len(self.genome)
            pos = self.genome_to_vec(idx)
            self.lattice_positions[self.lattice_count % MAX_POINTS] = pos
            self.lattice_count += 1

            # Organelles (stochastic, FASTA-driven)
            if np.random.rand() < 0.03:
                n = 5 + np.random.randint(5)
                organelle_pts = pos + np.random.normal(scale=0.2, size=(n,3))
                start = self.organelles_count
                end = min(start+n, MAX_ORG_POINTS)
                self.organelles_positions[start:end] = organelle_pts[:end-start]
                self.organelles_count += (end-start)

            self.frame += 1

        # GPU batch update
        self.lattice_marker.set_data(self.lattice_positions[:self.lattice_count])
        self.organelles_marker.set_data(self.organelles_positions[:self.organelles_count])

# ---------- MULTI-CELL SYSTEM ----------
cells = [Cell(genome_seq, origin=np.zeros(3))]
progress_text = Text("0%", pos=[0,0,core_radius*3], color='white', font_size=24,
                     anchor_x='center', parent=view.scene)

# ---------- UPDATE FUNCTION ----------
def update(ev):
    global cells
    # Step all cells
    for cell in cells:
        cell.step()

    # Cell division
    if len(cells) < MAX_CELLS:
        total_frames = sum(c.frame for c in cells)
        if total_frames > DIVISION_INTERVAL * len(cells):
            new_origin = np.random.uniform(-CELL_SPACING, CELL_SPACING, 3)
            cells.append(Cell(genome_seq, origin=new_origin))

    # Lattice backpressure / holographic repulsion
    for i, c1 in enumerate(cells):
        for j, c2 in enumerate(cells[i+1:], start=i+1):
            delta = c2.origin - c1.origin
            dist = np.linalg.norm(delta)
            if dist < CELL_SPACING:
                shift = 0.02*(CELL_SPACING - dist)
                if dist > 0:
                    c1.origin -= delta/dist*shift
                    c2.origin += delta/dist*shift

    # Camera rotation
    frame_total = sum(c.frame for c in cells)
    view.camera.azimuth = frame_total*0.12
    view.camera.elevation = 15 + 5*np.sin(frame_total*0.002)

    # Progress
    progress = min(frame_total/(len(genome_seq)*MAX_CELLS)*100, 100)
    progress_text.text = f"{progress:.1f}%"

    canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.005, connect=update, start=True)  # ultra-fast updates

if __name__ == '__main__':
    canvas.show()
    app.run()
