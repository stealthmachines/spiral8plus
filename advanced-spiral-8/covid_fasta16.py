"""
FASTA Universe: GPU-accelerated holographic DNA cell simulation with division
Everything is FASTA. Multi-cell spirals, organelles, lattice growth, backpressure — all genome-driven.
Install: pip install vispy pyqt6 numpy
Run: python fasta_multicell.py
"""

import os
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Markers, Text
from vispy.color import Color

# ---------- CONSTANTS ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)

bases = ['A','T','G','C']
base_map = {b:i for i,b in enumerate(bases)}  # A=0, T=1, G=2, C=3
geom_colors = ['red','green','blue','violet']

core_radius = 15.0
strand_sep = 0.5
MAX_POINTS = 200_000
MAX_ORG_POINTS = 100_000
BATCH_STEPS = 500

# ---------- LOAD GENOME ----------

# ---------- AUTO-DETECT AND LOAD GENOME ----------
def find_covid_fasta():
    """Automatically find the COVID-19 FASTA file in the ncbi_dataset"""
    import glob
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
    if not os.path.exists(fasta_file):
        raise FileNotFoundError(f"FASTA not found: {fasta_file}")
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"): continue
            seq.extend(list(line.strip().upper()))
    return seq

genome_seq = load_genome(find_covid_fasta())
genome_len = len(genome_seq)
print(f"Genome length: {genome_len}")

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='#000011')
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
        self.lattice_marker = Markers(pos=self.lattice_positions[:0], face_color=(1,1,1,0.7), size=2, parent=view.scene)
        self.organelles_marker = Markers(pos=self.organelles_positions[:0], face_color=(0,1,1,0.7), size=3, parent=view.scene)

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

            # Organelles (stochastic)
            if np.random.rand() < 0.02:
                n = 5 + np.random.randint(5)
                organelle_pts = pos + np.random.normal(scale=0.2, size=(n,3))
                self.organelles_positions[self.organelles_count:self.organelles_count+n] = organelle_pts
                self.organelles_count += n

            self.frame += 1

        # Update visuals
        self.lattice_marker.set_data(self.lattice_positions[:self.lattice_count])
        self.organelles_marker.set_data(self.organelles_positions[:self.organelles_count])

# ---------- MULTI-CELL SYSTEM ----------
cells = [Cell(genome_seq, origin=np.array([0,0,0]))]
division_interval = 20_000  # frames per division
max_cells = 4

progress_text = Text("0%", pos=[0,0,core_radius*2], color='white', font_size=24,
                     anchor_x='center', parent=view.scene)

# ---------- UPDATE FUNCTION ----------
def update(ev):
    # Step all cells
    for cell in cells:
        cell.step()

    # Division logic
    if len(cells) < max_cells:
        total_frames = sum(c.frame for c in cells)
        if total_frames > division_interval * len(cells):
            new_origin = np.random.uniform(-10,10,3)
            cells.append(Cell(genome_seq, origin=new_origin))

    # Camera rotation
    frame = sum(c.frame for c in cells)
    view.camera.azimuth = frame*0.12
    view.camera.elevation = 15 + 5*np.sin(frame*0.002)

    # Progress
    progress = min(frame/(len(genome_seq)*max_cells)*100,100)
    progress_text.text = f"{progress:.1f}%"

    canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SARS-CoV-2 Wuhan-Hu-1 Genome Visualization")
    print("="*60)
    if 'genome_len' in dir():
        print(f"Genome length: {genome_len:,} nucleotides")
    print("="*60)

    canvas.show()
    app.run()
