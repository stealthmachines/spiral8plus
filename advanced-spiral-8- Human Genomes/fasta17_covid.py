"""
FASTA Universe 2.0 — SARS-CoV-2 Wuhan-Hu-1 genome-driven holographic cells
Everything is FASTA. Multi-cell spirals, organelles, division, lattice interactions.
COVID-19 genome visualization with thousands of viral genome copies.
Install: pip install vispy pyqt6 numpy
Run: python fasta17_covid.py
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
base_map = {b:i for i,b in enumerate(bases)}
geom_colors = ['red','green','blue','violet']

core_radius = 15.0
strand_sep = 0.5
MAX_POINTS = 100_000
MAX_ORG_POINTS = 50_000
BATCH_STEPS = 500

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

# Load the SARS-CoV-2 Wuhan-Hu-1 genome
covid_fasta_path = r"ncbi_dataset\data\GCF_009858895.2\GCF_009858895.2_ASM985889v3_genomic.fna"
if not os.path.exists(covid_fasta_path):
    # Try alternative path
    covid_fasta_path = r"ncbi_dataset\data\GCA_009858895.3\GCA_009858895.3_ASM985889v3_genomic.fna"

genome_seq = load_genome(covid_fasta_path)
genome_len = len(genome_seq)
print(f"SARS-CoV-2 Wuhan-Hu-1 genome length: {genome_len} bases")

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1400, 900), bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# ---------- CELL CLASS ----------
class CovidCell:
    def __init__(self, genome, origin=np.zeros(3), cell_id=0):
        self.genome = genome
        self.origin = np.array(origin, dtype=np.float32)
        self.cell_id = cell_id
        self.frame = 0
        self.lattice_positions = np.zeros((MAX_POINTS,3),dtype=np.float32)
        self.organelles_positions = np.zeros((MAX_ORG_POINTS,3),dtype=np.float32)
        self.lattice_count = 0
        self.organelles_count = 0

        # Color coding based on cell generation/position
        base_color = np.array([1.0, 0.3 + 0.7*(cell_id % 3)/3, 0.2 + 0.8*(cell_id % 5)/5, 0.8])
        organelle_color = np.array([0.2 + 0.8*(cell_id % 4)/4, 1.0, 0.3 + 0.7*(cell_id % 6)/6, 0.7])

        self.lattice_marker = Markers(pos=self.lattice_positions[:0], face_color=base_color, size=2, parent=view.scene)
        self.organelles_marker = Markers(pos=self.organelles_positions[:0], face_color=organelle_color, size=3, parent=view.scene)

    def genome_to_vec(self, idx):
        base = self.genome[idx]
        dim = base_map.get(base, 0)

        # Viral genome spiral with RNA-like characteristics
        theta = idx * np.radians(golden_angle_deg)

        # Shorter genome creates tighter spiral
        r = core_radius * (1 - (idx/len(self.genome))**1.2)

        # More compressed Z-axis for viral genome
        z = np.sin(idx/len(self.genome) * np.pi*6)*1.5 + (idx/len(self.genome))*6

        # Base-specific rotation
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

            # Viral organelles/proteins (more frequent than cellular organelles)
            if np.random.rand() < 0.05:  # Higher frequency for viral proteins
                n = 3 + np.random.randint(4)  # Smaller clusters for viral proteins
                organelle_pts = pos + np.random.normal(scale=0.15, size=(n,3))
                end_idx = min(self.organelles_count + n, MAX_ORG_POINTS)
                actual_n = end_idx - self.organelles_count
                if actual_n > 0:
                    self.organelles_positions[self.organelles_count:end_idx] = organelle_pts[:actual_n]
                    self.organelles_count = end_idx

            self.frame += 1

        # Update visuals
        actual_lattice_count = min(self.lattice_count, MAX_POINTS)
        actual_organelle_count = min(self.organelles_count, MAX_ORG_POINTS)

        self.lattice_marker.set_data(self.lattice_positions[:actual_lattice_count])
        self.organelles_marker.set_data(self.organelles_positions[:actual_organelle_count])

# ---------- MULTI-VIRAL SYSTEM ----------
cells = [CovidCell(genome_seq, origin=np.zeros(3), cell_id=0)]
division_interval = 15_000  # Faster replication for viruses
max_cells = 500  # Viral particles
cell_spacing = 8.0

progress_text = Text("SARS-CoV-2 Wuhan-Hu-1: 0%", pos=[0,0,core_radius*3], color='yellow', font_size=20,
                     anchor_x='center', parent=view.scene)

info_text = Text(f"Genome: {genome_len} nucleotides", pos=[0,-20,core_radius*3], color='cyan', font_size=14,
                 anchor_x='center', parent=view.scene)

# ---------- UPDATE FUNCTION ----------
def update(ev):
    global cells
    # Step all viral particles
    for cell in cells:
        cell.step()

    # Viral replication / new particle creation
    if len(cells) < max_cells:
        total_frames = sum(c.frame for c in cells)
        if total_frames > division_interval * len(cells):
            # New viral particles spread in 3D space
            new_origin = np.random.uniform(-cell_spacing*2, cell_spacing*2, 3)
            cells.append(CovidCell(genome_seq, origin=new_origin, cell_id=len(cells)))

    # Viral particle interactions: slight attraction/clustering behavior
    for i, c1 in enumerate(cells):
        for j, c2 in enumerate(cells[i+1:], start=i+1):
            delta = c2.origin - c1.origin
            dist = np.linalg.norm(delta)
            if dist < cell_spacing and dist > 0:
                # Slight attraction for clustering
                shift = 0.01*(cell_spacing - dist)
                c1.origin += delta/dist*shift*0.5
                c2.origin -= delta/dist*shift*0.5

    # Camera movement
    frame_total = sum(c.frame for c in cells)
    view.camera.azimuth = frame_total*0.08
    view.camera.elevation = 20 + 8*np.sin(frame_total*0.001)

    # Progress tracking
    progress = min(frame_total/(len(genome_seq)*max_cells)*100, 100)
    progress_text.text = f"SARS-CoV-2 Wuhan-Hu-1: {progress:.1f}% | Particles: {len(cells)}"

    canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    print("Starting SARS-CoV-2 Wuhan-Hu-1 genome visualization...")
    print(f"Genome loaded: {genome_len} nucleotides")
    print("Press ESC to exit")
    canvas.show()
    app.run()