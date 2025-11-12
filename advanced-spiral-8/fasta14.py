"""
FASTA Universe: GPU-accelerated holographic DNA cell simulation
Everything is FASTA. All positions, spirals, organelles arise from the genome.
Install: pip install vispy pyqt6 numpy
Run: python fasta_universe.py
"""

import os
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- CONSTANTS ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)

# Genome mapping
bases = ['A','T','G','C']
base_map = {b:i for i,b in enumerate(bases)}  # A=0, T=1, G=2, C=3

# Geometry properties per base (color + structure)
geom_colors = ['red','green','blue','violet']
core_radius = 15.0
strand_sep = 0.5

MAX_POINTS = 200_000  # preallocate GPU buffers
BATCH_STEPS = 1000    # fast-forward genome per frame

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
print(f"Genome length: {genome_len}, Total bases: {len(genome_seq)}")

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# ---------- GPU BATCHED MARKERS ----------
lattice_positions = np.zeros((MAX_POINTS, 3), dtype=np.float32)
organelles_positions = np.zeros((MAX_POINTS, 3), dtype=np.float32)
lattice_count = 0
organelles_count = 0

lattice_marker = Markers(pos=lattice_positions[:lattice_count],
                         face_color=(1,1,1,0.7), size=2, parent=view.scene)
organelles_marker = Markers(pos=organelles_positions[:organelles_count],
                            face_color=(0,1,1,0.7), size=3, parent=view.scene)

# Text for progress
progress_text = Text("0%", pos=[0,0,core_radius*1.5], color='white',
                     font_size=24, anchor_x='center', parent=view.scene)

# ---------- HELPER FUNCTIONS ----------
def genome_to_vec(idx):
    """Map genome base to 3D φ-spiral position."""
    base = genome_seq[idx]
    dim = base_map.get(base, 0)
    theta = idx * np.radians(golden_angle_deg)
    r = core_radius * (1 - (idx/genome_len)**1.4)
    z = np.sin(idx/genome_len * np.pi*4)*2 + (idx/genome_len)*8
    a = np.radians(dim * 90)
    x = r*np.cos(theta)*np.cos(a) - r*np.sin(theta)*np.sin(a)
    y = r*np.sin(theta)*np.cos(a) + r*np.cos(theta)*np.sin(a)
    return np.array([x,y,z], dtype=np.float32)

# ---------- CELL STATE ----------
cell_state = {"frame":0}

# ---------- UPDATE FUNCTION ----------
def update(ev):
    global lattice_count, organelles_count

    # --- Batch simulation of genome steps ---
    for _ in range(BATCH_STEPS):
        idx = cell_state["frame"] % genome_len
        pos = genome_to_vec(idx)
        lattice_positions[lattice_count % MAX_POINTS] = pos
        lattice_count += 1

        # Spawn organelles stochastically
        if np.random.rand() < 0.02:
            n = 5 + np.random.randint(5)
            organelle_pts = pos + np.random.normal(scale=0.2, size=(n,3))
            organelles_positions[organelles_count:organelles_count+n] = organelle_pts
            organelles_count += n

        cell_state["frame"] += 1

    # --- Update GPU visuals ---
    lattice_marker.set_data(lattice_positions[:lattice_count])
    organelles_marker.set_data(organelles_positions[:organelles_count])

    # --- Camera rotation ---
    view.camera.azimuth = cell_state["frame"] * 0.15
    view.camera.elevation = 15 + 5*np.sin(cell_state["frame"]*0.002)

    # --- Progress display ---
    progress = min(cell_state["frame"]/genome_len*100, 100)
    progress_text.text = f"{progress:.1f}%"

    canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    app.run()
