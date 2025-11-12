# dna_full_genome.py
# --------------------------------------------------------------
# Full-genome φ-spiral, GPU-accelerated, color-coded A/C/G/T.
# Install: pip install vispy pyqt6
# --------------------------------------------------------------

import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Markers
from vispy.color import Color

# ---------- PARAMETERS ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle = 2 * np.pi / (phi**2)   # radians
rise_per_base = 0.005                 # vertical separation per base
scale = 0.2                            # radial scale

# Base-to-color mapping
base_colors = {
    'A': (1, 0, 0, 1),    # Red
    'C': (0, 1, 0, 1),    # Green
    'G': (0, 0, 1, 1),    # Blue
    'T': (1, 1, 0, 1),
    'N': (0.5, 0.5, 0.5, 1),  # Gray (unknown)    # Yellow
}

# ---------- LOAD GENOME ----------
# genome_seq should be a string of 'A','C','G','T'
# Example: genome_seq = "ATGCGTAC..."
from Bio import SeqIO
record = SeqIO.read("ecoli_k12.fasta", "fasta")
genome_seq = str(record.seq)

N = len(genome_seq)

# ---------- COMPUTE SPIRAL POSITIONS ----------
theta = np.arange(N) * golden_angle
r = scale * np.sqrt(np.arange(N))         # optional: sqrt spacing
z = np.arange(N) * rise_per_base

x = r * np.cos(theta)
y = r * np.sin(theta)

positions = np.column_stack((x, y, z))

# ---------- MAP COLORS ----------
colors = np.array([base_colors[b] for b in genome_seq], dtype=np.float32)

# ---------- VISPY ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# markers for all bases
markers = Markers(pos=positions, face_color=colors, size=3, parent=view.scene)

# ---------- ROTATION ----------
frame = 0
def update(ev):
    global frame
    frame += 1
    view.camera.azimuth = frame * 0.05
    view.camera.elevation = 20
    canvas.update()

timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    app.run()
