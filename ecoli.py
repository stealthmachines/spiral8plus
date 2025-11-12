# dna_full_genome_double.py
# --------------------------------------------------------------
# Full-genome φ-spiral with double strands, rungs, and echoes.
# GPU-accelerated; A/C/G/T color-coded.
# --------------------------------------------------------------

import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Markers, Line
from vispy.color import Color
from Bio import SeqIO

# ---------- PARAMETERS ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle = 2 * np.pi / (phi**2)    # radians
rise_per_base = 0.005
scale = 0.2
helix_radius = 0.05                     # strand offset from center
echo_scale = 0.7                         # shrink echo copies

# Base-to-color mapping
base_colors = {
    'A': (1, 0, 0, 1),   # Red
    'C': (0, 1, 0, 1),   # Green
    'G': (0, 0, 1, 1),   # Blue
    'T': (1, 1, 0, 1),
    'N': (0.5, 0.5, 0.5, 1),  # Gray (unknown)   # Yellow
}

# ---------- LOAD GENOME ----------
record = SeqIO.read("ecoli_k12.fasta", "fasta")
genome_seq = str(record.seq)
N = len(genome_seq)

# ---------- COMPUTE SPIRAL POSITIONS ----------
theta = np.arange(N) * golden_angle
r = scale * np.sqrt(np.arange(N))
z = np.arange(N) * rise_per_base

# two strands (offset + helix twist)
x1 = r * np.cos(theta) + helix_radius * np.cos(theta*2)
y1 = r * np.sin(theta) + helix_radius * np.sin(theta*2)

x2 = r * np.cos(theta) - helix_radius * np.cos(theta*2)
y2 = r * np.sin(theta) - helix_radius * np.sin(theta*2)

pos1 = np.column_stack((x1, y1, z))
pos2 = np.column_stack((x2, y2, z))

# ---------- COLORS ----------
colors1 = np.array([base_colors[b] for b in genome_seq], dtype=np.float32)
colors2 = colors1.copy()  # complementary strand same colors for visualization

# ---------- VISPY ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# markers
strand1 = Markers(pos=pos1, face_color=colors1, size=3, parent=view.scene)
strand2 = Markers(pos=pos2, face_color=colors2, size=3, parent=view.scene)

# rung connections: sample every M bases to reduce rendering load
M = 50
lines = []
for i in range(0, N, M):
    pts = np.array([pos1[i], pos2[i]])
    line = Line(pos=pts, color=(0.7,0.7,0.7,0.5), width=1, connect='segments', parent=view.scene)
    lines.append(line)

# faint echo copies
echo1 = Markers(pos=pos1*echo_scale, face_color=np.array(colors1)*0.25, size=2, parent=view.scene)
echo2 = Markers(pos=pos2*echo_scale, face_color=np.array(colors2)*0.25, size=2, parent=view.scene)

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
