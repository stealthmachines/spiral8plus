# dna_full_genome_batched.py
# --------------------------------------------------------------
# Full-genome φ-spiral, every base pair rung rendered.
# GPU-accelerated, batched lines for efficiency.
# --------------------------------------------------------------

import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Markers, Line
from vispy.color import Color
from Bio import SeqIO


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


# ---------- PARAMETERS ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle = 2 * np.pi / (phi**2)
rise_per_base = 0.005
scale = 0.2
helix_radius = 0.05
echo_scale = 0.7

# Base-to-color mapping
base_colors = {'A': (1,0,0,1), 'C': (0,1,0,1), 'G': (0,0,1,1), 'T': (1,1,0,1)}

# ---------- LOAD GENOME ----------
record = SeqIO.read(find_covid_fasta(), "fasta")
genome_seq = str(record.seq)
N = len(genome_seq)

# ---------- SPIRAL POSITIONS ----------
theta = np.arange(N) * golden_angle
r = scale * np.sqrt(np.arange(N))
z = np.arange(N) * rise_per_base

x1 = r * np.cos(theta) + helix_radius * np.cos(theta*2)
y1 = r * np.sin(theta) + helix_radius * np.sin(theta*2)

x2 = r * np.cos(theta) - helix_radius * np.cos(theta*2)
y2 = r * np.sin(theta) - helix_radius * np.sin(theta*2)

pos1 = np.column_stack((x1, y1, z))
pos2 = np.column_stack((x2, y2, z))

colors1 = np.array([base_colors[b] for b in genome_seq], dtype=np.float32)
colors2 = colors1.copy()

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='#000011')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# MARKERS
strand1 = Markers(pos=pos1, face_color=colors1, size=2, parent=view.scene)
strand2 = Markers(pos=pos2, face_color=colors2, size=2, parent=view.scene)

# RUNG CONNECTIONS - BATCHED
batch_size = 10000  # lines per batch for GPU efficiency
lines = []

for i in range(0, N, batch_size):
    end = min(i+batch_size, N)
    pts = np.empty(((end-i)*2,3), dtype=np.float32)
    pts[0::2] = pos1[i:end]
    pts[1::2] = pos2[i:end]
    line = Line(pos=pts, color=(0.7,0.7,0.7,0.5), width=1, connect='segments', parent=view.scene)
    lines.append(line)

# ECHO COPIES
echo1 = Markers(pos=pos1*echo_scale, face_color=np.array(colors1)*0.25, size=1.5, parent=view.scene)
echo2 = Markers(pos=pos2*echo_scale, face_color=np.array(colors2)*0.25, size=1.5, parent=view.scene)

# ---------- ROTATION ----------
frame = 0
def update(ev):
    global frame
    frame += 1
    view.camera.azimuth = frame * 0.03
    view.camera.elevation = 20
    canvas.update()

timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    app.run()
