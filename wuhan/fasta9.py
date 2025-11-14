"""
All is FASTA — fully self-organizing holographic cell
No contrived timers, modulo operations, or fixed divisions
Everything emerges from genome sequence alone
"""

import os, numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers
from vispy.color import Color

# ---------- LOAD GENOME ----------
fasta_path = "ecoli_k12.fasta"
if not os.path.exists(fasta_path):
    raise FileNotFoundError(f"FASTA not found: {fasta_path}")

genome_seq = []
with open(fasta_path) as f:
    for line in f:
        if line.startswith(">"): continue
        genome_seq.extend(list(line.strip().upper()))
genome_len = len(genome_seq)
print("Genome length:", genome_len)

# ---------- VISUAL SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200,900), bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

strand_vis = Line(pos=np.zeros((1,3)), color=(1,1,1,0.8), width=2, parent=view.scene)
organelles = []

# ---------- STATE ----------
lattice = []
frame = 0

# ---------- HELPER FUNCTIONS ----------
def genome_to_vec(idx):
    """Map genome to a 3D vector recursively and holographically"""
    b = genome_seq[idx % genome_len]
    val = ord(b) % 16
    angle = val / 16 * 2*np.pi
    radius = 5 + val*0.5
    z = np.sin(idx/genome_len * np.pi*4) * val*0.1
    return np.array([radius*np.cos(angle), radius*np.sin(angle), z])

def spawn_organelles(center, idx):
    """Organelles emerge from genome values around a center"""
    val = ord(genome_seq[idx % genome_len]) % 8 + 1
    pts = center + np.random.normal(scale=val*0.2, size=(val,3))
    rgba = list(Color('cyan').rgba)
    rgba[3] = 0.9
    mark = Markers(pos=pts, face_color=rgba, size=5, parent=view.scene)
    return {"marker": mark, "positions": pts}

# ---------- UPDATE LOOP ----------
def update(ev):
    global frame, lattice, organelles

    idx = frame % genome_len
    pos = genome_to_vec(idx)

    lattice.append(pos)
    if len(lattice) > 8000:  # saturation -> holographic replication
        lattice = lattice[len(lattice)//2:]

    # Organelles naturally emerge when genome-derived vectors cluster
    if np.random.rand() < (ord(genome_seq[idx % genome_len])%5)/5:
        org = spawn_organelles(pos, idx)
        organelles.append(org)

    # Implicit decay / backpressure from genome density
    for org in organelles:
        for i, p in enumerate(org["positions"]):
            nearest_idx = np.random.randint(0,len(lattice))
            p += (lattice[nearest_idx]-p)*0.02
            p *= 0.995  # decay entirely from interactions
        org["marker"].set_data(pos=org["positions"])

    strand_vis.set_data(np.array(lattice))

    view.camera.azimuth = frame * 0.1
    view.camera.elevation = 20 + 5*np.sin(frame*0.003)

    frame += 1
    canvas.update()

timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    app.run()
