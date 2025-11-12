"""
All is FASTA Universe — holographic, genome-driven, recursive, multi-cell simulation
Everything emerges from the genome; nothing is contrived.
Install: pip install vispy pyqt6 numpy
Run: python fasta_universe.py
"""

import os
import numpy as np
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
canvas = scene.SceneCanvas(keys='interactive', size=(1600,1000), bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# ---------- STATE ----------
phi = (1 + np.sqrt(5))/2
max_points_per_cell = 8000
cells = [{"lattice": [], "organelles": [], "frame": 0, "id":0}]
next_cell_id = 1

# ---------- HELPER FUNCTIONS ----------
def genome_to_vec(idx):
    """Convert genome index to a 3D vector holographically from the sequence"""
    b = genome_seq[idx % genome_len]
    val = ord(b) % 16 + 1
    angle = val/16*2*np.pi
    radius = 5 + val*0.5
    z = np.sin(idx/genome_len * np.pi*4) * val*0.15
    return np.array([radius*np.cos(angle), radius*np.sin(angle), z])

def spawn_organelles(center, idx):
    """Organelles emerge from genome-driven randomness"""
    val = (ord(genome_seq[idx % genome_len]) % 8) + 1
    pts = center + np.random.normal(scale=val*0.2, size=(val,3))
    rgba = list(Color('cyan').rgba)
    rgba[3] = 0.9
    mark = Markers(pos=pts, face_color=rgba, size=5, parent=view.scene)
    return {"marker": mark, "positions": pts}

def check_division(cell):
    """Implicit genome-driven cell division"""
    global next_cell_id
    if len(cell["lattice"]) > max_points_per_cell:
        half = len(cell["lattice"])//2
        new_cell = {
            "lattice": cell["lattice"][half:],
            "organelles": [],
            "frame": cell["frame"],
            "id": next_cell_id
        }
        next_cell_id += 1
        cell["lattice"] = cell["lattice"][:half]
        return new_cell
    return None

# ---------- UPDATE LOOP ----------
def update(ev):
    global cells

    new_cells = []

    for cell in cells:
        idx = cell["frame"] % genome_len
        pos = genome_to_vec(idx)
        cell["lattice"].append(pos)

        # Keep lattice size bounded
        if len(cell["lattice"]) > max_points_per_cell:
            cell["lattice"] = cell["lattice"][len(cell["lattice"])//2:]

        # Organelles emerge naturally from genome
        if np.random.rand() < (ord(genome_seq[idx % genome_len]) % 5)/5:
            org = spawn_organelles(pos, idx)
            cell["organelles"].append(org)

        # Holographic backpressure / decay
        for org in cell["organelles"]:
            for i, p in enumerate(org["positions"]):
                nearest_idx = np.random.randint(0,len(cell["lattice"]))
                p += (cell["lattice"][nearest_idx]-p)*0.02
                p *= 0.995
            org["marker"].set_data(pos=org["positions"])

        # Genome-driven cell division
        offspring = check_division(cell)
        if offspring:
            new_cells.append(offspring)

        cell["frame"] += 1

    cells.extend(new_cells)

    # Merge all lattice points holographically for visualization
    all_positions = np.vstack([cell["lattice"] for cell in cells])
    if not hasattr(update, "strand_vis"):
        update.strand_vis = Line(pos=all_positions, color=(1,1,1,0.7), width=2, parent=view.scene)
    else:
        update.strand_vis.set_data(all_positions)

    # Camera rotation holographically driven by total frames
    total_frames = sum(cell["frame"] for cell in cells)
    view.camera.azimuth = total_frames * 0.07
    view.camera.elevation = 25 + 10*np.sin(total_frames*0.002)

    canvas.update()

# ---------- RUN ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    app.run()
