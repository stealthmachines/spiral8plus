# dna_mapping_tester.py
# --------------------------------------------------------------
# Automated mapping test for DNA φ-spiral lattice
# --------------------------------------------------------------

import numpy as np
from itertools import permutations
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color
import os

# ---------- CONFIG ----------
fasta_file = "ecoli_k12.fasta"
output_dir = "mapping_results"
os.makedirs(output_dir, exist_ok=True)

# Geometries (simplified for split-test)
geometries = [
    ("Point", 1, "red"),
    ("Line", 2, "green"),
    ("Triangle", 3, "blue"),
    ("Tetrahedron", 4, "yellow")
]

nucleotides = ["A", "T", "G", "C"]
all_mappings = list(permutations(range(len(geometries))))  # 24 permutations

# ---------- LOAD GENOME ----------
def load_genome(fasta_file):
    seq = []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                continue
            seq.extend(list(line.strip().upper()))
    return seq

genome_seq = load_genome(fasta_file)
genome_len = len(genome_seq)

# ---------- STRUCTURE FITNESS ----------
def fitness_metric(centers):
    """Simple structural metric: mean distance between rung centers"""
    if len(centers) < 2:
        return 0
    dists = [np.linalg.norm(centers[i] - centers[i-1]) for i in range(1, len(centers))]
    return -np.std(dists)  # more uniform spacing = higher fitness

# ---------- RUN φ-SPIRAL SIMULATION ----------
def run_spiral(mapping):
    # VisPy setup
    canvas = scene.SceneCanvas(keys='interactive', size=(800,600), bgcolor='black', show=False)
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'

    # φ constants
    phi = (1 + np.sqrt(5))/2
    golden_angle_deg = 360 / (phi**2)

    frame = 0
    centers = []

    for base_idx, base in enumerate(genome_seq):
        geom_idx = mapping[nucleotides.index(base)]
        name, verts, col = geometries[geom_idx]

        # Simple spiral position (2D for demo)
        theta = frame * np.radians(golden_angle_deg)
        r = 5 + frame * 0.01
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = frame * 0.02
        centers.append(np.array([x,y,z]))

        # Optionally, draw a marker (minimal for speed)
        # mark = Markers(pos=[[x,y,z]], face_color=col, size=5, parent=view.scene)

        # Track progress
        percent_complete = (frame+1)/genome_len*100
        if frame % 10000 == 0:
            print(f"Mapping {mapping} | {percent_complete:.2f}% complete")
        frame += 1

    # Compute fitness
    fit = fitness_metric(centers)
    # Optional: save screenshot
    canvas.render_to_file(os.path.join(output_dir, f"mapping_{'_'.join(map(str,mapping))}.png"))
    canvas.close()
    return fit

# ---------- TEST ALL MAPPINGS ----------
results = []
for i, mapping in enumerate(all_mappings):
    print(f"Testing mapping {i+1}/{len(all_mappings)}: {mapping}")
    fit = run_spiral(mapping)
    results.append((mapping, fit))

# ---------- FIND BEST MAPPING ----------
best_mapping, best_fit = max(results, key=lambda x: x[1])
print("\n✅ Best mapping found:", best_mapping, "with fitness:", best_fit)
