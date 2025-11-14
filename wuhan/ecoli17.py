import numpy as np
from itertools import permutations

# ---------- CONFIG ----------
fasta_file = "ecoli_k12.fasta"

# Simplified geometries (4 for split test)
geometries = [("Point",1), ("Line",2), ("Triangle",3), ("Tetrahedron",4)]
nucleotides = ["A","T","G","C"]
all_mappings = list(permutations(range(4)))  # 24 possible mappings

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
    """Structural coherence metric: smaller std dev of rung distances = faster convergence"""
    if len(centers) < 2:
        return 0
    dists = [np.linalg.norm(centers[i]-centers[i-1]) for i in range(1,len(centers))]
    return -np.std(dists)

# ---------- RUN SIMULATION ----------
def run_spiral(mapping, max_frames=10000):
    phi = (1 + np.sqrt(5))/2
    golden_angle_deg = 360 / (phi**2)

    centers = []
    best_fit = -np.inf
    convergence_frame = None

    for frame in range(max_frames):
        base = genome_seq[frame % genome_len]
        geom_idx = mapping[nucleotides.index(base)]
        _, verts = geometries[geom_idx]

        # Spiral coordinates (simplified)
        theta = frame * np.radians(golden_angle_deg)
        r = 5 + frame*0.01
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        z = frame*0.02
        centers.append(np.array([x,y,z]))

        # Evaluate fitness every 100 frames
        if frame % 100 == 0 and len(centers)>1:
            fit = fitness_metric(centers)
            # Record frame when fitness stabilizes
            if fit > best_fit:
                best_fit = fit
                convergence_frame = frame

        # Optional: percent complete
        if frame % 1000 == 0:
            percent_complete = frame/max_frames*100
            print(f"Mapping {mapping} | {percent_complete:.2f}% complete")

    return best_fit, convergence_frame

# ---------- TEST ALL MAPPINGS ----------
results = []
for mapping in all_mappings:
    print(f"Testing mapping: {mapping}")
    fit, conv_frame = run_spiral(mapping)
    results.append((mapping, fit, conv_frame))

# ---------- FIND BEST MAPPING ----------
best_mapping = max(results, key=lambda x: x[2])  # earliest convergence_frame = fastest
print("\n✅ Best mapping (fastest convergence):", best_mapping)
