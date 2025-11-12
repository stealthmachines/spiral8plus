import numpy as np
from itertools import permutations
from concurrent.futures import ThreadPoolExecutor

# ---------- CONFIG ----------
fasta_file = "ecoli_k12.fasta"

geometries = [("Point",1), ("Line",2), ("Triangle",3), ("Tetrahedron",4)]
nucleotides = ["A","T","G","C"]
all_mappings = list(permutations(range(4)))  # 24 possible mappings

max_frames = 10000
eval_interval = 100  # check fitness every N frames

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

# ---------- FITNESS METRIC (ROLLING STD) ----------
def fitness_metric_rolling(frame_num, coords_buffer):
    """Rolling std dev of distances between consecutive centers"""
    if len(coords_buffer) < 2:
        return 0.0
    diffs = np.diff(coords_buffer, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return -np.std(dists)

# ---------- RUN SINGLE MAPPING ----------
def run_spiral(mapping):
    phi = (1 + np.sqrt(5)) / 2
    golden_angle_deg = 360 / (phi**2)

    best_fit = -np.inf
    convergence_frame = None

    # rolling buffer for last few coordinates
    coords_buffer = []

    for frame in range(max_frames):
        base = genome_seq[frame % genome_len]
        geom_idx = mapping[nucleotides.index(base)]
        _, verts = geometries[geom_idx]

        # Spiral coordinates (simplified)
        theta = frame * np.radians(golden_angle_deg)
        r = 5 + frame * 0.01
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = frame * 0.02
        coords_buffer.append(np.array([x, y, z]))

        # Keep buffer small (sliding window)
        if len(coords_buffer) > 1000:
            coords_buffer.pop(0)

        # Evaluate fitness periodically
        if frame % eval_interval == 0 and len(coords_buffer) > 1:
            fit = fitness_metric_rolling(frame, np.array(coords_buffer))
            if fit > best_fit:
                best_fit = fit
                convergence_frame = frame

        # Optional: percent complete
        if frame % 2000 == 0:
            percent_complete = frame / max_frames * 100
            print(f"Mapping {mapping} | {percent_complete:.2f}% complete")

    return mapping, best_fit, convergence_frame

# ---------- RUN ALL MAPPINGS IN PARALLEL ----------
def run_all_mappings_parallel():
    results = []
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [executor.submit(run_spiral, mapping) for mapping in all_mappings]
        for future in futures:
            results.append(future.result())
    return results

# ---------- MAIN ----------
if __name__ == "__main__":
    print("Starting parallel split test of 24 mappings...")
    results = run_all_mappings_parallel()
    best_mapping = max(results, key=lambda x: x[2])  # earliest convergence
    print("\n✅ Best mapping (fastest convergence):", best_mapping)
