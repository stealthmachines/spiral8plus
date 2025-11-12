import numpy as np
from itertools import permutations
from concurrent.futures import ThreadPoolExecutor, as_completed

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


# ---------- CONFIG ----------
fasta_file = find_covid_fasta()
geometries = [("Point",1), ("Line",2), ("Triangle",3), ("Tetrahedron",4)]
nucleotides = ["A","T","G","C"]
all_mappings = list(permutations(range(4)))  # 24 possible mappings

max_frames = 10000
window_size = 200  # rolling window for fitness
print_every = 1000

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

# ---------- RUN SPIRAL ----------
def run_spiral(mapping, max_frames=max_frames, window_size=window_size):
    phi = (1 + np.sqrt(5))/2
    golden_angle_deg = 360 / (phi**2)

    centers = []
    best_fit = -np.inf
    convergence_frame = None
    stability_frame = None

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
        centers.append(np.array([x, y, z], dtype=np.float32))

        # Maintain rolling window
        if len(centers) > window_size:
            centers.pop(0)

        # Evaluate fitness every 100 frames
        if frame % 100 == 0 and len(centers) > 1:
            fit = fitness_metric(centers)
            # Record convergence (first frame fitness improves)
            if fit > best_fit:
                best_fit = fit
                convergence_frame = frame
            # Record stability (fitness stops decreasing)
            if stability_frame is None and fit <= best_fit:
                stability_frame = frame

        # Optional: percent complete
        if frame % print_every == 0:
            percent_complete = frame/max_frames*100
            print(f"Mapping {mapping} | {percent_complete:.1f}% complete")

    return mapping, best_fit, convergence_frame, stability_frame

# ---------- PARALLEL RUN ----------
def run_all_mappings_parallel():
    results = []
    print("Starting parallel split test of 24 mappings...")
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(run_spiral, m): m for m in all_mappings}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Completed mapping: {result[0]} | Best fitness: {result[1]:.4f} | "
                  f"Convergence: {result[2]} | Stability: {result[3]}")
    return results

# ---------- EXECUTE ----------
if __name__ == "__main__":
    results = run_all_mappings_parallel()
    # Best by earliest convergence
    best_mapping = min(results, key=lambda x: x[2])
    print("\n✅ Best mapping (fastest convergence):", best_mapping)
