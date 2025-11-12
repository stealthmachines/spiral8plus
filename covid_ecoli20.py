import numpy as np
from itertools import permutations

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
rolling_window = 200  # frames for collapse detection

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
    """Structural coherence metric: smaller std dev of rung distances = more stable"""
    if len(centers) < 2:
        return 0
    dists = [np.linalg.norm(centers[i]-centers[i-1]) for i in range(1,len(centers))]
    return -np.std(dists)

# ---------- RUN SPIRAL WITH COLLAPSE CHECK ----------
def run_spiral(mapping, max_frames=max_frames):
    phi = (1 + np.sqrt(5))/2
    golden_angle_deg = 360 / (phi**2)

    centers = []
    best_fit = -np.inf
    convergence_frame = None
    collapse_frame = None

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

        # Evaluate fitness every 50 frames
        if frame % 50 == 0 and len(centers) > rolling_window:
            fit = fitness_metric(centers[-rolling_window:])
            # Track best rolling fitness
            if fit > best_fit:
                best_fit = fit
                convergence_frame = frame

            # Collapse detection: if fitness drops dramatically
            if fit < best_fit * 0.7 and collapse_frame is None:
                collapse_frame = frame

        # Optional: percent complete
        if frame % 1000 == 0:
            percent_complete = frame/max_frames*100
            print(f"Mapping {mapping} | {percent_complete:.2f}% complete")

    return best_fit, convergence_frame, collapse_frame

# ---------- TEST ALL MAPPINGS ----------
results = []
for mapping in all_mappings:
    print(f"Testing mapping: {mapping}")
    fit, conv_frame, collapse_frame = run_spiral(mapping)
    results.append((mapping, fit, conv_frame, collapse_frame))

# ---------- FIND BEST MAPPING ----------
# Fastest long-term convergence before collapse
best_mapping = max(results, key=lambda x: x[2] if x[3] is None else x[3])
print("\n✅ Best mapping (stable convergence):", best_mapping)

