# ecoli_fusion.py
# Fusion of composite (negative) + echo cell (positive)
# Run: python ecoli_fusion.py
# Requires: pip install vispy pyqt6 numpy

import os
import numpy as np
import itertools
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- CONFIG & CONSTANTS ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)
N = 600                      # sample points per strand trace
core_radius = 15.0
strand_sep = 0.55
MAX_POINTS = 9000
MAX_ORGANELLES = 300

# ---------- Mappings ----------
bases = ['A', 'T', 'G', 'C']
# 24 permutations for substrate composite
all_mappings = [dict(zip(bases, p)) for p in itertools.permutations([1, 2, 3, 4])]

# positive activator single-base mapping (keeps your simpler mapping)
base_map = {'A': 5, 'T': 2, 'G': 4, 'C': 1}

# shape metadata (dim,note,color,name,alpha,verts)
geometries = [
    (1, 'C', 'red',          'Point',        0.015269, 1),
    (2, 'D', 'green',        'Line',         0.008262, 2),
    (3, 'E', 'violet',       'Triangle',     0.110649, 3),
    (4, 'F', 'mediumpurple', 'Tetrahedron', -0.083485, 4),
    (5, 'G', 'blue',         'Pentachoron',  0.025847, 5),
    (6, 'A', 'indigo',       'Hexacross',   -0.045123, 12),
    (7, 'B', 'purple',       'Heptacube',    0.067891, 14),
    (8, 'C', 'white',        'Octacube',     0.012345, 16),
]

angles = [i * golden_angle_deg for i in range(8)]

# ---------- FASTA loader ----------
def load_genome(fasta_file):
    seq = []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"): continue
            seq.extend(list(line.strip().upper()))
    return seq

fasta_path = "ecoli_k12.fasta"
if not os.path.exists(fasta_path):
    raise FileNotFoundError(f"FASTA not found: " + fasta_path)
genome_seq = load_genome(fasta_path)
genome_len = len(genome_seq)
print("Total mappings:", len(all_mappings))
print("Genome length:", genome_len)

# ---------- VISPY scene ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 800), bgcolor='black', title="ecoli_fusion")
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# visible activator strands (grow over time)
vis_strand_A = Line(pos=np.zeros((1, 3)), color=(0.2, 0.9, 1.0, 0.95), width=2, parent=view.scene)
vis_strand_B = Line(pos=np.zeros((1, 3)), color=(1.0, 0.6, 0.1, 0.95), width=2, parent=view.scene)

# substrate lines (optional lightweight visualization — commented out to keep substrate "hidden")
# sub_strand1 = Line(pos=np.zeros((1,3)), color=(0.1,0.1,0.12,0.05), width=1, parent=view.scene)

# collections
accum_A, accum_B = [], []
substrate_mem_A, substrate_mem_B = [], []   # composite substrate per-sample (kept as memory)
organelles = []   # dicts with {'marker','pos','base_rgba','age','life'}
echo_shells = []
rungs = []
labels = []

# text UI
progress_text = Text("0.0%", pos=[0, 0, 20], color='white', font_size=20, anchor_x='center', parent=view.scene)

frame = 0

# ---------- helper: consensus dimension (for organelle color choices, heuristics) ----------
def consensus_dim_for_base(b):
    dim_counts = [m.get(b, 1) - 1 for m in all_mappings]
    return int(np.round(np.mean(dim_counts)))

# ---------- organelle helpers ----------
def spawn_organelle(center, color_rgb, n=18, size_scale=0.6, life=240):
    pts = center + np.random.normal(scale=0.25 * size_scale, size=(n, 3))
    rgba = list(Color(color_rgb).rgba)
    rgba[3] = 0.95
    mark = Markers(pos=pts, face_color=rgba, edge_color=(1,1,1,0.15), size=6, parent=view.scene)
    return {'marker': mark, 'pos': pts, 'base_rgba': rgba, 'age': 0, 'life': life}

def make_echo_shell(center, scale=1.0, color_rgb='white', alpha=0.08, life=300):
    pts = center + np.random.normal(scale=0.35 * scale, size=(12 + int(8*scale), 3))
    rgba = list(Color(color_rgb).rgba)
    rgba[3] = alpha
    marker = Markers(pos=pts, face_color=rgba, edge_color=None, size=3, parent=view.scene)
    return {'marker': marker, 'pos': pts, 'base_rgba': rgba, 'age': 0, 'life': life}

# ---------- main update: compute substrate + activator + fusion ----------
def update(ev):
    global frame, accum_A, accum_B, substrate_mem_A, substrate_mem_B, organelles, echo_shells

    frame += 1
    # sample sequence along t (we will compute arrays for N points per strand like earlier)
    t = np.linspace(0, frame, N)

    # ---- compute substrate (composite of all_mappings) ----
    # We'll compute composite arrays (hidden morphogen cloud)
    composite_A = np.zeros((N, 3), dtype=np.float32)
    composite_B = np.zeros((N, 3), dtype=np.float32)

    for base_map in all_mappings:
        sA, sB = [], []
        for tt in t:
            idx = int(tt) % genome_len
            base = genome_seq[idx]
            dim = base_map.get(base, 1) - 1
            _, _, col, _, alpha, verts = geometries[dim]

            r = core_radius * (1 - (tt / genome_len) ** 1.45)
            r = max(r, 0.4)
            theta = tt * np.radians(golden_angle_deg)
            z = np.sin(tt / genome_len * np.pi * 4) * 1.6 + (tt / genome_len) * 6.0
            a1 = np.radians(angles[dim])
            a2 = np.radians(-angles[dim])

            pA = np.array([r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                           r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
                           z])
            pB = np.array([r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
                           r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
                           z])
            sA.append(pA)
            sB.append(pB)

        composite_A += np.array(sA)
        composite_B += np.array(sB)

    composite_A /= len(all_mappings)
    composite_B /= len(all_mappings)

    # preserve a short substrate memory (low-weighted), decayed each frame
    substrate_mem_A.append(composite_A.copy())
    substrate_mem_B.append(composite_B.copy())
    if len(substrate_mem_A) > 10:
        substrate_mem_A.pop(0)
        substrate_mem_B.pop(0)

    # Optionally compute a temporal Gaussian-blurred substrate center (weighted average)
    weights = np.exp(-0.5 * np.linspace(len(substrate_mem_A)-1, 0, len(substrate_mem_A)) / 3.0)
    weights /= (weights.sum() + 1e-12)
    blended_sub_A = sum(w * s for w, s in zip(weights, substrate_mem_A))
    blended_sub_B = sum(w * s for w, s in zip(weights, substrate_mem_B))

    # ---- compute activator (FASTA-driven single mapping) ----
    sA_act = []
    sB_act = []
    for tt in t:
        idx = int(tt) % genome_len
        base = genome_seq[idx]
        dim = base_map.get(base, 1) - 1
        _, _, col, _, alpha, verts = geometries[dim]

        r = core_radius * (1 - (tt / genome_len) ** 1.3)
        r = max(r, 0.45)
        theta = tt * np.radians(golden_angle_deg)
        z = np.sin(tt / genome_len * np.pi * 4) * 1.4 + (tt / genome_len) * 6.2
        a1 = np.radians(angles[dim])
        a2 = np.radians(-angles[dim])

        pA = np.array([r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                       r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
                       z])
        pB = np.array([r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
                       r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
                       z])
        sA_act.append(pA)
        sB_act.append(pB)

    sA_act = np.array(sA_act)
    sB_act = np.array(sB_act)

    # ---- fusion: blend activator + substrate per-point ----
    # alpha can vary slowly over time to give breathing (and be influenced by genome local features)
    alpha_dyn = 0.55 + 0.35 * np.sin(frame * 0.0023)
    fused_A = alpha_dyn * sA_act + (1.0 - alpha_dyn) * blended_sub_A
    fused_B = alpha_dyn * sB_act + (1.0 - alpha_dyn) * blended_sub_B

    # small volumetric jitter to break planarity
    jitter_scale = 0.02 + 0.06 * (0.5 + 0.5 * np.sin(frame * 0.0011))
    fused_A += np.random.normal(scale=jitter_scale, size=fused_A.shape)
    fused_B += np.random.normal(scale=jitter_scale, size=fused_B.shape)

    # ---- compute energy / delta that drives organelle nucleation ----
    # per-sample norm between activator and substrate (use fused vs substrate difference)
    deltaA = np.linalg.norm(fused_A - blended_sub_A, axis=1)
    deltaB = np.linalg.norm(fused_B - blended_sub_B, axis=1)
    # normalize
    denom = max(deltaA.max(), deltaB.max(), 1e-6)
    energy = (deltaA + deltaB) / (2.0 * denom)

    # where energy peaks -> higher chance to spawn organelle clusters
    # sample a few indices proportionally to energy
    prob_spawn_map = np.clip(energy, 0.0, 1.0) ** 1.7  # emphasize peaks

    # For performance, we will create only a few spawn attempts per frame
    spawn_attempts = 3
    for _ in range(spawn_attempts):
        i = np.random.randint(0, N)
        p_energy = prob_spawn_map[i]
        # bias spawn based on current base at that sample
        idx_seq = int(t[i]) % genome_len
        b = genome_seq[idx_seq]
        # dynamic spawn probability scaling
        spawn_prob = 0.008 + 0.2 * p_energy
        if np.random.rand() < spawn_prob and len(organelles) < MAX_ORGANELLES:
            center = (fused_A[i] + fused_B[i]) / 2.0
            dim = consensus_dim_for_base(b)
            color_rgb = geometries[min(dim, len(geometries)-1)][2]
            cluster = spawn_organelle(center, color_rgb, n=12 + (dim % 6), size_scale=0.6 + dim * 0.05, life=220 + dim*40)
            organelles.append(cluster)
            # echo shell
            echo = make_echo_shell(center, scale=0.6 + dim*0.08, color_rgb=color_rgb, alpha=0.08, life=280)
            echo_shells.append(echo)

    # ---- occasional larger structures (rungs) driven by FASTA consensus ----
    if frame % 180 == 0:
        sample_idx = frame % genome_len
        b = genome_seq[sample_idx]
        dim = base_map.get(b, 1) - 1
        _, _, col, name, alpha_g, verts = geometries[dim]
        # pick a slice near the end of fused arrays for display
        pick = min(N - 1, int((dim / 8.0) * N + 4))
        ptsA = fused_A[max(0, pick - verts): pick][:verts]
        ptsB = fused_B[max(0, pick - verts): pick][:verts]
        if ptsA.size and ptsB.size:
            allpts = np.vstack((ptsA, ptsB))
            rgba = list(Color(col).rgba)
            rgba[3] = 0.95
            mark = Markers(pos=allpts, face_color=rgba, edge_color='white', size=6, parent=view.scene)
            rungs.append(mark)
            # label
            cen = allpts.mean(axis=0)
            lbl = Text(f"{b}:{name}", pos=cen + [0, 0, 0.25], color=col, font_size=10, parent=view.scene)
            labels.append(lbl)

    # ---- accumulate visible fused points gradually (additive stack) ----
    # we append a single center point per frame (average of mid-slices) to keep growth rate reasonable
    mid_idx = N // 2
    centerA = fused_A[mid_idx]
    centerB = fused_B[mid_idx]
    accum_A.append(centerA.copy())
    accum_B.append(centerB.copy())

    if len(accum_A) > MAX_POINTS:
        drop = len(accum_A) // 3
        accum_A = accum_A[drop:]
        accum_B = accum_B[drop:]

    # set line visuals
    vis_strand_A.set_data(np.array(accum_A))
    vis_strand_B.set_data(np.array(accum_B))

    # ---- age & fade organelles + echoes safely (VisPy: use set_data to update face_color) ----
    new_orgs = []
    for e in organelles:
        e['age'] += 1
        alpha_now = max(0.02, e['base_rgba'][3] * (1.0 - e['age'] / e['life']))
        rgba = list(e['base_rgba'])
        rgba[3] = alpha_now
        try:
            e['marker'].set_data(pos=e['pos'], face_color=rgba)
        except Exception:
            pass
        if e['age'] < e['life']:
            new_orgs.append(e)
        else:
            try: e['marker'].parent = None
            except Exception: pass
    organelles = new_orgs

    new_echoes = []
    for sc in echo_shells:
        sc['age'] += 1
        alpha_now = max(0.0, sc['base_rgba'][3] * (1.0 - sc['age'] / sc['life']))
        rgba = list(sc['base_rgba'])
        rgba[3] = alpha_now
        try:
            sc['marker'].set_data(pos=sc['pos'], face_color=rgba)
        except Exception:
            pass
        if sc['age'] < sc['life']:
            new_echoes.append(sc)
        else:
            try: sc['marker'].parent = None
            except Exception: pass
    echo_shells = new_echoes

    # ---- camera + progress UI ----
    view.camera.azimuth = frame * 0.11
    view.camera.elevation = 12 + 6 * np.sin(frame * 0.0017)

    # progress text (how far through genome in percent modulus)
    percent = (frame % genome_len) / float(max(1, genome_len)) * 100.0
    progress_text.text = f"{percent:.2f}%"

    canvas.update()

# ---------- start timer ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    canvas.show()
    app.run()
