"""
Generative DNA φ-Harmonic Spiral — volumetric, genome-driven cells & division
Requirements: pip install vispy pyqt6
Run: python generative_spiral_cells.py
"""

import os
import numpy as np
import itertools
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# ---------- CONFIG ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)  # 137.507°

bases = ['A','T','G','C']
all_mappings = [dict(zip(bases,p)) for p in itertools.permutations([1,2,3,4])]
print(f"Total mappings: {len(all_mappings)}")  # 24

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
core_radius = 15.0
strand_sep = 0.6
N = 600

# ---------- FASTA LOADER ----------
def load_genome(fasta_file):
    seq = []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"): continue
            seq.extend(list(line.strip().upper()))
    return seq

# EDIT THIS PATH IF NEEDED
fasta_path = "ecoli_k12.fasta"
if not os.path.exists(fasta_path):
    raise FileNotFoundError(f"FASTA not found: {fasta_path}")
genome_seq = load_genome(fasta_path)
genome_len = len(genome_seq)
print("Genome length:", genome_len)

# ---------- VISPY SCENE ----------
comp_canvas = scene.SceneCanvas(keys='interactive', size=(1200,800),
                                bgcolor='black', title="Generative DNA Cells (FASTA-driven)")
comp_view = comp_canvas.central_widget.add_view()
comp_view.camera = 'turntable'  # interactive

# Persistent visuals: two main strands that grow (list of points)
strand_vis_1 = Line(pos=np.zeros((1,3)), color=(0,1,1,0.9), width=2, parent=comp_view.scene)
strand_vis_2 = Line(pos=np.zeros((1,3)), color=(1,0.5,0,0.9), width=2, parent=comp_view.scene)

# Collections for organelles, echoes, labels (visual objects)
organelles = []     # list of Markers (small clusters)
echo_shells = []    # faint shells to solidify organelles
labels = []

# Accumulated coordinates (growing like print layers)
accum_s1 = []   # list of 3D points
accum_s2 = []

# Soft fade control: older markers get lower alpha (we manage by recreating them)
MAX_POINTS = 8000   # cap total points stored to avoid memory explosion
MAX_ORGANELLES = 200

frame = 0

# ---------- Helper: consensus-driven parameters ----------
def consensus_dim_for_base(b):
    # returns mean dim across the 24 mappings for base b
    dim_counts = [m.get(b,1)-1 for m in all_mappings]
    return int(np.round(np.mean(dim_counts)))

def organelle_params_from_base(b):
    """Return (spawn_prob, cluster_size, radial_bias, noise_scale) driven by base+consensus."""
    d = consensus_dim_for_base(b)
    # simple mapping heuristics — tune to taste
    spawn_prob = 0.02 + (d / 16.0) * 0.12     # A higher dim -> slightly higher chance
    cluster_size = 6 + (d % 5) * 3
    radial_bias = 0.2 + (d / 16.0) * 0.9
    noise_scale = 0.08 + (d / 16.0) * 0.4
    return spawn_prob, cluster_size, radial_bias, noise_scale

# ---------- Helper: spawn organelle cluster (returns vispy Markers) ----------
def spawn_organelle(center, color_rgb, size, n=12, persistence=180):
    pts = center + np.random.normal(scale=0.3, size=(n,3)) * size
    rgba = list(Color(color_rgb).rgba)
    rgba[3] = 0.95
    mark = Markers(pos=pts, face_color=rgba, edge_color='white', size=6, parent=comp_view.scene)
    # attach persistence (frames to live) so we can fade/remove later
    return {"marker": mark, "age": 0, "life": persistence}

# ---------- Helper: echo shell generator ----------
def make_echo_shell(center, scale, color_rgb, alpha=0.12):
    # small cloud scaled inwards/outwards to hint at solidification
    shell_pts = center + (np.random.normal(scale=0.4, size=(max(8, int(12*scale)),3)) * scale)
    rgba = list(Color(color_rgb).rgba)
    rgba[3] = alpha
    shell = Markers(pos=shell_pts, face_color=rgba, edge_color=None, size=3, parent=comp_view.scene)
    return {"marker": shell, "age": 0, "life": 300}

# ---------- UPDATE LOOP: additive growth + organelles + division ----------
def update(ev):
    global frame, accum_s1, accum_s2, organelles, echo_shells

    frame += 1
    idx = frame % genome_len
    base = genome_seq[idx]

    # compute a genomic-driven point for this frame (consensus average across mappings)
    composite_s1 = np.zeros(3)
    composite_s2 = np.zeros(3)
    for base_map in all_mappings:
        dim = base_map.get(base, 1) - 1
        _, _, col, _, alpha, verts = geometries[dim]

        # radius decays with genome progress to encourage initial bulk, later refinement
        r = core_radius * (1 - (idx / max(1, genome_len))**1.4)
        r = max(r, 0.5)
        theta = idx * np.radians(golden_angle_deg)
        z = np.sin(idx / max(1, genome_len) * np.pi * 4) * 1.6 + (idx / max(1, genome_len)) * 6.0
        a1 = np.radians(angles[dim])
        a2 = np.radians(-angles[dim])

        p1 = np.array([
            r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
            r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
            z
        ])
        p2 = np.array([
            r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
            r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
            z
        ])
        composite_s1 += p1
        composite_s2 += p2

    # average across all mappings
    composite_s1 /= len(all_mappings)
    composite_s2 /= len(all_mappings)

    # ---------- volumetric diffusion: push points outward and jitter slightly
    # radial offset controlled by base-driven radial_bias
    spawn_prob, cluster_size, radial_bias, noise_scale = organelle_params_from_base(base)
    # radial direction from core (0,0) in XY plane
    def radial_jitter(p, bias=radial_bias, noise=noise_scale):
        # push in XY plane away from origin with some noise and small Z jitter
        dir_xy = np.array([p[0], p[1], 0.0])
        norm = np.linalg.norm(dir_xy[:2]) + 1e-6
        unit = dir_xy / norm
        offset = unit * (bias * np.random.rand()) + np.random.normal(scale=noise, size=3)
        return p + offset

    composite_s1 = radial_jitter(composite_s1)
    composite_s2 = radial_jitter(composite_s2) * np.array([1.0,1.0,1.0])  # keep same dims
    # small anti-correlated noise so two strands don't collapse into same line
    anticor = np.random.normal(scale=0.02, size=3)
    composite_s1 += anticor
    composite_s2 -= anticor

    # ---------- division drift: slowly separate the two strands into daughter centers
    # Use a slow oscillation combined with a genome-progress drift
    progress = idx / max(1, genome_len)
    drift_amp = 0.02 + 0.8 * progress
    z_bias = np.sin(frame * 0.003 + progress * 10.0) * 0.5
    composite_s1[0] += drift_amp * np.cos(frame * 0.005)
    composite_s2[0] -= drift_amp * np.cos(frame * 0.005)
    composite_s1[2] += z_bias
    composite_s2[2] -= z_bias * 0.6

    # ---------- append to accumulative printed geometry (stacking)
    accum_s1.append(composite_s1.copy())
    accum_s2.append(composite_s2.copy())

    # Cap lists for performance: keep recent history while preserving persistent shells/organelles
    if len(accum_s1) > MAX_POINTS:
        # drop oldest third
        drop = len(accum_s1) // 3
        accum_s1 = accum_s1[drop:]
        accum_s2 = accum_s2[drop:]

    # ---------- organelle nucleation: genome-driven probabilistic spawning
    if np.random.rand() < spawn_prob:
        # spawn an organelle near the local center between strands
        local_center = (composite_s1 + composite_s2) / 2.0
        # color depends on consensus dim
        dim = consensus_dim_for_base(base)
        color_rgb = geometries[min(dim, len(geometries)-1)][2]
        cluster = spawn_organelle(local_center, color_rgb, size=0.8 + radial_bias, n=cluster_size, persistence=260)
        organelles.append(cluster)
        # also make an echo shell to start solidifying it
        echo = make_echo_shell(local_center, scale=0.6 + radial_bias, color_rgb=color_rgb, alpha=0.08)
        echo_shells.append(echo)

    # ---------- occasional larger organelle (rare) to simulate nucleus-like formation
    if (frame % 1000) == 0:
        big_center = (accum_s1[-1] + accum_s2[-1]) / 2.0
        big_echo = make_echo_shell(big_center, scale=1.8, color_rgb='gold', alpha=0.12)
        echo_shells.append(big_echo)

    # ---------- update visuals: strands (as lines of accumulated points)
    strand_vis_1.set_data(np.array(accum_s1))
    strand_vis_2.set_data(np.array(accum_s2))

    # ---------- fade / age organelles & echoes, remove when expired
    # ---------- fade / age organelles & echoes, remove when expired ----------
    new_organelles = []
    for ent in organelles:
        ent['age'] += 1
        m = ent['marker']
        base_rgba = ent.get('base_rgba', (1,1,1,1))
        alpha = max(0.02, 1.0 - (ent['age'] / ent['life']))
        rgba = list(base_rgba)
        rgba[3] = alpha
        try:
            m.set_data(pos=m._pos, face_color=rgba)
        except Exception:
            pass
        if ent['age'] < ent['life']:
            ent['base_rgba'] = rgba  # store for next cycle
            new_organelles.append(ent)
        else:
            try:
                m.parent = None
            except Exception:
                pass
    organelles = new_organelles[:MAX_ORGANELLES]

    new_echoes = []
    for ent in echo_shells:
        ent['age'] += 1
        m = ent['marker']
        base_rgba = ent.get('base_rgba', (1,1,1,1))
        alpha = max(0.0, 0.12 * (1.0 - ent['age'] / ent['life']))
        rgba = list(base_rgba)
        rgba[3] = alpha
        try:
            m.set_data(pos=m._pos, face_color=rgba)
        except Exception:
            pass
        if ent['age'] < ent['life']:
            ent['base_rgba'] = rgba
            new_echoes.append(ent)
        else:
            try:
                m.parent = None
            except Exception:
                pass
    echo_shells = new_echoes

    # ---------- occasional rung / label for clarity (slower cadence)
    if frame % 160 == 0:
        base_here = base
        dim = consensus_dim_for_base(base_here)
        col = geometries[min(dim, len(geometries)-1)][2]
        cen = (accum_s1[-1] + accum_s2[-1]) / 2.0
        lbl = Text(f"{base_here}:{geometries[min(dim,7)][3]}", pos=cen + [0,0,0.25],
                   color=col, font_size=9, bold=True, parent=comp_view.scene)
        labels.append(lbl)

    # rotate camera slowly to reveal 3D structure
    comp_view.camera.azimuth = frame * 0.12
    comp_view.camera.elevation = 15 + 8 * np.sin(frame * 0.002)

    comp_canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == "__main__":
    comp_canvas.show()
    app.run()
