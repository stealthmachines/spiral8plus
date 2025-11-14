"""
Generative DNA φ-Harmonic Spiral — deterministic genome-driven cells & division
Requirements: pip install vispy pyqt6
Run: python generative_spiral_cells_deterministic.py
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
MAX_POINTS = 8000

# ---------- FASTA LOADER ----------
def load_genome(fasta_file):
    seq = []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"): continue
            seq.extend(list(line.strip().upper()))
    return seq

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
comp_view.camera = 'turntable'

strand_vis_1 = Line(pos=np.zeros((1,3)), color=(0,1,1,0.9), width=2, parent=comp_view.scene)
strand_vis_2 = Line(pos=np.zeros((1,3)), color=(1,0.5,0,0.9), width=2, parent=comp_view.scene)
donut_trail = Line(pos=np.zeros((1,3)), color=(1,1,0,0.5), width=1.5, parent=comp_view.scene)

organelles = []
echo_shells = []
labels = []

accum_s1 = []
accum_s2 = []
accum_donut = []

frame = 0

# ---------- HELPERS ----------
def consensus_dim_for_base(b):
    dim_counts = [m.get(b,1)-1 for m in all_mappings]
    return int(np.round(np.mean(dim_counts)))

# Deterministic organelle
def spawn_organelle_deterministic(center, base, dim):
    cluster_size = 6 + (dim % 5) * 3
    angles_offset = np.linspace(0, 2*np.pi, cluster_size, endpoint=False)
    pts = []
    for i, ang in enumerate(angles_offset):
        r = 0.4 + 0.05 * dim + 0.02 * (ord(base) % 4)
        x = center[0] + r * np.cos(ang)
        y = center[1] + r * np.sin(ang)
        z = center[2] + 0.05 * i
        pts.append([x, y, z])
    pts = np.array(pts)
    color_rgb = geometries[min(dim, len(geometries)-1)][2]
    rgba = list(Color(color_rgb).rgba)
    rgba[3] = 0.95
    mark = Markers(pos=pts, face_color=rgba, edge_color='white', size=6, parent=comp_view.scene)
    return {"marker": mark}

# Deterministic echo shell
def make_echo_shell_deterministic(center, base, dim):
    cluster_size = 12 + dim
    angles_offset = np.linspace(0, 2*np.pi, cluster_size, endpoint=False)
    pts = []
    for i, ang in enumerate(angles_offset):
        r = 0.5 + 0.03 * dim + 0.01 * (ord(base) % 4)
        x = center[0] + r * np.cos(ang)
        y = center[1] + r * np.sin(ang)
        z = center[2] + 0.03 * ((i % 3) - 1)
        pts.append([x, y, z])
    pts = np.array(pts)
    color_rgb = geometries[min(dim, len(geometries)-1)][2]
    rgba = list(Color(color_rgb).rgba)
    rgba[3] = 0.12
    mark = Markers(pos=pts, face_color=rgba, edge_color=None, size=3, parent=comp_view.scene)
    return {"marker": mark}

# ---------- UPDATE LOOP ----------
def update(ev):
    global frame, accum_s1, accum_s2, accum_donut, organelles, echo_shells

    frame += 1
    idx = frame % genome_len
    base = genome_seq[idx]

    composite_s1 = np.zeros(3)
    composite_s2 = np.zeros(3)
    for base_map in all_mappings:
        dim = base_map.get(base, 1) - 1
        _, _, col, _, alpha, verts = geometries[dim]
        r = core_radius * (1 - (idx / max(1, genome_len))**1.4)
        r = max(r, 0.5)
        theta = idx * np.radians(golden_angle_deg)
        z = np.sin(idx / max(1, genome_len) * np.pi * 4) * 1.6 + (idx / max(1, genome_len)) * 6.0
        a1 = np.radians(angles[dim])
        a2 = np.radians(-angles[dim])

        p1 = np.array([r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
                       r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
                       z])
        p2 = np.array([r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
                       r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
                       z])
        composite_s1 += p1
        composite_s2 += p2

    composite_s1 /= len(all_mappings)
    composite_s2 /= len(all_mappings)

    # Deterministic radial drift
    progress = idx / max(1, genome_len)
    drift_amp = 0.02 + 0.8 * progress
    z_bias = np.sin(frame * 0.003 + progress * 10.0) * 0.5
    composite_s1[0] += drift_amp * np.cos(frame * 0.005)
    composite_s2[0] -= drift_amp * np.cos(frame * 0.005)
    composite_s1[2] += z_bias
    composite_s2[2] -= z_bias * 0.6

    accum_s1.append(composite_s1.copy())
    accum_s2.append(composite_s2.copy())
    if len(accum_s1) > MAX_POINTS:
        drop = len(accum_s1) // 3
        accum_s1 = accum_s1[drop:]
        accum_s2 = accum_s2[drop:]
        accum_donut = accum_donut[drop:]

    # Donut center
    donut_center = (composite_s1 + composite_s2) / 2.0
    accum_donut.append(donut_center.copy())

    # Deterministic organelles & echoes
    dim = consensus_dim_for_base(base)
    cluster = spawn_organelle_deterministic(donut_center, base, dim)
    organelles.append(cluster)
    echo = make_echo_shell_deterministic(donut_center, base, dim)
    echo_shells.append(echo)

    # Update visuals
    strand_vis_1.set_data(np.array(accum_s1))
    strand_vis_2.set_data(np.array(accum_s2))
    if len(accum_donut) > 1:
        donut_trail.set_data(np.array(accum_donut))

    # Labels
    if frame % 160 == 0:
        col = geometries[min(dim, len(geometries)-1)][2]
        lbl = Text(f"{base}:{geometries[min(dim,7)][3]}", pos=donut_center + [0,0,0.25],
                   color=col, font_size=9, bold=True, parent=comp_view.scene)
        labels.append(lbl)

    # Camera
    comp_view.camera.azimuth = frame * 0.12
    comp_view.camera.elevation = 15 + 8 * np.sin(frame * 0.002)

    comp_canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == "__main__":
    comp_canvas.show()
    app.run()
