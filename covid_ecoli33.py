"""
Generative DNA φ-Harmonic Spiral — lattice-driven volumetric cells
Requirements: pip install vispy pyqt6
Run: python generative_spiral_cells_lattice.py
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
MAX_POINTS = 8000
MAX_ORGANELLES = 500

# ---------- FASTA LOADER ----------

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


def load_genome(fasta_file):
    seq = []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"): continue
            seq.extend(list(line.strip().upper()))
    return seq

fasta_path = find_covid_fasta()
if not os.path.exists(fasta_path):
    raise FileNotFoundError(f"FASTA not found: {fasta_path}")
genome_seq = load_genome(fasta_path)
genome_len = len(genome_seq)
print("Genome length:", genome_len)

# ---------- VISPY SCENE ----------
comp_canvas = scene.SceneCanvas(keys='interactive', size=(1200,800),
                                bgcolor='#000011', title="Lattice-driven DNA Cells")
comp_view = comp_canvas.central_widget.add_view()
comp_view.camera = 'turntable'

strand_vis_1 = Line(pos=np.zeros((1,3)), color=(0,1,1,0.9), width=2, parent=comp_view.scene)
strand_vis_2 = Line(pos=np.zeros((1,3)), color=(1,0.5,0,0.9), width=2, parent=comp_view.scene)
donut_trail = Line(pos=np.zeros((1,3)), color=(1,1,0,0.5), width=1.5, parent=comp_view.scene)

organelles = []  # list of dicts with marker and local lattice coords
labels = []
accum_s1 = []
accum_s2 = []
accum_donut = []

frame = 0

# ---------- HELPER FUNCTIONS ----------
def consensus_dim_for_base(b):
    dim_counts = [m.get(b,1)-1 for m in all_mappings]
    return int(np.round(np.mean(dim_counts)))

def spawn_organelle(center, color_rgb, size=0.5, n=12):
    pts = center + np.zeros((n,3))  # start exactly at center
    rgba = list(Color(color_rgb).rgba)
    rgba[3] = 1.0
    mark = Markers(pos=pts, face_color=rgba, edge_color=None, size=5, parent=comp_view.scene)
    return {"marker": mark, "center": center.copy(), "size": size}

# Deterministic lattice-driven displacement
def lattice_push(org_pos, lattice_nodes, strength=0.02):
    # Push each point away from nearest lattice node
    idx = np.argmin(np.linalg.norm(lattice_nodes - org_pos, axis=1))
    nearest = lattice_nodes[idx]
    vec = org_pos - nearest
    norm = np.linalg.norm(vec) + 1e-6
    return org_pos + vec / norm * strength

# ---------- UPDATE LOOP ----------
def update(ev):
    global frame, accum_s1, accum_s2, accum_donut, organelles

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

    # Slow division drift
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
        drop = len(accum_s1)//3
        accum_s1[:] = accum_s1[drop:]
        accum_s2[:] = accum_s2[drop:]
        accum_donut[:] = accum_donut[drop:]

    donut_center = (composite_s1 + composite_s2)/2.0
    accum_donut.append(donut_center.copy())

    # Spawn organelles deterministically on FASTA positive positions
    dim = consensus_dim_for_base(base)
    color_rgb = geometries[min(dim,len(geometries)-1)][2]
    if len(organelles) < MAX_ORGANELLES:
        organelles.append(spawn_organelle(donut_center, color_rgb, size=0.5 + (dim/8.0)))

    # Update lattice nodes for deterministic push
    lattice_nodes = np.array(accum_s1 + accum_s2)

    # Update organelle positions according to lattice
    for org in organelles:
        pts = org['marker'].pos
        new_pts = np.array([lattice_push(p, lattice_nodes) for p in pts])
        org['marker'].set_data(pos=new_pts, face_color=org['marker'].face_color, size=5)

    # Update visuals
    strand_vis_1.set_data(np.array(accum_s1))
    strand_vis_2.set_data(np.array(accum_s2))
    if len(accum_donut) > 1:
        donut_trail.set_data(np.array(accum_donut))

    # Labels
    if frame % 160 == 0:
        lbl = Text(f"{base}:{geometries[min(dim,7)][3]}",
                   pos=donut_center + [0,0,0.25],
                   color=color_rgb, font_size=9, bold=True, parent=comp_view.scene)
        labels.append(lbl)

    # Rotate camera
    comp_view.camera.azimuth = frame * 0.12
    comp_view.camera.elevation = 15 + 8*np.sin(frame*0.002)

    comp_canvas.update()

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == "__main__":
    comp_canvas.show()
    app.run()
