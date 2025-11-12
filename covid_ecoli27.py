# 🌌 DNA φ-Harmonic Spiral Composite & Generative Stack (24 mappings)
# ------------------------------------------------------------------
# Each frame adds a new "slice" of the spiral in 3D space — like printing layer-by-layer.
# ------------------------------------------------------------------

import numpy as np
import itertools
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)

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
strand_sep = 0.5
N = 600


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
            if not line.startswith(">"):
                seq.extend(list(line.strip().upper()))
    return seq

genome_seq = load_genome(find_covid_fasta())
genome_len = len(genome_seq)

comp_canvas = scene.SceneCanvas(keys='interactive', size=(900,700),
                                bgcolor='#000011', title="Generative DNA Spiral")
comp_view = comp_canvas.central_widget.add_view()
comp_view.camera = 'turntable'

comp_strand1 = Line(pos=np.zeros((1,3)), color=(0,1,1,0.7), width=2, parent=comp_view.scene)
comp_strand2 = Line(pos=np.zeros((1,3)), color=(1,0.5,0,0.7), width=2, parent=comp_view.scene)

# --- NEW: accumulate coordinates
points_s1, points_s2 = [], []

collections = {'rungs': [], 'echoes': [], 'links': [], 'labels': [], 'centers': [], 'emerged': []}
frame = 0

def update(ev):
    global frame, points_s1, points_s2
    frame += 1

    idx = frame % genome_len
    base = genome_seq[idx]

    # Generate just ONE new point slice (layer)
    composite_s1, composite_s2 = np.zeros(3), np.zeros(3)

    for base_map in all_mappings:
        dim = base_map.get(base, 1) - 1
        _, _, col, _, alpha, verts = geometries[dim]

        r = core_radius * (1 - (idx/genome_len)**1.5)
        r = max(r, 0.5)
        theta = idx * np.radians(golden_angle_deg)
        z = np.sin(idx/genome_len * np.pi * 4) * 2 + (idx/genome_len)*8
        a1 = np.radians(angles[dim])
        a2 = np.radians(-angles[dim])

        composite_s1 += np.array([
            r*np.cos(theta)*np.cos(a1)-r*np.sin(theta)*np.sin(a1),
            r*np.sin(theta)*np.cos(a1)+r*np.cos(theta)*np.sin(a1),
            z
        ])
        composite_s2 += np.array([
            r*np.cos(theta)*np.cos(a2)-r*np.sin(theta)*np.sin(a2)+strand_sep,
            r*np.sin(theta)*np.cos(a2)+r*np.cos(theta)*np.sin(a2)-strand_sep,
            z
        ])

    composite_s1 /= len(all_mappings)
    composite_s2 /= len(all_mappings)

    # Append new layer
    points_s1.append(composite_s1)
    points_s2.append(composite_s2)

    # Update geometry incrementally
    comp_strand1.set_data(np.array(points_s1))
    comp_strand2.set_data(np.array(points_s2))

    # Build lattice occasionally
    if frame % 80 == 0:
        dim_counts = [m.get(base,1)-1 for m in all_mappings]
        cur_dim = int(np.round(np.mean(dim_counts)))
        dim, note, col, name, alpha, verts = geometries[cur_dim]

        rgba = Color(col).rgba
        cen = (composite_s1 + composite_s2) / 2
        lbl = Text(f"{base}-{name}", pos=cen + [0,0,0.3],
                   color=col, font_size=10, bold=True, parent=comp_view.scene)
        collections['labels'].append(lbl)

        mark = Markers(pos=np.vstack([composite_s1, composite_s2]),
                       face_color=rgba, edge_color='white', size=8, parent=comp_view.scene)
        collections['rungs'].append(mark)

    # Camera rotation
    comp_view.camera.azimuth = frame*0.2
    comp_view.camera.elevation = 20 + 5*np.sin(frame*0.005)
    comp_canvas.update()

timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == "__main__":
    comp_canvas.show()
    app.run()
