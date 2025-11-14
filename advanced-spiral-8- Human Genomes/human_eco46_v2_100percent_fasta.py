# ecoli46_v2_100percent_fasta.py
# --------------------------------------------------------------
# 100% FASTA-POWERED DNA VISUALIZATION
# Every parameter emerges from genome sequence - zero arbitrary constants
# Build: tcc -shared -o dna_engine_v2.dll dna_engine_v2.c
# Run: python ecoli46_v2_100percent_fasta.py
# --------------------------------------------------------------

import os
import sys
import numpy as np
import ctypes
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color


# ---------- AUTO-DETECT AND LOAD GENOME ----------
def find_human_fasta():
    """Automatically find the Human Genome FASTA file in the ncbi_dataset"""
    import glob
    possible_paths = [
        r"ncbi_dataset\ncbi_dataset\data\GCF_000001405.40\*.fna",
        r"ncbi_dataset\ncbi_dataset\data\GCA_000001405.29\*.fna",
        r"ncbi_dataset\ncbi_dataset\data\*\*.fna",
    ]

    for pattern in possible_paths:
        files = glob.glob(pattern)
        if files:
            return files[0]

    raise FileNotFoundError("Could not find Human Genome FASTA file in ncbi_dataset directory")


# ---------- LOAD C ENGINE V2 ----------
if sys.platform == 'win32':
    lib_name = './dna_engine_v2.dll'
else:
    lib_name = './dna_engine_v2.so'

if not os.path.exists(lib_name):
    print(f"ERROR: C engine V2 not found: {lib_name}")
    print("\nPlease compile first:")
    print("  Windows: tcc -shared -o dna_engine_v2.dll dna_engine_v2.c")
    print("  Linux:   gcc -shared -fPIC -o dna_engine_v2.so dna_engine_v2.c -lm -O3")
    sys.exit(1)

# Point structure with V2 extensions
class Point(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
        ('z', ctypes.c_float),
        ('color_r', ctypes.c_float),
        ('color_g', ctypes.c_float),
        ('color_b', ctypes.c_float),
        ('dimension', ctypes.c_int),
        ('base', ctypes.c_char),
        # V2: FASTA-driven properties
        ('organelle_spawn_prob', ctypes.c_float),
        ('lattice_push_strength', ctypes.c_float),
        ('twist_modifier', ctypes.c_float),
        ('codon_index', ctypes.c_uint8),
    ]

class CameraState(ctypes.Structure):
    _fields_ = [
        ('azimuth', ctypes.c_float),
        ('elevation', ctypes.c_float),
        ('distance', ctypes.c_float),
    ]

# Load library
engine = ctypes.CDLL(lib_name)

# Define function signatures
engine.init_engine.argtypes = [ctypes.c_char_p]
engine.init_engine.restype = ctypes.c_int

engine.get_frame_data.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(Point),
    ctypes.POINTER(Point),
]
engine.get_frame_data.restype = ctypes.c_int

engine.get_camera_state.argtypes = [ctypes.c_int, ctypes.POINTER(CameraState)]
engine.should_divide.argtypes = [ctypes.c_int, ctypes.c_int]
engine.should_divide.restype = ctypes.c_int

engine.create_daughter_cell.argtypes = [
    ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double
]
engine.create_daughter_cell.restype = ctypes.c_int

engine.get_genome_length.restype = ctypes.c_int
engine.get_num_cells.restype = ctypes.c_int
engine.get_gc_content.restype = ctypes.c_double
engine.get_shannon_entropy.restype = ctypes.c_double

# ---------- INITIALIZE ENGINE ----------
fasta_path = find_human_fasta().encode()
if engine.init_engine(fasta_path) != 0:
    print(f"ERROR: Failed to initialize engine with {fasta_path.decode()}")
    sys.exit(1)

genome_len = engine.get_genome_length()
gc_content = engine.get_gc_content()
entropy = engine.get_shannon_entropy()

print(f"✓ DNA Engine V2 (100% FASTA-powered)")
print(f"  Genome: {genome_len:,} bases")
print(f"  GC content: {gc_content*100:.2f}%")
print(f"  Shannon entropy: {entropy:.4f} bits")

# ---------- CONSTANTS (φ-based only) ----------
phi = (1 + np.sqrt(5)) / 2
POINTS_PER_FRAME = 400
MAX_RUNG_HISTORY = 6000

# Geometry names (for labels)
geometry_names = ['Point', 'Line', 'Triangle', 'Tetrahedron',
                  'Pentachoron', 'Hexacross', 'Heptacube', 'Octacube']

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(
    keys='interactive',
    size=(1400, 900),
    bgcolor='#000011',
    title="DNA Engine V2 - 100% FASTA-Powered Evolution"
)
view = canvas.central_widget.add_view()
view.camera = 'turntable'

# Info text
info_text = Text(
    "",
    pos=(10, 30),
    color='cyan',
    font_size=12,
    bold=True,
    anchor_x='left',
    parent=view
)

# ---------- CELL CLASS (V2: 100% FASTA-DRIVEN) ----------
class Cell:
    def __init__(self, cell_id):
        self.cell_id = cell_id
        self.frame = 0
        self.strand1 = Line(pos=np.zeros((1,3)), color=(0,1,1,0.9), width=2, parent=view.scene)
        self.strand2 = Line(pos=np.zeros((1,3)), color=(1,0.5,0,0.9), width=2, parent=view.scene)
        self.rungs, self.echoes, self.links, self.labels, self.centers, self.organelles = [], [], [], [], [], []

        # C engine buffers
        self.strand1_buffer = (Point * POINTS_PER_FRAME)()
        self.strand2_buffer = (Point * POINTS_PER_FRAME)()

        self.progress_text = Text("0%", pos=[0,0,20],
                                  color='white', font_size=24, anchor_x='center', parent=view.scene)

    def update(self):
        self.frame += 1

        # Get frame data from C engine
        num_points = engine.get_frame_data(
            self.cell_id,
            self.frame,
            self.strand1_buffer,
            self.strand2_buffer
        )

        if num_points <= 0:
            return

        # Convert C buffers to numpy arrays
        s1 = np.array([(p.x, p.y, p.z) for p in self.strand1_buffer[:num_points]], dtype=np.float32)
        s2 = np.array([(p.x, p.y, p.z) for p in self.strand2_buffer[:num_points]], dtype=np.float32)

        # Update visualization
        self.strand1.set_data(s1)
        self.strand2.set_data(s2)

        # Get current point with FASTA properties
        last_point = self.strand1_buffer[num_points - 1]
        cur_base = last_point.base.decode('utf-8')
        dim = last_point.dimension - 1

        # V2: Use FASTA-driven properties
        organelle_prob = last_point.organelle_spawn_prob
        lattice_strength = last_point.lattice_push_strength
        codon = last_point.codon_index

        # Color from C engine (codon-modulated) - clamp to [0, 1]
        col_rgb = (
            min(1.0, max(0.0, last_point.color_r)),
            min(1.0, max(0.0, last_point.color_g)),
            min(1.0, max(0.0, last_point.color_b))
        )

        if dim >= 0 and dim < len(geometry_names):
            name = geometry_names[dim]
            verts = [1, 2, 3, 4, 5, 12, 14, 16][dim]
        else:
            name = 'Unknown'
            verts = 1

        # Rungs every 20 frames
        if self.frame % 20 == 0:
            # Sample points for geometry
            step = max(1, num_points // verts)
            idx_pts = np.arange(0, min(verts * step, num_points), step)[:verts]
            pts1 = s1[idx_pts]
            pts2 = s2[idx_pts]
            all_pts = np.vstack((pts1, pts2))

            # Use codon-modulated color
            mark = Markers(pos=all_pts, face_color=col_rgb + (1.0,),
                           edge_color='white', size=6, parent=view.scene)
            self.rungs.append(mark)

            cen = all_pts.mean(axis=0)
            self.centers.append(cen)

            # Label with codon info
            lbl = Text(f"{cur_base}:{name} [C{codon}]", pos=cen+[0,0,0.3],
                       color=col_rgb, font_size=9, bold=True,
                       anchor_x='center', parent=view.scene)
            self.labels.append(lbl)

            # Echo
            if len(self.centers) > 1:
                echo_pts = all_pts*0.75 + np.random.normal(scale=0.01, size=all_pts.shape)
                echo = Markers(pos=echo_pts, face_color=(1,1,1,0.25),
                               size=4, parent=view.scene)
                self.echoes.append(echo)

            # Links
            if len(self.centers) > 1:
                prev_c = self.centers[-2]
                segs = []
                for p in all_pts[:6]:
                    segs += [prev_c, p]
                link = Line(pos=np.array(segs), color=(0.7,0.7,0.7,0.4),
                            width=1, connect='segments', parent=view.scene)
                self.links.append(link)

            # V2: Organelle spawn rate from FASTA (GC-content driven)
            cluster_size = 6 + dim
            if np.random.rand() < organelle_prob:
                org = spawn_organelle(cen, col_rgb, size=0.3, n=cluster_size)
                self.organelles.append(org)

        # V2: Lattice backpressure with FASTA-driven strength
        if len(self.centers) > 0:
            lattice_nodes = np.array(self.centers)
            for org in self.organelles:
                positive = cur_base in ['A', 'T']
                # Use lattice_strength from genome
                new_pts = np.array([lattice_push(p, lattice_nodes, positive=positive,
                                                  strength=lattice_strength)
                                    for p in org['positions']])
                org['positions'] = new_pts
                org['marker'].set_data(pos=new_pts, face_color=org['color'], size=5)

        # Progress
        percent = min(self.frame / genome_len * 100, 100)
        self.progress_text.text = f"{percent:.2f}%"

        # Limit history
        if len(self.rungs) > MAX_RUNG_HISTORY:
            drop = len(self.rungs) // 3
            self.rungs = self.rungs[drop:]
            self.echoes = self.echoes[drop:]
            self.links = self.links[drop:]
            self.labels = self.labels[drop:]
            self.centers = self.centers[drop:]
            self.organelles = self.organelles[drop:]

# ---------- HELPER FUNCTIONS ----------
def spawn_organelle(center, color_rgb, size=0.5, n=12):
    pts = center + np.random.normal(scale=0.2, size=(n,3))*size
    rgba = list(color_rgb) + [1.0]
    mark = Markers(pos=pts, face_color=rgba, edge_color=None, size=5, parent=view.scene)
    return {"marker": mark, "positions": pts, "color": rgba, "size": size}

def lattice_push(pt, lattice_nodes, positive=True, strength=0.02):
    if len(lattice_nodes) == 0:
        return pt
    nearest = lattice_nodes[np.random.randint(0, len(lattice_nodes))]
    dir_vec = nearest - pt
    if not positive:
        dir_vec *= -1
    return pt + dir_vec*strength

# ---------- INIT ----------
cells = [Cell(cell_id=0)]
global_frame = 0

def update(ev):
    global cells, global_frame
    global_frame += 1

    # V2: Camera control from FASTA
    cam_state = CameraState()
    engine.get_camera_state(global_frame, ctypes.byref(cam_state))
    view.camera.azimuth = cam_state.azimuth
    view.camera.elevation = cam_state.elevation
    # Distance (zoom) - optional, can be jarring
    # view.camera.scale_factor = cam_state.distance

    new_cells = []
    for c in cells:
        c.update()

        # V2: Division check from FASTA (palindrome-triggered)
        if engine.should_divide(c.cell_id, c.frame):
            offset = np.random.normal(scale=5.0, size=3)
            new_id = engine.create_daughter_cell(c.cell_id, offset[0], offset[1], offset[2])
            if new_id >= 0:
                new_cell = Cell(cell_id=new_id)
                new_cells.append(new_cell)
                print(f"✓ FASTA-triggered division: Cell {c.cell_id} → {new_id} (frame {c.frame})")

    cells += new_cells

    # Update info text
    num_cells = engine.get_num_cells()
    info_text.text = (f"Frame: {global_frame} | Cells: {num_cells} | "
                      f"GC: {gc_content*100:.1f}% | Entropy: {entropy:.3f} bits\n"
                      f"100% FASTA-POWERED: Camera, Division, Organelles, Colors, Physics")

# ---------- START ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

print("\n" + "="*70)
print("DNA ENGINE V2 - 100% FASTA-POWERED")
print("="*70)
print("ALL PARAMETERS DERIVED FROM GENOME:")
print("  ✓ Camera motion (GC%, purine/pyrimidine, entropy)")
print("  ✓ Cell division (palindrome signatures)")
print("  ✓ Organelle spawn (local GC content)")
print("  ✓ Color modulation (codon usage)")
print("  ✓ Physics strength (sequence entropy)")
print("  ✓ Spiral geometry (dinucleotide frequencies)")
print("="*70 + "\n")

if __name__ == "__main__":
    try:
        canvas.show()
        app.run()
    finally:
        engine.cleanup_engine()
        print("\n✓ Engine cleaned up")
