"""Quick debug test for V3 engine"""
import ctypes
import numpy as np
from pathlib import Path

class Point(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
        ('z', ctypes.c_float),
        ('color_h', ctypes.c_float),
        ('color_s', ctypes.c_float),
        ('color_v', ctypes.c_float),
        ('dimension', ctypes.c_int),
        ('base', ctypes.c_char),
        ('kmer_index', ctypes.c_uint8),
        ('genome_property', ctypes.c_float * 4),
    ]

lib_path = Path(__file__).parent / "dna_engine_v3_pure_fasta.dll"
print(f"Loading: {lib_path}")
print(f"Exists: {lib_path.exists()}")

engine = ctypes.CDLL(str(lib_path))

engine.init_engine.argtypes = [ctypes.c_char_p]
engine.init_engine.restype = ctypes.c_int

engine.get_frame_data.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(Point), ctypes.POINTER(Point)]
engine.get_frame_data.restype = ctypes.c_int

engine.get_genome_length.restype = ctypes.c_int
engine.get_points_per_frame.restype = ctypes.c_int
engine.get_max_cells.restype = ctypes.c_int
engine.get_core_radius.restype = ctypes.c_double

fasta_path = Path(__file__).parent / "ecoli_k12.fasta"
print(f"\nInitializing with: {fasta_path}")

result = engine.init_engine(str(fasta_path).encode('utf-8'))
print(f"Init result: {result}")

if result == 0:
    print(f"Genome length: {engine.get_genome_length()}")
    print(f"Points/frame: {engine.get_points_per_frame()}")
    print(f"Max cells: {engine.get_max_cells()}")
    print(f"Core radius: {engine.get_core_radius():.2f}")

    # Test getting frame data
    points_per_frame = engine.get_points_per_frame()
    strand1 = (Point * points_per_frame)()
    strand2 = (Point * points_per_frame)()

    print(f"\nTesting frame 0...")
    n = engine.get_frame_data(0, 0, strand1, strand2)
    print(f"Frame data returned: {n} points")

    if n > 0:
        print(f"\nFirst 3 points of strand1:")
        for i in range(min(3, n)):
            p = strand1[i]
            print(f"  [{i}] pos=({p.x:.2f}, {p.y:.2f}, {p.z:.2f}) color_h={p.color_h:.1f} dim={p.dimension} base={p.base.decode()}")

        # Check ranges
        positions = np.array([[p.x, p.y, p.z] for p in strand1[:n]])
        print(f"\nPosition ranges:")
        print(f"  X: [{positions[:,0].min():.2f}, {positions[:,0].max():.2f}]")
        print(f"  Y: [{positions[:,1].min():.2f}, {positions[:,1].max():.2f}]")
        print(f"  Z: [{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")

        colors = np.array([[p.color_h, p.color_s, p.color_v] for p in strand1[:n]])
        print(f"\nColor ranges:")
        print(f"  H: [{colors[:,0].min():.1f}, {colors[:,0].max():.1f}]")
        print(f"  S: [{colors[:,1].min():.2f}, {colors[:,1].max():.2f}]")
        print(f"  V: [{colors[:,2].min():.2f}, {colors[:,2].max():.2f}]")
    else:
        print("ERROR: No frame data returned!")
else:
    print("ERROR: Init failed!")
