import ctypes

class Point(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_float), ('y', ctypes.c_float), ('z', ctypes.c_float),
        ('color_r', ctypes.c_float), ('color_g', ctypes.c_float), ('color_b', ctypes.c_float),
        ('dimension', ctypes.c_int), ('base', ctypes.c_char),
        ('organelle_spawn_prob', ctypes.c_float), ('lattice_push_strength', ctypes.c_float),
        ('twist_modifier', ctypes.c_float), ('codon_index', ctypes.c_uint8)
    ]

engine = ctypes.CDLL('./dna_engine_v2.dll')
engine.init_engine.argtypes = [ctypes.c_char_p]
engine.init_engine.restype = ctypes.c_int
engine.get_frame_data.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(Point), ctypes.POINTER(Point)]
engine.get_frame_data.restype = ctypes.c_int

engine.init_engine(b'ecoli_k12.fasta')
s1 = (Point * 400)()
s2 = (Point * 400)()
engine.get_frame_data(0, 0, s1, s2)

colors = [(p.color_r, p.color_g, p.color_b) for p in s1]
print(f'Color range check (should be 0-1):')
print(f'  R: [{min(c[0] for c in colors):.3f}, {max(c[0] for c in colors):.3f}]')
print(f'  G: [{min(c[1] for c in colors):.3f}, {max(c[1] for c in colors):.3f}]')
print(f'  B: [{min(c[2] for c in colors):.3f}, {max(c[2] for c in colors):.3f}]')

all_valid = all(0 <= c[i] <= 1.0 for c in colors for i in range(3))
print(f'\n{"✓" if all_valid else "✗"} All colors in valid range [0, 1]')
engine.cleanup_engine()
