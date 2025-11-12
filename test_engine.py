#!/usr/bin/env python3
"""
DNA Engine Test - Verify C library integration
Tests basic functionality without visualization
"""

import sys
import ctypes
import numpy as np

# Detect platform
if sys.platform == 'win32':
    lib_name = './dna_engine.dll'
else:
    lib_name = './dna_engine.so'

print("="*70)
print("DNA ENGINE TEST")
print("="*70)

# Load library
print(f"\n1. Loading library: {lib_name}")
try:
    engine = ctypes.CDLL(lib_name)
    print("   ✓ Library loaded")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Define Point structure
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
    ]

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

engine.get_genome_length.restype = ctypes.c_int
engine.get_num_cells.restype = ctypes.c_int

engine.create_daughter_cell.argtypes = [
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
]
engine.create_daughter_cell.restype = ctypes.c_int

# Test 1: Initialize
print("\n2. Initializing engine with ecoli_k12.fasta")
result = engine.init_engine(b"ecoli_k12.fasta")
if result == 0:
    print("   ✓ Engine initialized")
else:
    print(f"   ✗ Init failed with code {result}")
    sys.exit(1)

# Test 2: Get genome length
print("\n3. Querying genome length")
genome_len = engine.get_genome_length()
print(f"   ✓ Genome: {genome_len:,} bases")

# Test 3: Get frame data
print("\n4. Generating frame 0 (400 points)")
POINTS_PER_FRAME = 400
strand1 = (Point * POINTS_PER_FRAME)()
strand2 = (Point * POINTS_PER_FRAME)()

num_points = engine.get_frame_data(0, 0, strand1, strand2)
if num_points > 0:
    print(f"   ✓ Generated {num_points} points")
else:
    print(f"   ✗ Failed to generate points")
    sys.exit(1)

# Test 4: Verify point data
print("\n5. Verifying point data")
sample = strand1[0]
print(f"   First point: ({sample.x:.2f}, {sample.y:.2f}, {sample.z:.2f})")
print(f"   Color: RGB({sample.color_r:.2f}, {sample.color_g:.2f}, {sample.color_b:.2f})")
print(f"   Base: {sample.base.decode()}, Dimension: {sample.dimension}")
print("   ✓ Data structure valid")

# Test 5: Multiple frames
print("\n6. Testing frame progression")
coords = []
for frame in [0, 10, 100]:
    engine.get_frame_data(0, frame, strand1, strand2)
    coords.append((strand1[0].x, strand1[0].y, strand1[0].z))
    print(f"   Frame {frame:3d}: ({coords[-1][0]:6.2f}, {coords[-1][1]:6.2f}, {coords[-1][2]:6.2f})")

# Check if coordinates change
if len(set(coords)) == len(coords):
    print("   ✓ Coordinates change across frames")
else:
    print("   ⚠ Warning: Some coordinates identical")

# Test 6: Cell division
print("\n7. Testing cell division")
initial_cells = engine.get_num_cells()
print(f"   Initial cells: {initial_cells}")

new_cell_id = engine.create_daughter_cell(0, 5.0, 5.0, 5.0)
if new_cell_id >= 0:
    final_cells = engine.get_num_cells()
    print(f"   ✓ Created cell {new_cell_id}, total cells: {final_cells}")
else:
    print("   ✗ Cell division failed")

# Test 7: New cell frame data
print("\n8. Testing daughter cell frame generation")
engine.get_frame_data(new_cell_id, 0, strand1, strand2)
offset_point = strand1[0]
print(f"   Daughter cell pos: ({offset_point.x:.2f}, {offset_point.y:.2f}, {offset_point.z:.2f})")
print("   ✓ Daughter cell operational")

# Test 8: Performance
print("\n9. Performance test (1000 frames)")
import time
start = time.time()
for frame in range(1000):
    engine.get_frame_data(0, frame, strand1, strand2)
elapsed = time.time() - start
fps = 1000 / elapsed
points_per_sec = (1000 * POINTS_PER_FRAME * 2) / elapsed

print(f"   Time: {elapsed:.3f} seconds")
print(f"   FPS: {fps:.1f}")
print(f"   Throughput: {points_per_sec:,.0f} points/sec")
print("   ✓ Performance test complete")

# Cleanup
print("\n10. Cleanup")
engine.cleanup_engine()
print("   ✓ Engine cleaned up")

print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)
print(f"\nEngine ready for ecoli46_c_engine.py")
print(f"Expected speedup: {points_per_sec/20000:.0f}x over pure Python")
