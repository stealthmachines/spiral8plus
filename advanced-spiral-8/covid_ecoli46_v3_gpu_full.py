"""
═══════════════════════════════════════════════════════════════════════════
E. COLI K-12 DNA ENGINE V3 - GPU-ACCELERATED FULL GENOME RENDERING
═══════════════════════════════════════════════════════════════════════════

RENDERS ALL GENOME FRAMES AT ONCE AS GPU-ACCELERATED POINTS
Hardware acceleration enables visualization of the complete structure
"""

import numpy as np
import ctypes
import time
from pathlib import Path
from vispy import app, scene
from vispy.scene import visuals
import sys


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


# ═══════════════════════════════════════════════════════════════════════════
# C ENGINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

class Point(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
        ('z', ctypes.c_float),
        ('base', ctypes.c_char * 4),
        ('dimension', ctypes.c_int),
        ('hue', ctypes.c_float),
        ('saturation', ctypes.c_float),
        ('value', ctypes.c_float)
    ]

def load_engine():
    """Load the C engine DLL"""
    dll_path = Path(__file__).parent / "dna_engine_v3_pure_fasta.dll"
    if not dll_path.exists():
        raise FileNotFoundError(f"Engine not found: {dll_path}")

    engine = ctypes.CDLL(str(dll_path))

    # Function signatures
    engine.init_engine.argtypes = [ctypes.c_char_p]
    engine.init_engine.restype = ctypes.c_int

    engine.get_genome_length.restype = ctypes.c_int
    engine.get_points_per_frame.restype = ctypes.c_int
    engine.get_max_cells.restype = ctypes.c_int
    engine.get_core_radius.restype = ctypes.c_double

    engine.get_frame_data.argtypes = [ctypes.c_int, ctypes.c_int,
                                      ctypes.POINTER(Point), ctypes.POINTER(Point)]
    engine.get_frame_data.restype = ctypes.c_int

    engine.cleanup_engine.restype = None

    return engine

# ═══════════════════════════════════════════════════════════════════════════
# FULL GENOME VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

class FullGenomeVisualizer:
    def __init__(self, fasta_path):
        print("Initializing DNA Engine V3 (GPU Full Genome Mode)...")

        # Load C engine
        self.engine = load_engine()
        print(f"Calling init_engine with: {fasta_path}")
        result = self.engine.init_engine(str(fasta_path).encode('utf-8'))
        print(f"init_engine returned: {result}")
        if result != 0:
            raise RuntimeError(f"Engine initialization failed: {result}")

        # Get genome parameters
        self.genome_length = self.engine.get_genome_length()
        self.points_per_frame = self.engine.get_points_per_frame()
        self.max_cells = self.engine.get_max_cells()
        self.core_radius = self.engine.get_core_radius()

        print(f"\n{'='*70}")
        print(f"GENOME: SARS-CoV-2 K-12 ({self.genome_length:,} bases)")
        print(f"RENDERING MODE: Full genome as GPU point cloud")
        print(f"{'='*70}")
        print(f"Points/frame: {self.points_per_frame}")
        print(f"Max cells: {self.max_cells}")
        print(f"Core radius: {self.core_radius:.2f}")

        # Calculate total frames needed - USE FEWER FOR TESTING
        total_frames = min(self.max_cells, 10)  # Start with just 10 frames for debugging
        print(f"\nGenerating {total_frames:,} frames (testing)...")
        print(f"Total points: ~{total_frames * self.points_per_frame * 2:,}")  # *2 for both strands

        # Generate all genome data at once
        self.positions, self.colors = self.generate_full_genome(total_frames)

        print(f"\n✓ Generated {len(self.positions):,} points")
        print(f"  X range: [{self.positions[:,0].min():.1f}, {self.positions[:,0].max():.1f}]")
        print(f"  Y range: [{self.positions[:,1].min():.1f}, {self.positions[:,1].max():.1f}]")
        print(f"  Z range: [{self.positions[:,2].min():.1f}, {self.positions[:,2].max():.1f}]")

        # Setup VisPy canvas
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(1920, 1080),
            bgcolor='#000011',
            title=f"DNA Engine V3 GPU: {len(self.positions):,} genome points"
        )

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'

        # Create GPU-accelerated Markers visual
        print("\nUploading to GPU...")
        self.markers = scene.visuals.Markers()
        self.markers.set_data(self.positions, face_color=self.colors, size=5, edge_width=0)
        self.view.add(self.markers)

        # Simple camera setup
        print(f"✓ Uploaded {len(self.positions):,} points to GPU")
        print(f"\n{'='*70}")
        print("VISUALIZATION READY!")
        print("Controls: Mouse drag to rotate, scroll to zoom")
        print(f"{'='*70}\n")

        self.canvas.show()

    def generate_full_genome(self, total_frames):
        """Generate all genome frames at once"""
        strand1 = (Point * self.points_per_frame)()
        strand2 = (Point * self.points_per_frame)()

        all_positions = []
        all_colors = []

        cell_id = 0
        z_offset = 0.0

        print("Progress: ", end='', flush=True)
        for frame in range(total_frames):
            if frame % 10 == 0:
                print(f"{frame}/{total_frames} ", end='', flush=True)

            try:
                n = self.engine.get_frame_data(cell_id, frame, strand1, strand2)
                time.sleep(0.001)  # Small delay to avoid overwhelming the DLL
            except Exception as e:
                print(f"\n[ERROR at frame {frame}] {e}")
                break

            if n <= 0:
                continue

            # Extract positions and colors for both strands
            try:
                for i in range(n):
                    # Strand 1 (cyan)
                    p1 = strand1[i]
                    all_positions.append([p1.x, p1.y, p1.z + z_offset])
                    all_colors.append([0.0, 1.0, 1.0])  # Cyan

                    # Strand 2 (orange)
                    p2 = strand2[i]
                    all_positions.append([p2.x, p2.y, p2.z + z_offset])
                    all_colors.append([1.0, 0.55, 0.0])  # Orange
            except Exception as e:
                print(f"\n[ERROR extracting points at frame {frame}, point {i}] {e}")
                break

            # Advance spiral upward
            z_offset += 10.0

        print("Done!")
        print(f"Successfully generated {len(all_positions):,} points from {frame+1} frames")

        return np.array(all_positions, dtype=np.float32), np.array(all_colors, dtype=np.float32)

    def run(self):
        """Start the visualization"""
        app.run()

    def cleanup(self):
        """Cleanup resources"""
        self.engine.cleanup_engine()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Force GLFW backend
    app.use_app('glfw')
    print("VisPy backend: glfw (GPU-accelerated)")

    fasta_path = find_covid_fasta()

    if not fasta_path.exists():
        print(f"ERROR: {fasta_path} not found!")
        sys.exit(1)

    viz = FullGenomeVisualizer(fasta_path)

    try:
        viz.run()
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    finally:
        viz.cleanup()
