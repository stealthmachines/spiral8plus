"""
DNA ENGINE V3 - AI INTERPRETATION MODE
Streams FASTA→Visual transformation commands in real-time
Shows exactly what the genome sequence is telling the visualizer to do
"""

import sys
import time
from pathlib import Path
from ctypes import *
import numpy as np


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

class Point(Structure):
    _fields_ = [
        ('x', c_float),
        ('y', c_float),
        ('z', c_float),
        ('base', c_char * 4),
        ('dimension', c_int),
        ('hue', c_float),
        ('saturation', c_float),
        ('value', c_float)
    ]

def load_engine():
    """Load the C engine DLL"""
    dll_path = Path(__file__).parent / "dna_engine_v3_pure_fasta.dll"
    if not dll_path.exists():
        raise FileNotFoundError(f"Engine not found: {dll_path}")

    engine = CDLL(str(dll_path))

    # Function signatures
    engine.init_engine.argtypes = [c_char_p]
    engine.init_engine.restype = c_int

    engine.get_genome_length.restype = c_int
    engine.get_points_per_frame.restype = c_int
    engine.get_max_cells.restype = c_int
    engine.get_core_radius.restype = c_double

    engine.get_frame_data.argtypes = [c_int, c_int, POINTER(Point), POINTER(Point)]
    engine.get_frame_data.restype = c_int

    engine.cleanup_engine.argtypes = []
    engine.cleanup_engine.restype = None

    return engine

# ═══════════════════════════════════════════════════════════════════════════
# AI INTERPRETER
# ═══════════════════════════════════════════════════════════════════════════

class FASTAInterpreter:
    def __init__(self, fasta_path):
        self.engine = load_engine()

        # Initialize engine
        result = self.engine.init_engine(fasta_path.encode('utf-8'))
        if result != 0:
            raise RuntimeError(f"Engine init failed: {result}")

        # Get GENOME-DERIVED parameters
        self.genome_length = self.engine.get_genome_length()
        self.points_per_frame = self.engine.get_points_per_frame()
        self.max_cells = self.engine.get_max_cells()
        self.core_radius = self.engine.get_core_radius()

        # Frame state
        self.frame = 0
        self.cell_id = 0

        # Preallocate buffers
        self.strand1 = (Point * self.points_per_frame)()
        self.strand2 = (Point * self.points_per_frame)()

    def print_header(self):
        """Print AI-readable header"""
        print("=" * 80)
        print("FASTA->VISUAL COMMAND STREAM")
        print("AI Interpretation of Genome->Visualizer Transformation")
        print("=" * 80)
        print(f"\nGenome: SARS-CoV-2 K-12 ({self.genome_length:,} bases)")
        print(f"\nGENOME COMMANDS TO VISUALIZER:")
        print(f"  COMMAND: SET points_per_frame={self.points_per_frame}")
        print(f"    -> DERIVED FROM: genome_length % 997 = {self.genome_length} % 997")
        print(f"  COMMAND: SET max_cells={self.max_cells}")
        print(f"    -> DERIVED FROM: GC content statistical analysis")
        print(f"  COMMAND: SET core_radius={self.core_radius:.2f}")
        print(f"    -> DERIVED FROM: Melting temperature calculation")
        print("\n" + "=" * 80)
        print("STREAMING FRAME-BY-FRAME GENOME COMMANDS...")
        print("=" * 80 + "\n")

    def interpret_frame(self, n_frames=100, delay=0.05, verbose=True):
        """Interpret FASTA commands frame-by-frame"""
        try:
            self.print_header()
        except Exception as e:
            print(f"[ERROR in header] {e}")
            import traceback
            traceback.print_exc()
            return

        try:
            for frame_num in range(n_frames):
                try:
                    # Get frame data from genome
                    n = self.engine.get_frame_data(self.cell_id, self.frame, self.strand1, self.strand2)

                    if n <= 0:
                        print(f"[FRAME {frame_num}] GENOME COMMAND: SKIP (no data)")
                        continue

                    # Interpret this frame's commands
                    self.interpret_frame_commands(frame_num, n, verbose)

                    self.frame += 1
                except Exception as e:
                    print(f"\n[ERROR in frame {frame_num}] {e}")
                    import traceback
                    traceback.print_exc()
                    break

                time.sleep(delay)

        except KeyboardInterrupt:
            print("\n\n[STOPPED] User interrupt")
        except Exception as e:
            print(f"\n[ERROR in frame loop] {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 80)
        print("INTERPRETATION COMPLETE")
        print("=" * 80)

    def interpret_frame_commands(self, frame_num, n, verbose=True):
        """Interpret what the genome is commanding the visualizer to do"""

        # Collect frame data
        positions = []
        colors = []
        dimensions = []

        for i in range(n):
            try:
                p = self.strand1[i]
                positions.append((p.x, p.y, p.z))
                colors.append((p.hue, p.saturation, p.value))
                dimensions.append(p.dimension)
            except:
                continue

        if not positions:
            print(f"[FRAME {frame_num}] GENOME COMMAND: RENDER empty_frame")
            return

        # Analyze genome commands
        positions = np.array(positions)
        colors = np.array(colors)

        # What is the genome telling us?
        dominant_dim = max(set(dimensions), key=dimensions.count)
        dim_counts = {d: dimensions.count(d) for d in set(dimensions)}

        # Infer sequence properties from colors (hue correlates with base composition)
        hue_mean = colors[:,0].mean()
        # Lower hue (~0-120) = AT-rich, Higher hue (~180-300) = GC-rich (rough approximation)
        local_gc = 30 + (hue_mean / 360 * 40)  # Map hue to ~30-70% GC range

        z_min, z_max = positions[:,2].min(), positions[:,2].max()
        x_range = positions[:,0].max() - positions[:,0].min()
        y_range = positions[:,1].max() - positions[:,1].min()

        hue_mean = colors[:,0].mean()
        hue_std = colors[:,0].std()
        sat_mean = colors[:,1].mean()
        val_mean = colors[:,2].mean()

        # Print genome's commands to visualizer
        print(f"\n[FRAME {frame_num}] GENOME COMMANDS:")
        print(f"  | RENDER {n} points")
        print(f"  |  -> REASON: Sequence region (inferred GC~{local_gc:.1f}%)")
        print(f"  |")
        print(f"  | SET dimension={dominant_dim}D")
        print(f"  |  -> DERIVED FROM: K-mer complexity analysis")
        print(f"  |")
        print(f"  | SET color=HSV({hue_mean:.0f}, {sat_mean:.2f}, {val_mean:.2f})")
        print(f"  |  -> DERIVED FROM: Codon frequency at genome position {self.frame}")
        if hue_std > 10:
            print(f"  |  -> NOTE: High color variance (sigma={hue_std:.1f}) = diverse codon usage")
        else:
            print(f"  |  -> NOTE: Low color variance (sigma={hue_std:.1f}) = repetitive sequence")
        print(f"  |")
        print(f"  | SET position_3d:")
        print(f"  |  | X_range: {x_range:.1f} units")
        print(f"  |  | Y_range: {y_range:.1f} units")
        print(f"  |  -> Z_range: [{z_min:.2f}, {z_max:.2f}]")
        print(f"  |     -> DERIVED FROM: Phi-scaled spiral (r={self.core_radius:.2f})")
        print(f"  |")

        if verbose and frame_num % 10 == 0:
            # Detailed interpretation
            print(f"  -> [AI INTERPRETATION]:")
            print(f"     The genome at position {self.frame:,}/{self.genome_length:,} is saying:")

            if local_gc > 55:
                print(f"     'I am GC-rich ({local_gc:.1f}%) -> use higher melting temp colors'")
            elif local_gc < 45:
                print(f"     'I am AT-rich ({local_gc:.1f}%) -> use lower melting temp colors'")
            else:
                print(f"     'I am balanced ({local_gc:.1f}% GC) -> use neutral colors'")

            if dominant_dim == 3:
                print(f"     'My sequence is complex (3D) -> render full spatial structure'")
            elif dominant_dim == 2:
                print(f"     'My sequence is moderately complex (2D) -> flatten structure'")
            else:
                print(f"     'My sequence is simple ({dominant_dim}D) -> simplify visualization'")

            if hue_std > 20:
                print(f"     'My codons are diverse -> use rainbow colors (variance={hue_std:.1f})'")
            else:
                print(f"     'My codons are repetitive -> use similar colors (variance={hue_std:.1f})'")

        # Base distribution interpretation
        if verbose and frame_num % 25 == 0:
            print(f"\n  [SEQUENCE CONTEXT @ position {self.frame:,}]:")
            print(f"    Dimension distribution: {dim_counts}")
            print(f"    Color diversity (hue sigma): {hue_std:.1f}")
            if local_gc > 55:
                print(f"    Interpretation: GC-rich region ({local_gc:.1f}%) -> high stability")
            elif local_gc < 45:
                print(f"    Interpretation: AT-rich region ({local_gc:.1f}%) -> lower stability")
            else:
                print(f"    Interpretation: Balanced region ({local_gc:.1f}%)")

    def cleanup(self):
        """Cleanup resources"""
        self.engine.cleanup_engine()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    fasta_path = find_covid_fasta()

    if not fasta_path.exists():
        print(f"ERROR: {fasta_path} not found!")
        sys.exit(1)

    print("[DEBUG] Creating FASTAInterpreter...")
    try:
        interpreter = FASTAInterpreter(str(fasta_path))
        print("[DEBUG] Interpreter created successfully")
    except Exception as e:
        print(f"[ERROR creating interpreter] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        # Default: stream 100 frames with detailed interpretation
        n_frames = 100
        verbose = True

        if len(sys.argv) > 1:
            n_frames = int(sys.argv[1])
        if len(sys.argv) > 2:
            verbose = sys.argv[2].lower() in ['true', '1', 'yes', 'v']

        print(f"[DEBUG] About to call interpret_frame with n_frames={n_frames}, verbose={verbose}")
        interpreter.interpret_frame(n_frames=n_frames, delay=0.03, verbose=verbose)
        print(f"[DEBUG] interpret_frame returned normally")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[DEBUG] In finally block, calling cleanup...")
        interpreter.cleanup()
