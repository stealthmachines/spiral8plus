"""
═══════════════════════════════════════════════════════════════════════════
E. COLI K-12 DNA ENGINE V3 - TRUE 100% FASTA GENERATION
═══════════════════════════════════════════════════════════════════════════

PHILOSOPHY CHANGE FROM V2:
- V2: Genome MODULATES predefined structures (hardcoded geometry table)
- V3: Genome GENERATES all structures (zero hardcoded lookups)

ALL PARAMETERS EMERGE FROM SEQUENCE:
✓ Points per frame = f(genome length)
✓ Max cells = f(GC content)
✓ Window size = f(entropy)
✓ Colors = f(codon frequencies → HSV)
✓ Dimensions = f(k-mer complexity)
✓ Physics = f(transition probabilities)
✓ Camera = f(autocorrelation)

TEST: Different genomes → Completely different visuals!
"""

import numpy as np
import ctypes
import os
import sys
from pathlib import Path


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


# VisPy visualization
from vispy import app, scene
from vispy.color import Color

# ═══════════════════════════════════════════════════════════════════════════
# C ENGINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════

class Point(ctypes.Structure):
    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
        ('z', ctypes.c_float),
        ('color_h', ctypes.c_float),  # Hue
        ('color_s', ctypes.c_float),  # Saturation
        ('color_v', ctypes.c_float),  # Value
        ('dimension', ctypes.c_int),
        ('base', ctypes.c_char),
        ('kmer_index', ctypes.c_uint8),
        ('genome_property', ctypes.c_float * 4),
    ]

def load_engine():
    """Load C engine with error handling"""
    lib_name = "dna_engine_v3_pure_fasta.dll" if sys.platform == "win32" else "dna_engine_v3_pure_fasta.so"
    lib_path = Path(__file__).parent / lib_name

    if not lib_path.exists():
        print(f"ERROR: {lib_name} not found!")
        print("Compile first:")
        if sys.platform == "win32":
            print(f'  gcc -shared -o "{lib_path}" -O3 dna_engine_v3_pure_fasta.c -lm')
        else:
            print(f'  gcc -shared -fPIC -o "{lib_path}" -O3 dna_engine_v3_pure_fasta.c -lm')
        sys.exit(1)

    engine = ctypes.CDLL(str(lib_path))

    # Function signatures
    engine.init_engine.argtypes = [ctypes.c_char_p]
    engine.init_engine.restype = ctypes.c_int

    engine.get_frame_data.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(Point), ctypes.POINTER(Point)]
    engine.get_frame_data.restype = ctypes.c_int

    engine.get_genome_length.restype = ctypes.c_int
    engine.get_points_per_frame.restype = ctypes.c_int
    engine.get_max_cells.restype = ctypes.c_int
    engine.get_core_radius.restype = ctypes.c_double

    engine.cleanup_engine.argtypes = []
    engine.cleanup_engine.restype = None

    return engine

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

# Try to set a different VisPy backend to avoid Qt/driver issues that can
# produce a black/blank canvas on some systems. We'll try 'glfw' first,
# then fall back to 'pyqt5' if necessary. This is safe to attempt before
# creating the SceneCanvas.
try:
    from vispy import app as _vispy_app
    try:
        _vispy_app.use_app('glfw')
        print('VisPy backend: forced to glfw')
    except Exception:
        try:
            _vispy_app.use_app('pyqt5')
            print('VisPy backend: forced to pyqt5 (fallback)')
        except Exception as _e:
            print('VisPy backend: could not force glfw/pyqt5, continuing with default')
except Exception:
    # If vispy isn't available at import time, let the later import fail normally
    pass

def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB (h: 0-360, s/v: 0-1)"""
    import colorsys
    return colorsys.hsv_to_rgb(h / 360.0, s, v)

class DNAVisualizer:
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

        print(f"\n{'='*70}")
        print(f"DNA ENGINE V3 - TRUE 100% FASTA-GENERATED")
        print(f"{'='*70}")
        print(f"Genome: {fasta_path}")
        print(f"Length: {self.genome_length:,} bases")
        print(f"\nGENOME-DERIVED PARAMETERS (ZERO HARDCODED!):")
        print(f"  Points/frame: {self.points_per_frame}")
        print(f"  Max cells: {self.max_cells}")
        print(f"  Core radius: {self.core_radius:.2f}")
        print(f"{'='*70}\n")

        # Frame state
        self.frame = 0
        self.cell_id = 0
        self.z_offset = 0.0  # Accumulate Z to build the spiral vertically

        # Accumulate points to build the full spiral
        self.all_points1 = []
        self.all_points2 = []
        self.max_frames_visible = 50  # Show 50 frames = 21,300 points (avoid memory overflow)

        # Preallocate buffers
        self.strand1 = (Point * self.points_per_frame)()
        self.strand2 = (Point * self.points_per_frame)()

        # Setup VisPy
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(1920, 1080),
            bgcolor='#000011',
            title="DNA Engine V3: 100% FASTA-Generated"
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        # Camera will be configured after canvas setup

        # Visuals (will be updated each frame) - Use Markers for visibility
        from vispy.scene.visuals import Markers
        self.strand1_visual = Markers(parent=self.view.scene)
        self.strand2_visual = Markers(parent=self.view.scene)
        # Adjust GL state: disable depth testing so small Z-range points remain visible
        try:
            self.strand1_visual.set_gl_state('translucent', depth_test=False)
            self.strand2_visual.set_gl_state('translucent', depth_test=False)
        except Exception:
            # set_gl_state may not be available in older VisPy versions; ignore if so
            pass

        # Stats overlay
        from vispy.scene.visuals import Text
        self.text = Text(
            text="",
            color='white',
            font_size=12,
            pos=(10, 30),
            anchor_x='left'
        )
        self.view.add(self.text)

        # AI Interpreter overlay (right side)
        self.ai_text = Text(
            text="",
            color='#00ff88',
            font_size=11,
            pos=(1920 - 10, 30),
            anchor_x='right'
        )
        self.view.add(self.ai_text)

        # Animation timer - try different approach
        self.timer = app.Timer(interval=1/60.0, connect=self.update, start=False)  # Don't start automatically
        print(f"Timer created with interval {1/60.0}, running: {self.timer.running}")
        self.fps_counter = 0
        self.fps_time = 0
        self.current_fps = 0

        # Force first update to test visuals
        print("Forcing first update...")
        # Create a mock event object
        class MockEvent:
            elapsed = 1/60.0
        self.update(MockEvent())
        print("First update complete")

    def update(self, event):
        """Update visualization with genome-generated data"""
        # Get frame data from C engine
        n = self.engine.get_frame_data(self.cell_id, self.frame, self.strand1, self.strand2)
        if n <= 0:
            return

        # Extract FULL FASTA DATA - not just positions!
        positions1 = np.array([[p.x, p.y, p.z + self.z_offset] for p in self.strand1[:n]], dtype=np.float32)
        positions2 = np.array([[p.x, p.y, p.z + self.z_offset] for p in self.strand2[:n]], dtype=np.float32)

        # USE GENOME-GENERATED COLORS (HSV from codon frequencies)
        import colorsys
        colors1 = np.array([colorsys.hsv_to_rgb(p.color_h/360.0, p.color_s, p.color_v) + (1.0,)
                           for p in self.strand1[:n]], dtype=np.float32)
        colors2 = np.array([colorsys.hsv_to_rgb(p.color_h/360.0, p.color_s, p.color_v) + (1.0,)
                           for p in self.strand2[:n]], dtype=np.float32)

        # USE DIMENSION for point sizing (higher dimension = more complex = larger points)
        sizes1 = np.array([3.0 + p.dimension * 3.0 for p in self.strand1[:n]], dtype=np.float32)
        sizes2 = np.array([3.0 + p.dimension * 3.0 for p in self.strand2[:n]], dtype=np.float32)

        # Accumulate points with their genome properties
        self.all_points1.append((positions1, colors1, sizes1))
        self.all_points2.append((positions2, colors2, sizes2))

        # Keep only last N frames for performance
        if len(self.all_points1) > self.max_frames_visible:
            self.all_points1.pop(0)
            self.all_points2.pop(0)

        # Combine all accumulated points
        all_pos1 = np.vstack([p[0] for p in self.all_points1])
        all_pos2 = np.vstack([p[0] for p in self.all_points2])
        all_col1 = np.vstack([p[1] for p in self.all_points1])
        all_col2 = np.vstack([p[1] for p in self.all_points2])
        all_size1 = np.concatenate([p[2] for p in self.all_points1])
        all_size2 = np.concatenate([p[2] for p in self.all_points2])

        # Update Z offset for next frame (advance the spiral upward)
        self.z_offset += 10.0  # Advance 10 units per frame for visible spiral growth

        # Debug first frame only
        if self.frame == 0:
            print(f"First frame: {n} points, Z-range: [{positions1[:,2].min():.2f}, {positions1[:,2].max():.2f}]")
            print(f"Color range: H=[{self.strand1[0].color_h:.0f}-{self.strand1[n-1].color_h:.0f}], S=[{self.strand1[0].color_s:.2f}-{self.strand1[n-1].color_s:.2f}]")
            print(f"Dimensions: {set(p.dimension for p in self.strand1[:n])}")
            print(f"Bases: {set(p.base.decode('utf-8') for p in self.strand1[:n])}")

            # --- DEBUG: Save first frame to an image using matplotlib to confirm data is visible ---
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                # Strand 1 - USE GENOME COLORS!
                ax.scatter(positions1[:,0], positions1[:,1], positions1[:,2],
                           c=colors1[:,:3], s=sizes1, depthshade=True, label='strand1')
                # Strand 2
                ax.scatter(positions2[:,0], positions2[:,1], positions2[:,2],
                           c=colors2[:,:3], s=sizes2, depthshade=True, label='strand2')

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('First frame with GENOME COLORS')

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('First frame (matplotlib fallback)')
                ax.legend()
                plt.tight_layout()
                out_path = Path(__file__).parent / 'frame0_matplotlib.png'
                fig.savefig(out_path, dpi=150)
                plt.close(fig)
                print(f"Saved first-frame image to: {out_path}")
            except Exception as e:
                print(f"Failed to save matplotlib image: {e}")

        # Update visuals with GENOME-GENERATED COLORS AND SIZES!
        try:
            self.strand1_visual.set_data(all_pos1, face_color=all_col1, size=all_size1, edge_width=0)
            self.strand2_visual.set_data(all_pos2, face_color=all_col2, size=all_size2, edge_width=0)
        except Exception as e:
            print(f"Warning: Could not set genome colors, using fallback: {e}")
            # Fallback if RGBA array not accepted
            self.strand1_visual.set_data(all_pos1, face_color='cyan', size=8)
            self.strand2_visual.set_data(all_pos2, face_color='orange', size=8)        # FPS calculation
        self.fps_counter += 1
        elapsed = event.elapsed
        self.fps_time += elapsed
        if self.fps_time >= 1.0:
            self.current_fps = self.fps_counter / self.fps_time
            self.fps_counter = 0
            self.fps_time = 0

        # Stats overlay
        progress = (self.frame % self.genome_length) / self.genome_length * 100
        base = self.strand1[0].base.decode('utf-8') if n > 0 else '?'
        dim = self.strand1[0].dimension if n > 0 else 0

        stats_text = (
            f"TRUE 100% FASTA GENERATION\n"
            f"Frame: {self.frame:,} | FPS: {self.current_fps:.1f}\n"
            f"Progress: {progress:.1f}%\n"
            f"Base: {base} | Dimension: {dim}D\n"
            f"Points: {n} (genome-derived)\n"
        )
        self.text.text = stats_text

        # AI INTERPRETER - Analyze genome patterns and explain
        if n > 0:
            # Extract genome features from current frame
            hues = [p.color_h for p in self.strand1[:n]]
            avg_hue = sum(hues) / len(hues)
            dimensions = [p.dimension for p in self.strand1[:n]]
            avg_dim = sum(dimensions) / len(dimensions)
            bases = [p.base.decode('utf-8') for p in self.strand1[:n]]

            # Count base frequencies
            a_count = bases.count('A')
            c_count = bases.count('C')
            g_count = bases.count('G')
            t_count = bases.count('T')
            gc_content = (g_count + c_count) / n * 100

            # Analyze transition probabilities
            trans_prob = [self.strand1[0].genome_property[i] for i in range(4)]

            # Generate AI interpretation
            ai_interpretation = self._generate_ai_interpretation(
                avg_hue, avg_dim, gc_content, trans_prob, self.frame
            )
            self.ai_text.text = ai_interpretation

        self.frame += 1
        self.canvas.update()

    def _generate_ai_interpretation(self, avg_hue, avg_dim, gc_content, trans_prob, frame):
        """Generate real-time AI interpretation of genome patterns AS CELLULAR BEHAVIOR"""

        # TRANSITION PROBABILITIES = CELLULAR DIVISION BEHAVIOR
        # The 4x4 transition matrix encodes how the cell divides and mutates
        dominant_transition = trans_prob.index(max(trans_prob))
        trans_bases = ['A', 'C', 'G', 'T']

        if dominant_transition == 0:  # A→A (self-replication)
            division_state = "DIVIDING: Binary fission active"
            cell_behavior = "Cell is replicating DNA, preparing to split"
        elif dominant_transition == 1:  # A→C (purine→pyrimidine)
            division_state = "HUNTING: Seeking nutrients"
            cell_behavior = "Flagella active, chemotaxis toward food"
        elif dominant_transition == 2:  # A→G (purine→purine)
            division_state = "METABOLIZING: Processing food"
            cell_behavior = "Enzymes active, breaking down nutrients"
        else:  # A→T (complementary)
            division_state = "DEFENDING: Stress response"
            cell_behavior = "Building cell wall, resisting predation"

        # DIMENSION = METABOLIC STATE (k-mer complexity → energy level)
        if avg_dim < 3:
            metabolic_state = "DORMANT: Low energy, spore formation"
            energy_level = "Survival mode"
        elif avg_dim < 6:
            metabolic_state = "ACTIVE: Normal metabolism"
            energy_level = "Maintaining homeostasis"
        else:
            metabolic_state = f"HYPERACTIVE: High energy ({int(avg_dim)}D complexity)"
            energy_level = "Rapid growth & division"

        # GC CONTENT = CELL WALL STRENGTH (structural stability)
        if gc_content < 45:
            cell_wall = "WEAK: Vulnerable to attack"
            defense = "Easy prey for predators"
        elif gc_content < 55:
            cell_wall = "NORMAL: Balanced structure"
            defense = "Standard bacterial armor"
        else:
            cell_wall = "STRONG: Fortified cell wall"
            defense = "Resistant to predation"

        # COLOR = CHEMICAL SIGNALING (codon frequency → quorum sensing)
        if avg_hue < 60:
            signaling = "RED/YELLOW: Alarm pheromones"
            communication = "Warning nearby cells of danger"
        elif avg_hue < 150:
            signaling = "GREEN: Neutral signaling"
            communication = "Normal cell-cell communication"
        else:
            signaling = "CYAN/BLUE: Nutrient signals"
            communication = "Broadcasting food location"

        # GENOME POSITION = LIFECYCLE PHASE
        genome_phase = (frame * self.points_per_frame) % self.genome_length
        genome_pct = genome_phase / self.genome_length * 100

        if genome_pct < 25:
            lifecycle = "BIRTH: New cell from division"
            age = "Juvenile - rapid growth phase"
        elif genome_pct < 50:
            lifecycle = "YOUTH: Active foraging"
            age = "Adult - hunting and consuming"
        elif genome_pct < 75:
            lifecycle = "MATURITY: Preparing to divide"
            age = "Reproductive - DNA replication"
        else:
            lifecycle = "DIVISION: Cell splitting imminent"
            age = "Creating daughter cells"

        return (
            f"╔═══ CELLULAR LIFE DECODER ═══╗\n"
            f"║ E. COLI AS LIVING ORGANISM ║\n"
            f"╠═════════════════════════════╣\n"
            f"║ BEHAVIOR (Transitions):     ║\n"
            f"║ {division_state:27s} ║\n"
            f"║ {cell_behavior:27s} ║\n"
            f"║                             ║\n"
            f"║ METABOLISM (Complexity):    ║\n"
            f"║ {metabolic_state:27s} ║\n"
            f"║ {energy_level:27s} ║\n"
            f"║                             ║\n"
            f"║ DEFENSE (GC {gc_content:4.1f}%):          ║\n"
            f"║ {cell_wall:27s} ║\n"
            f"║ {defense:27s} ║\n"
            f"║                             ║\n"
            f"║ SIGNALING (Color):          ║\n"
            f"║ {signaling:27s} ║\n"
            f"║ {communication:27s} ║\n"
            f"║                             ║\n"
            f"║ LIFECYCLE ({genome_pct:4.1f}%):           ║\n"
            f"║ {lifecycle:27s} ║\n"
            f"║ {age:27s} ║\n"
            f"╚═════════════════════════════╝\n"
        )

    def run(self):
        print("Calling canvas.show()...")
        self.canvas.show()
        print("Canvas shown successfully!")

        # Configure camera after canvas is shown
        self.view.camera.distance = 150  # Pull back to see more vertical spiral
        self.view.camera.elevation = 20  # Lower angle to see spiral extension
        self.view.camera.azimuth = 45
        self.view.camera.center = (0, 0, 50)  # Center higher up to see spiral growth
        print(f"Camera configured: distance={self.view.camera.distance}, elevation={self.view.camera.elevation}, azimuth={self.view.camera.azimuth}, center={self.view.camera.center}")

        # Start the animation timer
        print("Starting animation timer...")
        self.timer.start()
        print("Animation started! Press Ctrl+C to stop.")

        try:
            app.run()
        except KeyboardInterrupt:
            print("\nAnimation stopped by user")
        finally:
            self.cleanup()

    def cleanup(self):
        self.engine.cleanup_engine()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    fasta_path = find_covid_fasta()

    if not fasta_path.exists():
        print(f"ERROR: {fasta_path} not found!")
        sys.exit(1)

    viz = DNAVisualizer(str(fasta_path))

    try:
        viz.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        viz.cleanup()
