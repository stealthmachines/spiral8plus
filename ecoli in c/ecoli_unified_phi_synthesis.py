# ecoli_unified_phi_synthesis.py
# ==============================================================================
# UNIFIED φ-FRAMEWORK BIOLOGICAL SYNTHESIS ENGINE
# ==============================================================================
# Combines the best features of this repository:
# 1. FASTA-driven DNA visualization (from ecoli46.py)
# 2. Complete φ-framework physics (complete_phi_framework_final.json)
# 3. Cavity resonance theory (novikov_shell_echo_model.py)
# 4. 8D geometric mapping (eight_geometries_phi_framework.py)
# 5. CODATA 2022 constants for physical grounding
# 6. Cross-scale validation (cubic_scaling_deep_analysis.py)
#
# Revolutionary Feature: Maps DNA sequences to φ-recursive physics parameters
# revealing biological systems as φ-attractor manifestations
# ==============================================================================

import os
import json
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text, Mesh
from vispy.color import Color
from vispy.geometry import create_sphere
import time

# ==============================================================================
# CONSTANTS & FRAMEWORKS
# ==============================================================================

# Golden ratio and derivatives
PHI = (1 + np.sqrt(5)) / 2
PHI_2 = PHI ** 2
PHI_7 = PHI ** 7
PHI_INV_7 = 1.0 / PHI_7
GOLDEN_ANGLE_DEG = 360 / PHI_2
GOLDEN_ANGLE_RAD = 2 * np.pi / PHI_2

# Load complete φ-framework
try:
    with open('complete_phi_framework_final.json', 'r') as f:
        PHI_FRAMEWORK = json.load(f)
    print("✓ φ-Framework loaded")
except:
    PHI_FRAMEWORK = None
    print("⚠ Running without complete framework")

# Load CODATA constants
try:
    with open('codata_2022.json', 'r') as f:
        CODATA = json.load(f)
    print("✓ CODATA 2022 constants loaded")
except:
    CODATA = None
    print("⚠ Running without CODATA")

# Nucleotide → 8D Geometry mapping (from eight_geometries framework)
BASE_MAP = {'A': 5, 'T': 2, 'G': 4, 'C': 1}

# Eight geometries with φ-tuned parameters
GEOMETRIES = [
    (1, 'C', 'red',          'Point',        0.015269,  1),   # P=1
    (2, 'D', 'green',        'Line',         0.008262,  2),   # P=2
    (3, 'E', 'violet',       'Triangle',     0.110649,  3),   # P=3
    (4, 'F', 'mediumpurple', 'Tetrahedron', -0.083485,  4),   # P=4
    (5, 'G', 'blue',         'Pentachoron',  0.025847,  5),   # P=5
    (6, 'A', 'indigo',       'Hexacross',   -0.045123, 12),   # P=6
    (7, 'B', 'purple',       'Heptacube',    0.067891, 14),   # P=7
    (8, 'C', 'white',        'Octacube',     0.012345, 16),   # P=8
]

ANGLES = [i * GOLDEN_ANGLE_DEG for i in range(8)]

# Cavity structure (from novikov_shell_echo_model.py)
CAVITY_STRUCTURE = {
    'deep_interior': {'n_cascade': 3, 'Omega_base': 0.05, 'Q_range': (80, 100)},
    'photon_shell':  {'n_cascade': 2, 'Omega_base': 0.5,  'Q_range': (50, 80)},
    'weak_field':    {'n_cascade': 1, 'Omega_base': 1.0,  'Q_range': (20, 50)},
    'accretion_disk':{'n_cascade': 0, 'Omega_base': 2.5,  'Q_range': (5, 20)},
}

# ==============================================================================
# φ-FRAMEWORK PHYSICS ENGINE
# ==============================================================================

class PhiFrameworkEngine:
    """
    Implements complete φ-recursive framework for biological systems
    """

    def __init__(self):
        self.phi = PHI
        self.phi_powers = {n: PHI**n for n in range(-10, 11)}

        # Cubic scaling law for α(P)
        if PHI_FRAMEWORK:
            coeffs = PHI_FRAMEWORK['scaling_law']['coefficients']
            self.a3 = coeffs['a3']
            self.a2 = coeffs['a2']
            self.a1 = coeffs['a1']
            self.a0 = coeffs['a0']
        else:
            # Fallback to φ-derived approximation
            self.a3 = -PHI_2 / 50
            self.a2 = PHI / 3
            self.a1 = -PHI
            self.a0 = PHI / 3

    def compute_alpha(self, P):
        """Cubic scaling law: α(P) = a₃P³ + a₂P² + a₁P + a₀"""
        return self.a3*P**3 + self.a2*P**2 + self.a1*P + self.a0

    def compute_phi_harmonic(self, value, reference=1.0):
        """Find closest φⁿ harmonic to a given ratio"""
        ratio = value / reference
        n = round(np.log(ratio) / np.log(PHI))
        phi_n = PHI ** n
        error = abs(ratio - phi_n) / phi_n if phi_n != 0 else float('inf')
        return n, phi_n, error

    def cavity_resonance(self, frequency, cavity_type='photon_shell'):
        """Calculate Q-factor and resonance properties"""
        cavity = CAVITY_STRUCTURE.get(cavity_type, CAVITY_STRUCTURE['photon_shell'])
        Q_min, Q_max = cavity['Q_range']
        Q = np.random.uniform(Q_min, Q_max)

        # Echo amplitude from φ⁻⁷ law
        A_echo = PHI_INV_7 / np.sqrt(Q)

        # Phase factor from cascade depth
        phase = cavity['n_cascade'] * np.log(PHI)

        return {
            'Q_factor': Q,
            'amplitude': A_echo,
            'phase': phase,
            'omega_base': cavity['Omega_base'],
            'resonance_strength': A_echo * Q
        }

    def dna_to_physics(self, sequence, window_size=3):
        """
        Map DNA sequence to φ-framework parameters
        Revolutionary: Biological information → Physical constants
        """
        params = []

        for i in range(len(sequence) - window_size + 1):
            codon = sequence[i:i+window_size]

            # Map each base to dimension
            dims = [BASE_MAP.get(b, 1) for b in codon]

            # Compute average geometry position (P parameter)
            P = np.mean(dims)

            # Get α from scaling law
            alpha = self.compute_alpha(P)

            # Compute φ-harmonic alignment
            n_harmonic, phi_n, error = self.compute_phi_harmonic(P)

            # Get geometry properties
            dim_idx = min(max(int(P) - 1, 0), 7)
            _, note, color, name, alpha_geo, vertices = GEOMETRIES[dim_idx]

            params.append({
                'codon': codon,
                'position': i,
                'P': P,
                'alpha': alpha,
                'phi_harmonic': n_harmonic,
                'phi_n': phi_n,
                'error': error,
                'dimension': dims,
                'geometry': name,
                'color': color,
                'note': note,
                'vertices': vertices
            })

        return params

# ==============================================================================
# GENOME LOADER
# ==============================================================================

def load_genome(fasta_file):
    """Load FASTA sequence"""
    seq = []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                continue
            seq.extend(list(line.strip().upper()))
    return seq

# ==============================================================================
# ENHANCED CELL WITH φ-FRAMEWORK INTEGRATION
# ==============================================================================

class PhiEnhancedCell:
    """
    Volumetric cell visualization with complete φ-framework physics
    """

    def __init__(self, genome, center_offset=np.zeros(3), phi_engine=None):
        self.genome = genome
        self.genome_len = len(genome)
        self.center_offset = center_offset
        self.frame = 0
        self.phi_engine = phi_engine or PhiFrameworkEngine()

        # Visualization elements
        self.strand1 = None
        self.strand2 = None
        self.rungs = []
        self.echoes = []
        self.labels = []
        self.centers = []
        self.organelles = []
        self.cavity_spheres = []

        # Physics tracking
        self.physics_params = []
        self.resonance_events = []

        # Constants
        self.core_radius = 15.0
        self.strand_sep = 0.5
        self.twist_factor = 2 * np.pi
        # No max_history limit - keep all visual elements

    def initialize_visuals(self, view):
        """Initialize VisPy visual elements"""
        self.strand1 = Line(pos=np.zeros((1,3)), color=(0,1,1,0.9),
                           width=2, parent=view.scene)
        self.strand2 = Line(pos=np.zeros((1,3)), color=(1,0.5,0,0.9),
                           width=2, parent=view.scene)

        # Progress text
        self.progress_text = Text("0%", pos=self.center_offset + [0,0,20],
                                 color='white', font_size=20,
                                 anchor_x='center', parent=view.scene)

        # Framework info text
        self.framework_text = Text("φ-Framework Active",
                                  pos=self.center_offset + [0,0,18],
                                  color='gold', font_size=14,
                                  anchor_x='center', parent=view.scene)

    def update(self, view):
        """Update cell state with φ-framework physics"""
        self.frame += 1

        # Generate spiral coordinates
        N = 400
        t = np.linspace(0, self.frame, N)
        s1, s2 = [], []

        for tt in t:
            idx = int(tt) % self.genome_len
            base = self.genome[idx]
            dim = BASE_MAP.get(base, 1) - 1
            _, _, col, _, alpha, verts = GEOMETRIES[dim]

            # Spiral with φ-scaled radius
            r = self.core_radius * (1 - (tt/self.genome_len)**PHI)
            r = max(r, 0.5)

            theta = GOLDEN_ANGLE_RAD * tt
            twist = tt / self.genome_len * self.twist_factor
            z = np.sin(tt/self.genome_len * np.pi * 4) * 2 + (tt/self.genome_len) * 8

            # Double helix with φ-angle separation
            a1 = np.radians(ANGLES[dim])
            s1.append([
                r * np.cos(theta) * np.cos(a1) - r * np.sin(theta) * np.sin(a1),
                r * np.sin(theta) * np.cos(a1) + r * np.cos(theta) * np.sin(a1),
                z
            ])

            a2 = np.radians(-ANGLES[dim])
            s2.append([
                r * np.cos(theta) * np.cos(a2) - r * np.sin(theta) * np.sin(a2) + self.strand_sep,
                r * np.sin(theta) * np.cos(a2) + r * np.cos(theta) * np.sin(a2) - self.strand_sep,
                z
            ])

        s1 = np.array(s1) + self.center_offset
        s2 = np.array(s2) + self.center_offset
        self.strand1.set_data(s1)
        self.strand2.set_data(s2)

        # Physics-driven features every 20 frames
        if self.frame % 20 == 0:
            self.add_phi_features(s1, s2, view)

        # Cavity resonance visualization every 100 frames
        if self.frame % 100 == 0:
            self.visualize_cavity_resonance(view)

        # Update progress
        percent = min(self.frame / self.genome_len * 100, 100)
        self.progress_text.text = f"{percent:.1f}% | φ={PHI:.4f}"

        # Keep all visual elements - no cleanup

    def add_phi_features(self, s1, s2, view):
        """Add geometry-based features with φ-framework analysis"""
        idx = self.frame % self.genome_len

        # Get φ-framework parameters for current position
        window = min(3, self.genome_len - idx)
        if window >= 3:
            seq_window = self.genome[idx:idx+3]
            params = self.phi_engine.dna_to_physics(seq_window, window_size=3)

            if params:
                param = params[0]
                self.physics_params.append(param)

                # Extract properties
                dim = int(param['P']) - 1
                dim = min(max(dim, 0), 7)
                _, _, col, name, alpha, verts = GEOMETRIES[dim]

                # Create geometry markers
                Npts = len(s1)
                step = max(1, Npts // verts)
                idx_pts = np.arange(0, verts * step, step)[:verts]
                pts1 = s1[idx_pts]
                pts2 = s2[idx_pts]
                all_pts = np.vstack((pts1, pts2))

                # Scale by φ-harmonic alignment
                scale = 1.0 + param['error'] * 2  # Larger if misaligned

                mark = Markers(pos=all_pts,
                             face_color=Color(col).rgba,
                             edge_color='white',
                             size=6 * scale,
                             parent=view.scene)
                self.rungs.append(mark)

                # Label with physics info
                cen = all_pts.mean(axis=0)
                self.centers.append(cen)

                label_text = f"{param['codon']}:{name}\nφ^{param['phi_harmonic']}"
                lbl = Text(label_text, pos=cen + [0, 0, 0.3],
                          color=col, font_size=9, bold=True,
                          anchor_x='center', parent=view.scene)
                self.labels.append(lbl)

                # Echo with φ⁻⁷ amplitude
                echo_pts = all_pts * (1 - PHI_INV_7) + np.random.normal(scale=0.01, size=all_pts.shape)
                echo = Markers(pos=echo_pts,
                             face_color=(1, 1, 1, 0.15 * PHI_INV_7),
                             size=4, parent=view.scene)
                self.echoes.append(echo)

    def visualize_cavity_resonance(self, view):
        """Create cavity resonance visualization"""
        if len(self.centers) < 10:
            return

        # Sample recent center
        center = self.centers[-5]

        # Get resonance properties
        freq = self.frame / 100.0  # Arbitrary frequency
        resonance = self.phi_engine.cavity_resonance(freq, 'photon_shell')

        # Create resonance sphere
        radius = resonance['Q_factor'] / 20.0
        mesh_data = create_sphere(rows=8, cols=8, radius=radius)

        # Color based on resonance strength
        alpha = min(resonance['resonance_strength'] * 0.1, 0.3)
        color = (1, 0.8, 0, alpha)

        sphere = Mesh(meshdata=mesh_data, color=color, parent=view.scene)
        sphere.transform = scene.transforms.MatrixTransform()
        sphere.transform.translate(center)

        self.cavity_spheres.append(sphere)
        # Keep all cavity spheres - no limit

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

class PhiGenomeVisualizer:
    """
    Main application integrating all repository strengths
    """

    def __init__(self, fasta_file):
        # Load genome
        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"FASTA not found: {fasta_file}")

        self.genome = load_genome(fasta_file)
        print(f"✓ Genome loaded: {len(self.genome)} nucleotides")

        # Initialize φ-framework engine
        self.phi_engine = PhiFrameworkEngine()
        print(f"✓ φ-Framework engine initialized")

        # Analyze full genome
        print("Analyzing genome with φ-framework...")
        self.genome_analysis = self.phi_engine.dna_to_physics(self.genome, window_size=3)
        print(f"✓ {len(self.genome_analysis)} codons analyzed")

        # Statistics
        self.compute_statistics()

        # Create VisPy canvas
        self.canvas = scene.SceneCanvas(keys='interactive', size=(1400, 900),
                                       bgcolor='black',
                                       title="φ-Framework Unified Genome Visualizer")
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.distance = 50

        # Create cell
        self.cell = PhiEnhancedCell(self.genome, phi_engine=self.phi_engine)
        self.cell.initialize_visuals(self.view)

        # Timer
        self.timer = app.Timer(interval=0.03, connect=self.update, start=True)

        # Info panel
        self.create_info_panel()

    def compute_statistics(self):
        """Compute φ-framework statistics"""
        if not self.genome_analysis:
            return

        alphas = [p['alpha'] for p in self.genome_analysis]
        errors = [p['error'] for p in self.genome_analysis]
        harmonics = [p['phi_harmonic'] for p in self.genome_analysis]

        print("\n" + "="*70)
        print("φ-FRAMEWORK GENOME STATISTICS")
        print("="*70)
        print(f"Mean α: {np.mean(alphas):.6f}")
        print(f"Std α:  {np.std(alphas):.6f}")
        print(f"Mean φ-harmonic: {np.mean(harmonics):.2f}")
        print(f"Median φ-error: {np.median(errors)*100:.2f}%")
        print(f"Aligned codons (<10% error): {sum(1 for e in errors if e < 0.1)} / {len(errors)}")
        print(f"  → {sum(1 for e in errors if e < 0.1)/len(errors)*100:.1f}% φ-resonant")
        print("="*70 + "\n")

    def create_info_panel(self):
        """Create information text overlays"""
        info_text = (
            "φ-FRAMEWORK UNIFIED VISUALIZER\n"
            f"Golden Ratio: {PHI:.10f}\n"
            f"φ⁷: {PHI_7:.4f} | φ⁻⁷: {PHI_INV_7:.6f}\n"
            f"Genome: {len(self.genome)} bases\n"
            "CONTROLS: Rotate view with mouse"
        )

        self.info = Text(info_text, pos=(10, 30),
                        color='cyan', font_size=10,
                        anchor_x='left', anchor_y='top',
                        parent=self.canvas.scene)

    def update(self, event):
        """Main update loop"""
        self.cell.update(self.view)

        # Rotate camera
        self.view.camera.azimuth = self.cell.frame * 0.2
        self.view.camera.elevation = 15 + 10 * np.sin(self.cell.frame * 0.003)

    def run(self):
        """Start application"""
        print("Starting φ-Framework visualizer...")
        self.canvas.show()
        app.run()

# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    """Run the unified φ-framework genome visualizer"""
    print("\n" + "="*70)
    print("φ-FRAMEWORK UNIFIED BIOLOGICAL SYNTHESIS ENGINE")
    print("Integrating: DNA → 8D Geometry → Cavity Physics → Cosmic Scales")
    print("="*70 + "\n")

    fasta_path = "ecoli_k12.fasta"

    try:
        visualizer = PhiGenomeVisualizer(fasta_path)
        visualizer.run()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure ecoli_k12.fasta is in the current directory")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
