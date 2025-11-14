#!/usr/bin/env python3
"""
==============================================================================
φ-FRAMEWORK GENOME-SCALE PHYSICS ENGINE WITH CELL DIVISION
==============================================================================
Combines:
- FASTA-driven DNA visualization with volumetric cell division
- High-precision D_n operator (arbitrary precision concepts)
- Genome-scale physics mapping (no cosmological assumptions)
- Real-time φ-harmonic analysis of biological sequences
- Organelle dynamics driven by nucleotide physics

Key Innovation: Maps DNA sequences directly to physical parameters
at GENOME SCALE - no cosmological extrapolation needed!
==============================================================================
"""

import os
import numpy as np
from decimal import Decimal, getcontext
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

# Set high precision for Decimal operations
getcontext().prec = 100

# ==============================================================================
# FUNDAMENTAL CONSTANTS
# ==============================================================================

PHI = (1 + np.sqrt(5)) / 2
PHI_2 = PHI ** 2
PHI_7 = PHI ** 7
PHI_INV_7 = 1.0 / PHI_7
GOLDEN_ANGLE_DEG = 360 / PHI_2
GOLDEN_ANGLE_RAD = 2 * np.pi / PHI_2

# First 50 primes for D_n operator
PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229
]

# ==============================================================================
# NUCLEOTIDE → 8D GEOMETRY MAPPING
# ==============================================================================

BASE_MAP = {'A': 5, 'T': 2, 'G': 4, 'C': 1}

GEOMETRIES = [
    (1, 'C', 'red',          'Point',        0.015269,  1),
    (2, 'D', 'green',        'Line',         0.008262,  2),
    (3, 'E', 'violet',       'Triangle',     0.110649,  3),
    (4, 'F', 'mediumpurple', 'Tetrahedron', -0.083485,  4),
    (5, 'G', 'blue',         'Pentachoron',  0.025847,  5),
    (6, 'A', 'indigo',       'Hexacross',   -0.045123, 12),
    (7, 'B', 'purple',       'Heptacube',    0.067891, 14),
    (8, 'C', 'white',        'Octacube',     0.012345, 16),
]

ANGLES = [i * GOLDEN_ANGLE_DEG for i in range(8)]

# ==============================================================================
# HIGH-PRECISION PHYSICS ENGINE (GENOME-SCALE)
# ==============================================================================

class GenomeScalePhysicsEngine:
    """
    Implements D_n operator with arbitrary precision for genome-scale analysis
    No cosmological assumptions - pure biological information processing
    """

    def __init__(self):
        # Cubic scaling coefficients (φ-derived)
        self.a3 = -PHI_2 / 50
        self.a2 = PHI / 3
        self.a1 = -PHI
        self.a0 = PHI / 3

        # Caches
        self.fibonacci_cache = {}

    def fibonacci_real(self, n):
        """Binet's formula with harmonic correction"""
        if n in self.fibonacci_cache:
            return self.fibonacci_cache[n]

        if n > 100:
            return 0.0

        # High precision calculation
        phi_d = Decimal(str(PHI))
        phi_inv_d = Decimal(str(1/PHI))
        sqrt5_d = Decimal(5).sqrt()
        n_d = Decimal(str(n))

        term1 = float(phi_d ** n_d / sqrt5_d)
        term2 = float(phi_inv_d ** n_d) * np.cos(np.pi * n)

        result = term1 - term2
        self.fibonacci_cache[n] = result
        return result

    def prime_product_index(self, n, beta):
        """Get prime from index with offset"""
        idx = (int(np.floor(n + beta)) + 50) % 50
        return float(PRIMES[idx])

    def power_extreme(self, base, exponent):
        """Handle extreme exponents using logarithms"""
        if base <= 0:
            return 0.0

        is_negative = (exponent < 0)
        exp_abs = abs(exponent)

        log_base = np.log(base)
        log_result = exp_abs * log_base

        # Check representable range
        if log_result > 709:
            base_d = Decimal(str(base))
            result_d = base_d ** Decimal(str(exp_abs))
            result = float(result_d) if result_d < Decimal('1e308') else np.inf
        elif log_result < -709:
            result = np.exp(log_result)
        else:
            result = np.exp(log_result)

        return (1.0 / result) if is_negative else result

    def D_n(self, n, beta=0, r=1.0, k=1.0, Omega=1.0, base=2.0):
        """
        Universal D_n operator: sqrt(φ * F_n * P_n * base^n * Omega) * r^k
        """
        Fn = self.fibonacci_real(n + beta)
        Pn = self.prime_product_index(n, beta)

        # Handle extreme base^n
        base_power = self.power_extreme(base, n + beta)

        # Compute product
        inside_sqrt = PHI * abs(Fn) * Pn * base_power * Omega
        inside_sqrt = max(inside_sqrt, 1e-308)

        result = np.sqrt(inside_sqrt)

        # Apply sign and scaling
        if Fn < 0:
            result = -result
        result *= (r ** k)

        return result

    def compute_alpha(self, P):
        """Cubic scaling law: α(P) = a₃P³ + a₂P² + a₁P + a₀"""
        return self.a3 * P**3 + self.a2 * P**2 + self.a1 * P + self.a0

    def phi_harmonic_decomposition(self, value, reference=1.0):
        """Find closest φⁿ harmonic"""
        if reference == 0:
            return 0, 1.0, float('inf')

        ratio = value / reference
        if ratio <= 0:
            return 0, 1.0, float('inf')

        n = round(np.log(ratio) / np.log(PHI))
        phi_n = PHI ** n
        error = abs(ratio - phi_n) / phi_n

        return n, phi_n, error

    def cavity_Q_factor(self, position, genome_length):
        """
        Calculate Q-factor for resonance
        Scales with position in genome
        """
        # Map position to Q-factor range
        progress = position / genome_length
        Q_min, Q_max = 20, 100
        Q = Q_min + (Q_max - Q_min) * (0.5 + 0.5 * np.sin(progress * 2 * np.pi))

        # Echo amplitude from φ⁻⁷
        A_echo = PHI_INV_7 / np.sqrt(Q)

        # Phase factor
        phase = (position % 50) * np.log(PHI)

        return {
            'Q': Q,
            'amplitude': A_echo,
            'phase': phase,
            'resonance_strength': A_echo * Q
        }

# ==============================================================================
# DNA → PHYSICS MAPPER (GENOME-SCALE)
# ==============================================================================

class DNAPhysicsMapper:
    """Maps DNA sequences to physics parameters at genome scale"""

    def __init__(self):
        self.engine = GenomeScalePhysicsEngine()

    def analyze_codon(self, codon, position, genome_length):
        """
        Convert single codon to physics parameters
        Returns dict with all physical properties
        """
        # Map bases to dimensions
        dims = [BASE_MAP.get(b, 1) for b in codon]
        P = np.mean(dims)

        # Get α from cubic scaling
        alpha = self.engine.compute_alpha(P)

        # Compute D_n value
        n = position % 100
        Dn_value = self.engine.D_n(n, beta=P-3, r=1.05, k=alpha, Omega=1.0, base=2.0)

        # φ-harmonic analysis
        n_harmonic, phi_n, error = self.engine.phi_harmonic_decomposition(P)

        # Cavity Q-factor
        cavity = self.engine.cavity_Q_factor(position, genome_length)

        # Geometry properties
        dim_idx = min(max(int(P) - 1, 0), 7)
        _, note, color, name, alpha_geo, vertices = GEOMETRIES[dim_idx]

        return {
            'position': position,
            'codon': codon,
            'P': P,
            'alpha': alpha,
            'Dn_value': Dn_value,
            'phi_harmonic': n_harmonic,
            'phi_n': phi_n,
            'phi_error': error,
            'Q_factor': cavity['Q'],
            'echo_amplitude': cavity['amplitude'],
            'resonance': cavity['resonance_strength'],
            'dimension': dim_idx + 1,
            'geometry': name,
            'color': color,
            'note': note,
            'vertices': vertices
        }

# ==============================================================================
# GENOME LOADER
# ==============================================================================

def load_genome(fasta_file):
    """Load FASTA sequence"""
    sequence = []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                continue
            sequence.extend(list(line.strip().upper()))
    return sequence

# ==============================================================================
# ORGANELLE SYSTEM
# ==============================================================================

def spawn_organelle(center, color_rgb, size=0.5, n=12, parent=None):
    """Create organelle cluster"""
    pts = center + np.random.normal(scale=0.2, size=(n, 3)) * size
    rgba = list(Color(color_rgb).rgba)
    rgba[3] = 1.0
    mark = Markers(pos=pts, face_color=rgba, edge_color=None, size=5, parent=parent)
    return {"marker": mark, "positions": pts, "color": rgba, "size": size}

def lattice_push(pt, lattice_nodes, positive=True, strength=0.02):
    """Push organelle based on lattice backpressure"""
    if len(lattice_nodes) == 0:
        return pt
    nearest = lattice_nodes[np.random.randint(0, len(lattice_nodes))]
    dir_vec = nearest - pt
    if not positive:
        dir_vec *= -1
    return pt + dir_vec * strength

# ==============================================================================
# ENHANCED CELL WITH DIVISION
# ==============================================================================

class PhysicsEnhancedCell:
    """DNA cell with physics-driven visualization and division capability"""

    def __init__(self, genome, mapper, center_offset=np.zeros(3), parent_scene=None):
        self.genome = genome
        self.genome_len = len(genome)
        self.mapper = mapper
        self.center_offset = center_offset
        self.frame = 0
        self.scene = parent_scene

        # Visual elements
        self.strand1 = Line(pos=np.zeros((1,3)), color=(0,1,1,0.9),
                           width=2, parent=self.scene)
        self.strand2 = Line(pos=np.zeros((1,3)), color=(1,0.5,0,0.9),
                           width=2, parent=self.scene)

        self.rungs = []
        self.echoes = []
        self.links = []
        self.labels = []
        self.centers = []
        self.organelles = []

        # Physics tracking
        self.current_params = None

        # Progress display
        self.progress_text = Text("0%", pos=self.center_offset + [0, 0, 20],
                                 color='white', font_size=24,
                                 anchor_x='center', parent=self.scene)

        # Physics info
        self.physics_text = Text("", pos=self.center_offset + [0, 0, 18],
                                color='gold', font_size=12,
                                anchor_x='center', parent=self.scene)

        # Constants
        self.core_radius = 15.0
        self.twist_factor = 2 * np.pi
        self.max_history = 6000

    def update(self):
        """Update cell state with physics"""
        self.frame += 1

        # Generate helix
        N = 400
        t = np.linspace(0, self.frame, N)
        s1, s2 = [], []

        for idx, tt in enumerate(t):
            genome_idx = int(tt) % self.genome_len
            base = self.genome[genome_idx]
            dim = BASE_MAP.get(base, 1) - 1

            # Get physics for current base
            if len(self.genome) >= 3:
                codon_start = max(0, genome_idx - 1)
                codon = ''.join(self.genome[codon_start:codon_start+3])
                if len(codon) == 3:
                    params = self.mapper.analyze_codon(codon, genome_idx, self.genome_len)
                    # Scale radius by D_n value
                    r_scale = 1.0 + params['Dn_value'] * 0.01
                else:
                    r_scale = 1.0
            else:
                r_scale = 1.0

            # Spiral coordinates
            r = self.core_radius * r_scale * (1 - (idx/N)**PHI)
            r = max(r, 0.5)

            theta = GOLDEN_ANGLE_RAD * tt
            z = np.sin(tt/N * np.pi * 4) * 2 + (idx/N) * 8

            # Double helix
            a1 = np.radians(ANGLES[dim])
            s1.append([
                r * np.cos(theta) * np.cos(a1) - r * np.sin(theta) * np.sin(a1),
                r * np.sin(theta) * np.cos(a1) + r * np.cos(theta) * np.sin(a1),
                z
            ])

            a2 = np.radians(-ANGLES[dim])
            s2.append([
                r * np.cos(theta) * np.cos(a2) - r * np.sin(theta) * np.sin(a2) + 0.5,
                r * np.sin(theta) * np.cos(a2) + r * np.cos(theta) * np.sin(a2) - 0.5,
                z
            ])

        s1 = np.array(s1) + self.center_offset
        s2 = np.array(s2) + self.center_offset
        self.strand1.set_data(s1)
        self.strand2.set_data(s2)

        # Add features every 20 frames
        if self.frame % 20 == 0:
            self.add_physics_features(s1, s2)

        # Update organelles with lattice backpressure
        genome_idx = self.frame % self.genome_len
        current_base = self.genome[genome_idx]
        lattice_nodes = np.array(self.centers) if self.centers else np.array([])

        for org in self.organelles:
            positive = current_base in ['A', 'T']
            new_pts = np.array([lattice_push(p, lattice_nodes, positive=positive, strength=0.02)
                               for p in org['positions']])
            org['positions'] = new_pts
            org['marker'].set_data(pos=new_pts, face_color=org['color'], size=5)

        # Update displays
        percent = min(self.frame / self.genome_len * 100, 100)
        self.progress_text.text = f"{percent:.2f}%"

        if self.current_params:
            info = (f"φ^{self.current_params['phi_harmonic']} | "
                   f"Q={self.current_params['Q_factor']:.1f} | "
                   f"Dn={self.current_params['Dn_value']:.2e}")
            self.physics_text.text = info

        # Limit history
        if len(self.rungs) > self.max_history:
            drop = len(self.rungs) // 3
            self.rungs = self.rungs[drop:]
            self.echoes = self.echoes[drop:]
            self.links = self.links[drop:]
            self.labels = self.labels[drop:]
            self.centers = self.centers[drop:]
            self.organelles = self.organelles[drop:]

    def add_physics_features(self, s1, s2):
        """Add geometry markers with physics info"""
        genome_idx = self.frame % self.genome_len

        # Get codon
        codon_start = max(0, genome_idx - 1)
        codon = ''.join(self.genome[codon_start:codon_start+3])

        if len(codon) != 3:
            return

        # Get physics parameters
        params = self.mapper.analyze_codon(codon, genome_idx, self.genome_len)
        self.current_params = params

        # Get geometry
        dim = params['dimension'] - 1
        _, _, col, name, _, verts = GEOMETRIES[dim]

        # Sample points
        Npts = len(s1)
        step = max(1, Npts // verts)
        idx_pts = np.arange(0, verts * step, step)[:verts]
        pts1 = s1[idx_pts]
        pts2 = s2[idx_pts]
        all_pts = np.vstack((pts1, pts2))

        # Scale by resonance strength
        size = 6 * (1 + params['resonance'] * 0.1)

        # Create rung markers
        mark = Markers(pos=all_pts, face_color=Color(col).rgba,
                      edge_color='white', size=size, parent=self.scene)
        self.rungs.append(mark)

        # Center tracking
        cen = all_pts.mean(axis=0)
        self.centers.append(cen)

        # Label
        label_text = f"{params['codon']}:{name}\nφ^{params['phi_harmonic']}"
        lbl = Text(label_text, pos=cen + [0, 0, 0.3],
                  color=col, font_size=10, bold=True,
                  anchor_x='center', parent=self.scene)
        self.labels.append(lbl)

        # Echo with φ⁻⁷ amplitude
        if len(self.centers) > 1:
            echo_pts = all_pts * (1 - PHI_INV_7) + np.random.normal(scale=0.01, size=all_pts.shape)
            echo = Markers(pos=echo_pts, face_color=(1, 1, 1, 0.25 * PHI_INV_7),
                          size=4, parent=self.scene)
            self.echoes.append(echo)

        # Links to previous
        if len(self.centers) > 1:
            prev_c = self.centers[-2]
            segs = []
            for p in all_pts[:6]:
                segs += [prev_c, p]
            link = Line(pos=np.array(segs), color=(0.7, 0.7, 0.7, 0.4),
                       width=1, connect='segments', parent=self.scene)
            self.links.append(link)

        # Spawn organelles based on geometry dimension
        spawn_prob = 0.02 + dim * 0.02
        cluster_size = 6 + dim
        if np.random.rand() < spawn_prob:
            org = spawn_organelle(cen, col, size=0.3, n=cluster_size, parent=self.scene)
            self.organelles.append(org)

# ==============================================================================
# MAIN APPLICATION WITH CELL DIVISION
# ==============================================================================

class GenomeVisualizerApp:
    """Main application with cell division"""

    def __init__(self, fasta_file, division_interval=2000):
        # Load genome
        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"FASTA not found: {fasta_file}")

        self.genome = load_genome(fasta_file)
        print(f"✓ Genome loaded: {len(self.genome)} nucleotides")

        # Initialize mapper
        self.mapper = DNAPhysicsMapper()

        # Division parameters
        self.division_interval = division_interval

        # Compute statistics
        self.compute_statistics()

        # Create canvas
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(1600, 1000),
            bgcolor='black',
            title="φ-Framework Genome Visualizer with Cell Division"
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.distance = 50

        # Create initial cell
        self.cells = [PhysicsEnhancedCell(self.genome, self.mapper,
                                          center_offset=np.array([0, 0, 0]),
                                          parent_scene=self.view.scene)]

        # Timer
        self.timer = app.Timer(interval=0.02, connect=self.update, start=True)

        # Info panel
        self.create_info_panel()

        self.frame = 0

    def compute_statistics(self):
        """Compute physics statistics on genome sample"""
        print("\n" + "="*80)
        print("GENOME-SCALE PHYSICS ANALYSIS")
        print("="*80)

        # Analyze first 1000 codons
        sample_size = min(3000, len(self.genome))
        codons_analyzed = 0
        alphas = []
        Qs = []
        errors = []

        for i in range(0, sample_size - 2, 3):
            codon = ''.join(self.genome[i:i+3])
            if len(codon) == 3:
                params = self.mapper.analyze_codon(codon, i, len(self.genome))
                alphas.append(params['alpha'])
                Qs.append(params['Q_factor'])
                errors.append(params['phi_error'])
                codons_analyzed += 1

        if codons_analyzed > 0:
            print(f"Sample size: {codons_analyzed} codons")
            print(f"Mean α: {np.mean(alphas):.6f} ± {np.std(alphas):.6f}")
            print(f"Mean Q-factor: {np.mean(Qs):.2f} ± {np.std(Qs):.2f}")
            print(f"Median φ-error: {np.median(errors)*100:.2f}%")
            aligned = sum(1 for e in errors if e < 0.1)
            print(f"φ-aligned codons (<10% error): {aligned}/{codons_analyzed}")
            print(f"  → {aligned/codons_analyzed*100:.1f}% φ-resonant")

        print("="*80 + "\n")

    def create_info_panel(self):
        """Create information overlay"""
        info_text = (
            "φ-FRAMEWORK GENOME VISUALIZER\n"
            f"φ = {PHI:.10f}\n"
            f"φ⁷ = {PHI_7:.4f} | φ⁻⁷ = {PHI_INV_7:.6f}\n"
            f"Genome: {len(self.genome)} bases\n"
            f"Division every {self.division_interval} frames\n"
            "Scale: GENOME (not cosmological)"
        )

        self.info = Text(
            info_text,
            pos=(10, 30),
            color='cyan',
            font_size=10,
            anchor_x='left',
            anchor_y='top',
            parent=self.canvas.scene
        )

    def update(self, event):
        """Main update loop with cell division"""
        self.frame += 1

        # Update all cells
        new_cells = []
        for cell in self.cells:
            cell.update()

            # Check for division
            if cell.frame > 0 and cell.frame % self.division_interval == 0:
                # Create daughter cell
                offset = np.random.normal(scale=5.0, size=3)
                daughter = PhysicsEnhancedCell(
                    self.genome,
                    self.mapper,
                    center_offset=cell.center_offset + offset,
                    parent_scene=self.view.scene
                )
                new_cells.append(daughter)
                print(f"✓ Cell division at frame {self.frame} - Total cells: {len(self.cells) + len(new_cells)}")

        self.cells.extend(new_cells)

        # Rotate camera
        self.view.camera.azimuth = self.frame * 0.2
        self.view.camera.elevation = 15 + 10 * np.sin(self.frame * 0.003)

    def run(self):
        """Start application"""
        print("Starting genome visualizer with cell division...")
        self.canvas.show()
        app.run()

# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    """Run the genome-scale visualizer"""
    print("\n" + "="*80)
    print("φ-FRAMEWORK GENOME-SCALE VISUALIZER WITH CELL DIVISION")
    print("DNA → Physics → Organelle Dynamics → Division")
    print("="*80 + "\n")

    fasta_path = "ecoli_k12.fasta"

    try:
        visualizer = GenomeVisualizerApp(fasta_path, division_interval=2000)
        visualizer.run()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Creating demo with synthetic sequence...")

        # Create synthetic genome
        synthetic = ['A', 'T', 'G', 'C'] * 250
        import random
        random.shuffle(synthetic)

        with open('demo_genome.fasta', 'w') as f:
            f.write(">Demo Sequence\n")
            for i in range(0, len(synthetic), 80):
                f.write(''.join(synthetic[i:i+80]) + '\n')

        print("✓ Created demo_genome.fasta")
        visualizer = GenomeVisualizerApp('demo_genome.fasta', division_interval=500)
        visualizer.run()

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()