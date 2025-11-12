#!/usr/bin/env python3
"""
==============================================================================
FASTA-FIRST φ-FRAMEWORK GENOME ENGINE
==============================================================================
Grand Master Equation Implementation:
- FASTA sequence drives ALL parameters (τ, k, α, v)
- Continuous φ-spiral flow: S±(τ) with genome-determined growth
- Discrete φ-octave jumps: Rung emergence at nucleotide boundaries
- Recursive φ-wavefunction: Ψₖ₊₁ = φ^(1/2) r^φ e^(i2πφ) Ψₖ
- Cell division: Natural consequence of genome completion cycles

No arbitrary time steps - the genome IS the clock!
==============================================================================
"""

import os
import numpy as np
from decimal import Decimal, getcontext
from vispy import scene, app
from vispy.scene.visuals import Line, Markers, Text
from vispy.color import Color

getcontext().prec = 100

# ==============================================================================
# φ-HARMONIC CONSTANTS (from Grand Master Equation)
# ==============================================================================

PHI = (1 + np.sqrt(5)) / 2
PHI_2 = PHI ** 2
PHI_INV = 1.0 / PHI
PHI_INV_7 = PHI ** (-7)
GOLDEN_ANGLE_RAD = 2 * np.pi / PHI_2  # G = 2π/φ²

# First 50 primes for D_n operator
PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229
]

# ==============================================================================
# NUCLEOTIDE → φ-OCTAVE MAPPING (k ∈ {0,...,7})
# ==============================================================================

BASE_MAP = {'A': 5, 'T': 2, 'G': 4, 'C': 1}  # Maps to octave index

# Eight geometries with φ-tuned αₖ (growth exponents) and vₖ (vertex counts)
GEOMETRIES = [
    # (k, note, color, name, αₖ, vₖ)
    (1, 'C', 'red',          'Point',        0.015269,  1),
    (2, 'D', 'green',        'Line',         0.008262,  2),
    (3, 'E', 'violet',       'Triangle',     0.110649,  3),
    (4, 'F', 'mediumpurple', 'Tetrahedron', -0.083485,  4),
    (5, 'G', 'blue',         'Pentachoron',  0.025847,  5),
    (6, 'A', 'indigo',       'Hexacross',   -0.045123, 12),
    (7, 'B', 'purple',       'Heptacube',    0.067891, 14),
    (8, 'C', 'white',        'Octacube',     0.012345, 16),
]

# ==============================================================================
# GENOME LOADER
# ==============================================================================

def load_genome(fasta_file):
    """Load FASTA - this is our primary clock/data source"""
    sequence = []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                continue
            sequence.extend(list(line.strip().upper()))
    return sequence

# ==============================================================================
# φ-RECURSIVE PHYSICS ENGINE (Grand Master Equation)
# ==============================================================================

class PhiRecursiveEngine:
    """
    Implements the Grand Master Equation:
    - Flow: S±(τ) = (r(τ)cos(θ±aₖ), r(τ)sin(θ±aₖ), ℓτ)
    - Jump: Discrete octave rungs at nucleotide boundaries
    - Recursive wavefunction: Ψₖ₊₁ = φ^(1/2) r^φ e^(i2πφ) Ψₖ
    """

    def __init__(self, genome):
        self.genome = genome
        self.genome_len = len(genome)

        # Period parameters (from genome structure)
        self.P = 100  # Temporal period (nucleotides per octave cycle)
        self.s = 1.0  # Speed factor
        self.ell = 0.8 / self.P  # Vertical scale ℓ = 0.8/P
        self.gamma = 0.75  # Echo scale γ < 1

        # Fibonacci cache
        self.fib_cache = {}

    def get_octave_k(self, tau):
        """k(τ) = min(⌊sτ/P⌋, 7) - octave index from genome position"""
        return min(int(self.s * tau / self.P), 7)

    def get_tau_tilde(self, tau):
        """τ̃(τ) = τ mod P - phase within period"""
        return tau % self.P

    def get_alpha_k(self, k):
        """Growth exponent αₖ from geometry table"""
        return GEOMETRIES[k][4]

    def get_vertices_k(self, k):
        """Vertex count vₖ from geometry table"""
        return GEOMETRIES[k][5]

    def compute_r(self, tau):
        """r(τ) = exp(αₖ(τ) τ̃(τ)) - radial growth"""
        k = self.get_octave_k(tau)
        tau_tilde = self.get_tau_tilde(tau)
        alpha_k = self.get_alpha_k(k)
        return np.exp(alpha_k * tau_tilde)

    def compute_theta(self, tau):
        """θ(τ) = (2π/P)τ - angular progression"""
        return (2 * np.pi / self.P) * tau

    def compute_a_k(self, k):
        """aₖ = k·(2π/φ²) - golden angle offset"""
        return k * GOLDEN_ANGLE_RAD

    def S_plus(self, tau, base_radius=15.0):
        """
        S₊(τ) - positive strand of double helix
        Returns: [x, y, z] coordinates
        """
        k = self.get_octave_k(tau)
        r = self.compute_r(tau) * base_radius
        theta = self.compute_theta(tau)
        a_k = self.compute_a_k(k)
        z = self.ell * tau

        x = r * np.cos(theta + a_k)
        y = r * np.sin(theta + a_k)

        return np.array([x, y, z])

    def S_minus(self, tau, base_radius=15.0, separation=0.5):
        """
        S₋(τ) - negative strand of double helix
        Returns: [x, y, z] coordinates
        """
        k = self.get_octave_k(tau)
        r = self.compute_r(tau) * base_radius
        theta = self.compute_theta(tau)
        a_k = self.compute_a_k(k)
        z = self.ell * tau

        x = r * np.cos(theta - a_k) + separation
        y = r * np.sin(theta - a_k) - separation

        return np.array([x, y, z])

    def fibonacci_real(self, n):
        """Fibonacci with Binet's formula for wavefunction"""
        if n in self.fib_cache:
            return self.fib_cache[n]

        if n > 100:
            return 0.0

        phi_d = Decimal(str(PHI))
        phi_inv_d = Decimal(str(PHI_INV))
        sqrt5_d = Decimal(5).sqrt()
        n_d = Decimal(str(n))

        term1 = float(phi_d ** n_d / sqrt5_d)
        term2 = float(phi_inv_d ** n_d) * np.cos(np.pi * n)

        result = term1 - term2
        self.fib_cache[n] = result
        return result

    def psi_recursive(self, k, tau):
        """
        Ψₖ(τ) = φ^(k/2) · r(τ)^(φᵏ) · e^(i2πkφ) · (cos(φᵏθ) + i·sin(φᵏθ))
        Recursive wavefunction generating octave hierarchy
        """
        r = self.compute_r(tau)
        theta = self.compute_theta(tau)

        # Amplitude normalization
        amplitude = PHI ** (k / 2)

        # Radial scaling
        radial = r ** (PHI ** k)

        # Phase factor
        phase_global = np.exp(1j * 2 * np.pi * k * PHI)
        phase_local = np.cos(PHI ** k * theta) + 1j * np.sin(PHI ** k * theta)

        psi = amplitude * radial * phase_global * phase_local

        return psi

    def nucleotide_to_params(self, position):
        """
        FASTA-FIRST: Extract ALL parameters from nucleotide at position
        Returns dict with τ, k, α, v, geometry info
        """
        if position >= self.genome_len:
            return None

        base = self.genome[position]

        # τ is simply the position in genome
        tau = float(position)

        # k determined by base type
        k_raw = BASE_MAP.get(base, 1) - 1
        k = min(max(k_raw, 0), 7)

        # Get geometry parameters
        _, note, color, name, alpha_k, v_k = GEOMETRIES[k]

        # Compute wavefunction
        psi = self.psi_recursive(k, tau)

        # Fibonacci and prime for D_n
        F_n = self.fibonacci_real(tau % 100)
        P_n = PRIMES[int(tau) % 50]

        return {
            'position': position,
            'base': base,
            'tau': tau,
            'k': k,
            'alpha_k': alpha_k,
            'v_k': v_k,
            'geometry': name,
            'color': color,
            'note': note,
            'psi': psi,
            'psi_magnitude': abs(psi),
            'psi_phase': np.angle(psi),
            'F_n': F_n,
            'P_n': P_n,
            'r': self.compute_r(tau),
            'theta': self.compute_theta(tau),
            'a_k': self.compute_a_k(k)
        }

# ==============================================================================
# FASTA-FIRST CELL (Genome Drives Everything)
# ==============================================================================

class GenomeDrivenCell:
    """
    Cell where genome is the PRIMARY driver
    Every update: read next nucleotide → generate structures
    """

    def __init__(self, genome, engine, center_offset=np.zeros(3), parent_scene=None):
        self.genome = genome
        self.genome_len = len(genome)
        self.engine = engine
        self.center_offset = center_offset
        self.scene = parent_scene

        # Current genome position (this IS our time parameter τ)
        self.tau = 0
        self.frame = 0

        # Visual elements (all persistent)
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

        # Stats
        self.total_structures = 0

        # Displays
        self.progress_text = Text("0%", pos=self.center_offset + [0, 0, 20],
                                 color='white', font_size=20,
                                 anchor_x='center', parent=self.scene)

        self.physics_text = Text("", pos=self.center_offset + [0, 0, 18],
                                color='gold', font_size=10,
                                anchor_x='center', parent=self.scene)

        self.structure_text = Text("", pos=self.center_offset + [0, 0, 16],
                                   color='lime', font_size=9,
                                   anchor_x='center', parent=self.scene)

    def update(self):
        """
        FASTA-FIRST update:
        1. Read current nucleotide at position τ
        2. Generate S±(τ) strand coordinates
        3. Create rungs at octave boundaries
        4. Increment τ (genome position)
        """
        self.frame += 1

        # Generate continuous strand segments
        N = 400  # Points to render
        tau_start = self.tau
        tau_end = min(self.tau + N, self.genome_len)

        s1_points = []
        s2_points = []

        for tau_val in range(int(tau_start), int(tau_end)):
            if tau_val >= self.genome_len:
                break

            # Get nucleotide parameters
            params = self.engine.nucleotide_to_params(tau_val)
            if params is None:
                continue

            # Generate S±(τ) coordinates
            p1 = self.engine.S_plus(tau_val)
            p2 = self.engine.S_minus(tau_val)

            s1_points.append(p1)
            s2_points.append(p2)

        if s1_points:
            s1 = np.array(s1_points) + self.center_offset
            s2 = np.array(s2_points) + self.center_offset
            self.strand1.set_data(s1)
            self.strand2.set_data(s2)

        # Create rung at current position (every 20 frames for visibility)
        if self.frame % 20 == 0 and self.tau < self.genome_len:
            self.create_rung()

        # Update organelle dynamics
        self.update_organelles()

        # Increment τ (genome clock)
        self.tau += 1
        if self.tau >= self.genome_len:
            self.tau = 0  # Loop back for continuous visualization

        # Update displays
        self.update_displays()

    def create_rung(self):
        """
        Create discrete φ-octave rung at current τ
        Sample vₖ vertices from each strand
        """
        params = self.engine.nucleotide_to_params(int(self.tau))
        if params is None:
            return

        k = params['k']
        v_k = params['v_k']
        color = params['color']
        name = params['geometry']
        base = params['base']

        # Sample v_k points around current τ
        tau_samples = np.linspace(self.tau - 5, self.tau + 5, v_k)

        pts1 = []
        pts2 = []
        for tau_val in tau_samples:
            if 0 <= tau_val < self.genome_len:
                p1 = self.engine.S_plus(tau_val)
                p2 = self.engine.S_minus(tau_val)
                pts1.append(p1 + self.center_offset)
                pts2.append(p2 + self.center_offset)

        if not pts1:
            return

        all_pts = np.vstack((pts1, pts2))

        # Scale size by wavefunction magnitude
        size = 6 * (1 + params['psi_magnitude'] * 0.1)

        # Create rung markers
        mark = Markers(pos=all_pts, face_color=Color(color).rgba,
                      edge_color='white', size=size, parent=self.scene)
        self.rungs.append(mark)

        # Center tracking
        cen = all_pts.mean(axis=0)
        self.centers.append(cen)

        # Label
        label_text = f"{base}:{name}\nτ={int(self.tau)}"
        lbl = Text(label_text, pos=cen + [0, 0, 0.3],
                  color=color, font_size=9, bold=True,
                  anchor_x='center', parent=self.scene)
        self.labels.append(lbl)

        # Echo (γ-scaled toward origin)
        echo_pts = all_pts * self.engine.gamma + np.random.normal(scale=0.01, size=all_pts.shape)
        echo = Markers(pos=echo_pts, face_color=(1, 1, 1, 0.2),
                      size=4, parent=self.scene)
        self.echoes.append(echo)

        # Link to previous center
        if len(self.centers) > 1:
            prev_c = self.centers[-2]
            segs = []
            for p in all_pts[:min(6, len(all_pts))]:
                segs += [prev_c, p]
            link = Line(pos=np.array(segs), color=(0.7, 0.7, 0.7, 0.3),
                       width=1, connect='segments', parent=self.scene)
            self.links.append(link)

        # Spawn organelles based on k
        spawn_prob = 0.02 + k * 0.02
        cluster_size = 6 + k
        if np.random.rand() < spawn_prob:
            org = self.spawn_organelle(cen, color, size=0.3, n=cluster_size)
            self.organelles.append(org)

    def spawn_organelle(self, center, color_rgb, size=0.5, n=12):
        """Create organelle cluster"""
        pts = center + np.random.normal(scale=0.2, size=(n, 3)) * size
        rgba = list(Color(color_rgb).rgba)
        rgba[3] = 1.0
        mark = Markers(pos=pts, face_color=rgba, edge_color=None, size=5, parent=self.scene)
        return {"marker": mark, "positions": pts, "color": rgba}

    def update_organelles(self):
        """Update organelle positions with lattice backpressure"""
        if not self.organelles or not self.centers:
            return

        params = self.engine.nucleotide_to_params(int(self.tau))
        if params is None:
            return

        base = params['base']
        lattice_nodes = np.array(self.centers)

        for org in self.organelles:
            positive = base in ['A', 'T']
            strength = 0.02

            new_pts = []
            for p in org['positions']:
                if len(lattice_nodes) > 0:
                    nearest = lattice_nodes[np.random.randint(0, len(lattice_nodes))]
                    dir_vec = nearest - p
                    if not positive:
                        dir_vec *= -1
                    new_p = p + dir_vec * strength
                else:
                    new_p = p
                new_pts.append(new_p)

            org['positions'] = np.array(new_pts)
            org['marker'].set_data(pos=org['positions'], face_color=org['color'], size=5)

    def update_displays(self):
        """Update text displays"""
        percent = min(100 * self.tau / self.genome_len, 100)
        self.progress_text.text = f"{percent:.1f}%"

        params = self.engine.nucleotide_to_params(int(self.tau))
        if params:
            self.physics_text.text = (f"k={params['k']} | |Ψ|={params['psi_magnitude']:.2f} | "
                                     f"φ={params['psi_phase']:.2f}")

        self.total_structures = (len(self.rungs) + len(self.echoes) +
                                len(self.links) + len(self.labels) +
                                len(self.organelles))
        self.structure_text.text = f"Structures: {self.total_structures}"

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

class FastaFirstVisualizerApp:
    """Main application - genome is the clock!"""

    def __init__(self, fasta_file, division_interval=2000):
        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"FASTA not found: {fasta_file}")

        self.genome = load_genome(fasta_file)
        print(f"✓ Genome loaded: {len(self.genome)} nucleotides")

        # Initialize φ-recursive engine
        self.engine = PhiRecursiveEngine(self.genome)
        print(f"✓ φ-Recursive engine initialized")
        print(f"  Period P = {self.engine.P}")
        print(f"  Vertical scale ℓ = {self.engine.ell:.6f}")
        print(f"  Echo scale γ = {self.engine.gamma}")

        self.division_interval = division_interval

        # Statistics
        self.compute_statistics()

        # Create canvas
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(1600, 1000),
            bgcolor='black',
            title="FASTA-First φ-Framework Visualizer"
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.view.camera.distance = 50

        # Create initial cell
        self.cells = [GenomeDrivenCell(self.genome, self.engine,
                                       center_offset=np.array([0, 0, 0]),
                                       parent_scene=self.view.scene)]

        # Timer
        self.timer = app.Timer(interval=0.02, connect=self.update, start=True)

        # Info
        self.create_info_panel()
        self.frame = 0

    def compute_statistics(self):
        """Compute genome statistics"""
        print("\n" + "="*80)
        print("GENOME-SCALE φ-HARMONIC ANALYSIS")
        print("="*80)

        base_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        octave_counts = {i: 0 for i in range(8)}

        for base in self.genome[:10000]:  # Sample first 10k
            if base in base_counts:
                base_counts[base] += 1
                k = BASE_MAP[base] - 1
                octave_counts[k] += 1

        total = sum(base_counts.values())
        print(f"Base distribution (first 10k):")
        for base, count in base_counts.items():
            k = BASE_MAP[base] - 1
            geom = GEOMETRIES[k][3]
            print(f"  {base} → k={k} ({geom}): {count} ({100*count/total:.1f}%)")

        print("\nOctave coverage:")
        for k, count in octave_counts.items():
            if count > 0:
                geom = GEOMETRIES[k][3]
                print(f"  k={k} ({geom}): {count} occurrences")

        print("="*80 + "\n")

    def create_info_panel(self):
        """Create info overlay"""
        info_text = (
            "FASTA-FIRST φ-FRAMEWORK\n"
            f"φ = {PHI:.8f}\n"
            f"G = 2π/φ² = {GOLDEN_ANGLE_RAD:.6f}\n"
            f"Genome: {len(self.genome)} bases\n"
            "τ = genome position (the clock!)\n"
            "PERSISTENT: All structures remain"
        )

        self.info = Text(info_text, pos=(10, 30),
                        color='cyan', font_size=10,
                        anchor_x='left', anchor_y='top',
                        parent=self.canvas.scene)

        self.cell_count_text = Text("Active Cells: 1",
                                    pos=(10, 160),
                                    color='lime', font_size=14,
                                    anchor_x='left',
                                    parent=self.canvas.scene)

    def update(self, event):
        """Main update loop"""
        self.frame += 1

        # Update all cells
        new_cells = []
        for cell in self.cells:
            cell.update()

            # Division when cell completes genome cycle
            if cell.tau == 0 and cell.frame > self.division_interval:
                offset = np.random.normal(scale=5.0, size=3)
                daughter = GenomeDrivenCell(
                    self.genome,
                    self.engine,
                    center_offset=cell.center_offset + offset,
                    parent_scene=self.view.scene
                )
                new_cells.append(daughter)
                print(f"✓ Cell division at frame {self.frame}")

        self.cells.extend(new_cells)

        # Update displays
        if hasattr(self, 'cell_count_text'):
            total_structures = sum(c.total_structures for c in self.cells)
            self.cell_count_text.text = (f"Cells: {len(self.cells)} | "
                                        f"Structures: {total_structures}")

        # Camera
        self.view.camera.azimuth = self.frame * 0.2
        self.view.camera.elevation = 15 + 10 * np.sin(self.frame * 0.003)

    def run(self):
        """Start visualizer"""
        print("Starting FASTA-first visualizer...")
        self.canvas.show()
        app.run()

# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("FASTA-FIRST φ-FRAMEWORK VISUALIZER")
    print("Grand Master Equation: Genome IS the Clock")
    print("="*80 + "\n")

    fasta_path = "ecoli_k12.fasta"

    try:
        visualizer = FastaFirstVisualizerApp(fasta_path, division_interval=2000)
        visualizer.run()
    except FileNotFoundError:
        print("Creating demo genome...")
        synthetic = ['A', 'T', 'G', 'C'] * 250
        import random
        random.shuffle(synthetic)

        with open('demo_genome.fasta', 'w') as f:
            f.write(">Demo Sequence\n")
            for i in range(0, len(synthetic), 80):
                f.write(''.join(synthetic[i:i+80]) + '\n')

        print("✓ Created demo_genome.fasta")
        visualizer = FastaFirstVisualizerApp('demo_genome.fasta', division_interval=500)
        visualizer.run()

if __name__ == "__main__":
    main()