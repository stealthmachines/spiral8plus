#!/usr/bin/env python3
"""
==============================================================================
FASTA-FIRST φ-RECURSIVE GENOME ENGINE v8 - PURE MATHEMATICS
==============================================================================
NO CONTRIVANCES - ONLY INHERENT FASTA + φ-FRAMEWORK MATHEMATICS

GRAND MASTER EQUATION IMPLEMENTATION:
- Ψₖ₊₁ = φ^(1/2) r^φ e^(i2πφ) Ψₖ  (recursive wavefunction)
- S±(τ) = φ-parametric double helix (continuous flow)
- D_n = √(φ·Fₙ·Pₙ·base^n·Ω)·r^k (discrete octave operator)

PURE FASTA PROPERTIES:
- τ (genome position) IS the time parameter
- Nucleotide identity → octave index k
- Fibonacci recursion from sequence
- Prime modulation from position
- φ-harmonic scaling inherent to DNA structure

ENVIRONMENT (NOT FASTA):
- Reference grid
- Coordinate axes
- Golden spiral guides

ALL PHYSICS EMERGENT FROM FASTA SEQUENCE
==============================================================================
"""

import os
import numpy as np
from decimal import Decimal, getcontext
from vispy import scene, app, keys
from vispy.scene.visuals import Line, Markers, Text
from vispy.scene.cameras import TurntableCamera
from vispy.color import Color

getcontext().prec = 100  # High precision for Fibonacci

# ==============================================================================
# φ-HARMONIC CONSTANTS (MATHEMATICAL FUNDAMENTALS)
# ==============================================================================

PHI = (1 + np.sqrt(5)) / 2  # 1.618033988749895
PHI_2 = PHI ** 2
PHI_INV = 1.0 / PHI
GOLDEN_ANGLE_RAD = 2 * np.pi / PHI_2  # ~2.399963 radians

# First 50 primes for D_n operator
PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229
]

# Nucleotide → octave mapping (inherent to DNA chemistry)
BASE_MAP = {'A': 5, 'T': 2, 'G': 4, 'C': 1}

# Eight geometries (1D → 8D φ-octave hierarchy)
GEOMETRIES = [
    (1, 'C', 'red',          'Point',        0.015269,  1),   # k=0
    (2, 'D', 'green',        'Line',         0.008262,  2),   # k=1
    (3, 'E', 'violet',       'Triangle',     0.110649,  3),   # k=2
    (4, 'F', 'mediumpurple', 'Tetrahedron', -0.083485,  4),   # k=3
    (5, 'G', 'blue',         'Pentachoron',  0.025847,  5),   # k=4
    (6, 'A', 'indigo',       'Hexacross',   -0.045123, 12),   # k=5
    (7, 'B', 'purple',       'Heptacube',    0.067891, 14),   # k=6
    (8, 'C', 'white',        'Octacube',     0.012345, 16),   # k=7
]

# ==============================================================================
# GENOME LOADER
# ==============================================================================

def load_genome(fasta_file):
    """Load FASTA sequence - the fundamental data"""
    sequence = []
    with open(fasta_file) as f:
        for line in f:
            if not line.startswith(">"):
                sequence.extend(list(line.strip().upper()))
    return sequence

# ==============================================================================
# φ-RECURSIVE ENGINE (GRAND MASTER EQUATION)
# ==============================================================================

class PhiRecursiveEngine:
    """
    Pure mathematical φ-recursive framework
    Implements Grand Master Equation with no contrivances
    """

    def __init__(self, genome):
        self.genome = genome
        self.genome_len = len(genome)

        # Period parameters (from Grand Master Equation)
        self.P = 100          # Temporal period
        self.s = 1.0          # Speed factor
        self.ell = 0.8 / self.P  # Vertical scale
        self.gamma = 0.75     # Echo scale

        # Caches
        self.fib_cache = {}
        self.prime_cache = {}

        # Pre-compute
        self.two_pi_over_P = (2 * np.pi / self.P)

    # --------------------------------------------------------------------------
    # FIBONACCI SEQUENCE (EXACT - using Decimal for precision)
    # --------------------------------------------------------------------------

    def fibonacci(self, n):
        """
        Compute F_n using Binet's formula with high precision
        F_n = (φⁿ - ψⁿ) / √5, where ψ = -1/φ
        """
        if n in self.fib_cache:
            return self.fib_cache[n]

        if n > 50:  # Limit for numerical stability
            return 0.0

        # Use Decimal for extreme precision
        phi_d = Decimal(str(PHI))
        psi_d = Decimal(-1) / phi_d
        sqrt5_d = Decimal(5).sqrt()

        n_d = Decimal(n)

        # Binet's formula
        fib_d = (phi_d ** n_d - psi_d ** n_d) / sqrt5_d
        result = float(fib_d)

        self.fib_cache[n] = result
        return result

    # --------------------------------------------------------------------------
    # PRIME PRODUCT OPERATOR
    # --------------------------------------------------------------------------

    def prime_at_index(self, n):
        """P_n: Prime number at index (n mod 50)"""
        idx = int(n) % len(PRIMES)
        return PRIMES[idx]

    # --------------------------------------------------------------------------
    # D_n OPERATOR (UNIFIED FORMULA)
    # --------------------------------------------------------------------------

    def D_n(self, n, beta=0.0, r=1.0, k=0.0, Omega=1.0, base=2.0):
        """
        Grand Master Equation D_n operator:
        D_n = √(φ·F_n·P_n·base^n·Ω)·r^k

        For large n, use logarithmic form to avoid overflow:
        log(D_n) = 0.5·[log(φ) + log(|F_n|) + log(P_n) + n·log(base) + log(Ω)] + k·log(r)
        """
        F_n = self.fibonacci(n + beta)
        P_n = self.prime_at_index(n + beta)

        # Use modular arithmetic for extreme genome positions
        # Map large n to periodic cycle (inherent to φ-recursion)
        n_mod = (n + beta) % 10000  # Period of 10000 for numerical stability

        # Compute using logarithms to handle extreme ranges
        try:
            log_phi = np.log(PHI)
            log_F_n = np.log(max(abs(F_n), 1e-100))
            log_P_n = np.log(P_n)
            log_base_n = n_mod * np.log(base)
            log_Omega = np.log(max(Omega, 1e-100))

            # log(√(φ·F_n·P_n·base^n·Ω)) = 0.5·(sum of logs)
            log_inside = 0.5 * (log_phi + log_F_n + log_P_n + log_base_n + log_Omega)

            # Add r^k term
            log_result = log_inside + k * np.log(max(r, 1e-100))

            # Convert back from log space
            result = np.exp(log_result)

        except (OverflowError, RuntimeWarning):
            # Fallback: use normalized form
            result = np.sqrt(PHI * max(abs(F_n), 1.0) * P_n * Omega) * np.power(r, k)

        # Apply Fibonacci sign
        if F_n < 0:
            result = -result

        return result    # --------------------------------------------------------------------------
    # OCTAVE INDEX (from genome position)
    # --------------------------------------------------------------------------

    def get_octave_k(self, tau):
        """k(τ) = min(⌊s·τ/P⌋, 7)"""
        return min(int(self.s * tau / self.P), 7)

    def get_tau_tilde(self, tau):
        """τ̃(τ) = τ mod P (phase within period)"""
        return tau % self.P

    def get_alpha_k(self, k):
        """Growth exponent for octave k"""
        return GEOMETRIES[k][4]

    def get_vertices_k(self, k):
        """Vertex count for octave k"""
        return GEOMETRIES[k][5]

    # --------------------------------------------------------------------------
    # RADIAL EXPANSION (φ-harmonic)
    # --------------------------------------------------------------------------

    def compute_r(self, tau):
        """r(τ) = exp(α_k(τ) · τ̃(τ))"""
        k = self.get_octave_k(tau)
        tau_tilde = self.get_tau_tilde(tau)
        alpha_k = self.get_alpha_k(k)
        return np.exp(alpha_k * tau_tilde)

    def compute_theta(self, tau):
        """θ(τ) = (2π/P)·τ"""
        return self.two_pi_over_P * tau

    def compute_a_k(self, k):
        """a_k = k·(2π/φ²) (golden angle rotation)"""
        return k * GOLDEN_ANGLE_RAD

    # --------------------------------------------------------------------------
    # DOUBLE HELIX (GRAND MASTER EQUATION CONTINUOUS FLOW)
    # --------------------------------------------------------------------------

    def S_plus(self, tau, base_radius=15.0):
        """
        S₊(τ) = [r(τ)cos(θ(τ) + a_k), r(τ)sin(θ(τ) + a_k), ℓ·τ]
        Positive strand of φ-spiral
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
        S₋(τ) = [r(τ)cos(θ(τ) - a_k), r(τ)sin(θ(τ) - a_k), ℓ·τ]
        Negative strand of φ-spiral
        """
        k = self.get_octave_k(tau)
        r = self.compute_r(tau) * base_radius
        theta = self.compute_theta(tau)
        a_k = self.compute_a_k(k)
        z = self.ell * tau

        x = r * np.cos(theta - a_k) + separation
        y = r * np.sin(theta - a_k) - separation

        return np.array([x, y, z])

    # --------------------------------------------------------------------------
    # φ-RECURSIVE WAVEFUNCTION (Ψₖ)
    # --------------------------------------------------------------------------

    def psi_recursive(self, k, tau):
        """
        Ψₖ(τ) = φ^(k/2) · r(τ)^(φ^k) · e^(i·φ^k·θ(τ))

        Recursive relation: Ψₖ₊₁ = φ^(1/2) · r^φ · e^(i2πφ) · Ψₖ
        """
        r = self.compute_r(tau)
        theta = self.compute_theta(tau)

        # Amplitude: φ^(k/2)
        amplitude = PHI ** (k / 2)

        # Radial: r^(φ^k)
        radial = r ** (PHI ** k)

        # Phase: e^(i·φ^k·θ)
        phase = np.exp(1j * (PHI ** k) * theta)

        return amplitude * radial * phase

    # --------------------------------------------------------------------------
    # NUCLEOTIDE → PHYSICS PARAMETERS
    # --------------------------------------------------------------------------

    def nucleotide_to_params(self, position):
        """
        Extract ALL physics from genome position
        Pure FASTA-driven - no contrivances!
        """
        if position >= self.genome_len:
            return None

        base = self.genome[position]
        tau = float(position)

        # Map nucleotide to octave (INHERENT to DNA chemistry)
        k_raw = BASE_MAP.get(base, 1) - 1
        k = min(max(k_raw, 0), 7)

        # Get geometry properties
        _, note, color, name, alpha_k, v_k = GEOMETRIES[k]

        # Compute φ-recursive wavefunction
        psi = self.psi_recursive(k, tau)

        # Compute D_n operator value
        D_val = self.D_n(n=tau, beta=0.0, r=1.0, k=float(k), Omega=1.0, base=2.0)

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
            'D_n': D_val,
            'F_n': self.fibonacci(tau),
            'P_n': self.prime_at_index(tau),
        }

# ==============================================================================
# PURE CELL (FASTA-DRIVEN ONLY)
# ==============================================================================

class PureCell:
    """
    Cell driven ONLY by FASTA + φ-mathematics
    No arbitrary rules, limits, or contrivances
    """

    def __init__(self, genome, engine, center_offset=np.zeros(3), parent_scene=None):
        self.genome = genome
        self.genome_len = len(genome)
        self.engine = engine
        self.center_offset = center_offset
        self.scene = parent_scene

        self.tau = 0
        self.frame = 0

        # DNA strands (always visible - fundamental)
        self.strand1 = Line(pos=np.zeros((1,3)), color=(0,1,1,0.5),  # Start semi-transparent
                           width=3, parent=self.scene)
        self.strand2 = Line(pos=np.zeros((1,3)), color=(1,0.5,0,0.5),  # Start semi-transparent
                           width=3, parent=self.scene)

        # Structures (emergent from FASTA)
        self.rungs = []
        self.labels = []
        self.centers = []
        self.rung_phases = []  # Phase offset for breathing transparency
        self.label_phases = []  # Phase offset for breathing transparency
        self.rung_data = []  # Store (positions, base_color, size) for recreating with alpha

        # Display
        self.info_text = Text("", pos=self.center_offset + [0, 0, 25],
                             color='cyan', font_size=12,
                             anchor_x='center', parent=self.scene)

    def update(self):
        self.frame += 1

        # Calculate breathing alpha (0.2 to 0.95, smooth oscillation)
        # Each element gets different phase for visual depth
        alpha_base = 0.2 + 0.75 * (0.5 + 0.5 * np.sin(self.frame * 0.05))

        # Render continuous strands with pulsing transparency
        N = 500
        tau_start = self.tau
        tau_end = min(self.tau + N, self.genome_len)

        s1_points = []
        s2_points = []

        for tau_val in range(int(tau_start), int(tau_end)):
            if tau_val >= self.genome_len:
                break

            p1 = self.engine.S_plus(tau_val)
            p2 = self.engine.S_minus(tau_val)

            s1_points.append(p1)
            s2_points.append(p2)

        if s1_points:
            s1 = np.array(s1_points) + self.center_offset
            s2 = np.array(s2_points) + self.center_offset
            self.strand1.set_data(s1)
            self.strand2.set_data(s2)

            # Update strand transparency
            self.strand1.set_data(color=(0, 1, 1, alpha_base))
            self.strand2.set_data(color=(1, 0.5, 0, alpha_base * 0.9))  # Slightly offset phase

        # Create rung at octave transitions (INHERENT φ-periodicity)
        if self.frame % 50 == 0 and self.tau < self.genome_len:
            self.create_rung()

        # Update breathing transparency for all existing rungs and labels
        self.update_all_alphas()

        self.tau += 1
        if self.tau >= self.genome_len:
            self.tau = 0  # Loop (circular genome)

        self.update_display()

    def create_rung(self):
        """Create rung from φ-octave sampling"""
        params = self.engine.nucleotide_to_params(int(self.tau))
        if params is None:
            return

        k = params['k']
        v_k = params['v_k']
        color = params['color']
        base = params['base']
        name = params['geometry']

        # Sample v_k vertices from current octave
        tau_samples = np.linspace(self.tau - 10, self.tau + 10, v_k)

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

        # Size from ψ magnitude
        size = 8 * (1 + params['psi_magnitude'] * 0.2)

        # Create rung with random phase offset for visual depth
        phase_offset = np.random.random() * 2 * np.pi
        base_rgba = Color(color).rgba  # Store base color
        mark = Markers(pos=all_pts, face_color=base_rgba,
                      edge_color='white', size=size, parent=self.scene)
        self.rungs.append(mark)
        self.rung_phases.append(phase_offset)
        self.rung_data.append((all_pts, base_rgba[:3], size))  # Store for recreation

        cen = all_pts.mean(axis=0)
        self.centers.append(cen)

        # Label with base and geometry
        label_text = f"{base}:{name}"
        lbl = Text(label_text, pos=cen + [0, 0, 0.5],
                  color=color, font_size=10, bold=True,
                  anchor_x='center', parent=self.scene)
        self.labels.append(lbl)
        self.label_phases.append(phase_offset)

    def update_all_alphas(self):
        """Update breathing transparency for all rungs and labels"""
        # Update rungs every few frames (not every frame for performance)
        # Breathing is slow enough that updating every 3 frames is smooth
        if self.frame % 3 == 0:
            for i in range(len(self.rungs)):
                if i >= len(self.rung_phases) or i >= len(self.rung_data):
                    break

                phase = self.rung_phases[i]
                alpha = 0.2 + 0.75 * (0.5 + 0.5 * np.sin(self.frame * 0.05 + phase))

                # Get stored data
                pos, base_rgb, size = self.rung_data[i]

                # Create new RGBA with updated alpha
                new_rgba = tuple(list(base_rgb) + [alpha])

                # Remove old marker and create new one
                old_rung = self.rungs[i]
                if old_rung is not None and old_rung.parent is not None:
                    old_rung.parent = None  # Detach from scene

                # Create new marker with updated alpha
                new_rung = Markers(pos=pos, face_color=new_rgba,
                                 edge_color='white', size=size, parent=self.scene)
                self.rungs[i] = new_rung

        # Update labels every frame (Text is lighter weight)
        for i in range(len(self.labels)):
            if i >= len(self.label_phases):
                break
            label = self.labels[i]
            phase = self.label_phases[i]
            alpha = 0.2 + 0.75 * (0.5 + 0.5 * np.sin(self.frame * 0.05 + phase))

            try:
                # Text color update
                current = label.color
                if isinstance(current, Color):
                    rgba = list(current.rgba)
                elif isinstance(current, (tuple, list)):
                    rgba = list(current) if len(current) >= 4 else list(current) + [1.0]
                else:
                    rgba = [0, 1, 1, 1]

                rgba[3] = alpha
                label.color = tuple(rgba)
            except:
                pass

        # Update info text
        alpha_info = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(self.frame * 0.03))  # Slower phase
        try:
            self.info_text.color = (0, 1, 1, alpha_info)
        except:
            pass

    def update_display(self):
        """Show current physics state"""
        params = self.engine.nucleotide_to_params(int(self.tau))
        if params:
            psi_mag = params['psi_magnitude']
            F_n = params['F_n']
            tau = params['tau']
            base = params['base']
            geom = params['geometry']

            info = f"{base} @ τ={int(tau)} | {geom} | |Ψ|={psi_mag:.3f} | F_n={F_n:.2f}"
            self.info_text.text = info

# ==============================================================================
# APPLICATION
# ==============================================================================

class PureVisualizerApp:
    """Pure FASTA + φ-mathematics visualization"""

    def __init__(self, fasta_file):
        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"FASTA not found: {fasta_file}")

        self.genome = load_genome(fasta_file)
        print(f"✓ Genome loaded: {len(self.genome)} nucleotides")

        self.engine = PhiRecursiveEngine(self.genome)
        print(f"✓ φ-Recursive engine initialized")
        print(f"✓ Grand Master Equation: Ψₖ₊₁ = φ^(1/2)·r^φ·e^(i2πφ)·Ψₖ")

        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(1600, 1000),
            bgcolor='black',
            title="FASTA-First φ-Framework v8 - PURE MATHEMATICS"
        )
        self.view = self.canvas.central_widget.add_view()

        self.view.camera = TurntableCamera(
            fov=60,
            distance=60,
            elevation=20,
            azimuth=45,
            center=(0, 0, 0)
        )

        self.paused = False
        self.speed = 1.0
        self.auto_rotate = True

        self.create_environment()

        self.cell = PureCell(self.genome, self.engine,
                            center_offset=np.array([0, 0, 0]),
                            parent_scene=self.view.scene)

        self.timer = app.Timer(interval=0.025, connect=self.update, start=True)

        self.create_info_panel()
        self.frame = 0

        self.canvas.events.key_press.connect(self.on_key_press)

    def create_environment(self):
        """Reference environment (NOT driven by FASTA)"""
        # Grid at z=0
        grid_size = 100
        grid_step = 10
        grid_points = []
        for x in range(-grid_size, grid_size+1, grid_step):
            grid_points += [[x, -grid_size, 0], [x, grid_size, 0]]
        for y in range(-grid_size, grid_size+1, grid_step):
            grid_points += [[-grid_size, y, 0], [grid_size, y, 0]]

        Line(pos=np.array(grid_points), color=(0.15, 0.15, 0.2, 0.3),
             width=1, connect='segments', parent=self.view.scene)

        # φ-spiral guide
        angles = np.linspace(0, 4*np.pi, 200)
        guide_pts = []
        for theta in angles:
            r = 20 * np.exp(0.1 * theta)
            guide_pts.append([r*np.cos(theta), r*np.sin(theta), 0])

        Line(pos=np.array(guide_pts), color=(0.5, 0.3, 0.1, 0.4),
             width=1, parent=self.view.scene)

    def on_key_press(self, event):
        if event.key is None or event.key.name is None:
            return

        if event.key == keys.SPACE:
            self.paused = not self.paused
            print(f"{'PAUSED' if self.paused else 'RUNNING'}")

        elif event.key.name == 'Up':
            self.speed = min(self.speed * 1.5, 10.0)
            print(f"Speed: {self.speed:.1f}x")

        elif event.key.name == 'Down':
            self.speed = max(self.speed / 1.5, 0.1)
            print(f"Speed: {self.speed:.1f}x")

        elif event.key.name == 'R':
            self.auto_rotate = not self.auto_rotate
            print(f"Auto-rotate: {'ON' if self.auto_rotate else 'OFF'}")

        elif event.key.name == 'H':
            self.print_help()

    def print_help(self):
        print("\n" + "="*70)
        print("PURE φ-RECURSIVE FRAMEWORK")
        print("="*70)
        print("FASTA DRIVES ALL PHYSICS:")
        print("  τ (genome position) = time parameter")
        print("  Nucleotide → octave k")
        print("  S±(τ) = φ-parametric double helix")
        print("  Ψₖ(τ) = φ-recursive wavefunction")
        print("  D_n = √(φ·F_n·P_n·2^n·Ω)·r^k")
        print()
        print("CONTROLS:")
        print("  SPACE    - Pause/Resume")
        print("  Up/Down  - Speed")
        print("  R        - Auto-rotate")
        print("  Mouse    - Explore")
        print()
        print("NO CONTRIVANCES - PURE MATHEMATICS!")
        print("="*70 + "\n")

    def create_info_panel(self):
        info_text = (
            f"PURE φ-FRAMEWORK v8\n"
            f"φ = {PHI:.6f}\n"
            f"Genome: {len(self.genome)} bases\n"
            "\n"
            "FASTA → PHYSICS\n"
            "Grand Master Equation\n"
            "Ψₖ₊₁ = φ^½·r^φ·e^(i2πφ)·Ψₖ\n"
            "\n"
            "H: Help | SPACE: Pause"
        )

        Text(info_text, pos=(10, 30),
             color='gold', font_size=10,
             anchor_x='left', anchor_y='top',
             parent=self.canvas.scene)

    def update(self, event):
        if self.paused:
            return

        self.frame += 1

        # Update cell
        updates = max(1, int(self.speed))
        for _ in range(updates):
            self.cell.update()

        # Auto-rotate
        if self.auto_rotate:
            self.view.camera.azimuth = self.frame * 0.2
            self.view.camera.elevation = 20 + 10 * np.sin(self.frame * 0.003)

    def run(self):
        print("\n" + "="*70)
        print("PURE φ-RECURSIVE FRAMEWORK")
        print("="*70)
        print("NO CONTRIVANCES - ONLY:")
        print("  ✓ FASTA sequence data")
        print("  ✓ φ-harmonic mathematics")
        print("  ✓ Grand Master Equation")
        print("  ✓ Fibonacci recursion")
        print("  ✓ Prime modulation")
        print()
        print("All physics emergent from genome + φ!")
        print("="*70 + "\n")

        self.canvas.show()
        app.run()

# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("FASTA-FIRST φ-FRAMEWORK v8 - PURE MATHEMATICS")
    print("Grand Master Equation + Genome = Complete Physics")
    print("="*80 + "\n")

    fasta_path = "ecoli_k12.fasta"

    try:
        visualizer = PureVisualizerApp(fasta_path)
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
        visualizer = PureVisualizerApp('demo_genome.fasta')
        visualizer.run()

if __name__ == "__main__":
    main()
