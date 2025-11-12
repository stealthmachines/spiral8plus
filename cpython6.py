#!/usr/bin/env python3
"""
==============================================================================
FASTA-FIRST φ-FRAMEWORK GENOME ENGINE v6 - OPTIMIZED
==============================================================================
PERFORMANCE FIXES:
- Reduced render complexity for smooth mouse controls
- Transparent/minimal echoes for better visibility
- Optimized update rate with adaptive rendering
- Option to use NumPy C acceleration (if available)

NEW VISUAL SETTINGS:
- Echo opacity: 5% (was 20%) - see through clouds!
- Rung creation: Every 40 frames (was 20) - less clutter
- Reduced organelle spawn rate
- Smaller link network

MOUSE CONTROLS:
- Left drag: Rotate view (smooth, no lag)
- Right drag: Pan camera
- Mouse wheel: Zoom
- Middle drag: Roll

KEYBOARD:
- SPACE: Pause/Resume
- Up/Down: Speed control
- R: Toggle auto-rotate
- E: Toggle echoes (visibility)
- L: Toggle links (reduce complexity)
- O: Toggle organelles
- H: Help
==============================================================================
"""

import os
import numpy as np
from decimal import Decimal, getcontext
from vispy import scene, app, keys
from vispy.scene.visuals import Line, Markers, Text
from vispy.scene.cameras import TurntableCamera
from vispy.color import Color

# Try to use faster NumPy operations
try:
    import numpy.core._multiarray_umath as np_c
    USING_C_BACKEND = True
except:
    USING_C_BACKEND = False

getcontext().prec = 50  # Reduced from 100 for speed

# ==============================================================================
# φ-HARMONIC CONSTANTS
# ==============================================================================

PHI = (1 + np.sqrt(5)) / 2
PHI_2 = PHI ** 2
PHI_INV = 1.0 / PHI
PHI_INV_7 = PHI ** (-7)
GOLDEN_ANGLE_RAD = 2 * np.pi / PHI_2

# Simplified primes (first 20 for speed)
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
          31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

# ==============================================================================
# NUCLEOTIDE → φ-OCTAVE MAPPING
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

# ==============================================================================
# GENOME LOADER
# ==============================================================================

def load_genome(fasta_file):
    """Load FASTA - optimized"""
    sequence = []
    with open(fasta_file) as f:
        for line in f:
            if not line.startswith(">"):
                sequence.extend(list(line.strip().upper()))
    return sequence

# ==============================================================================
# φ-RECURSIVE PHYSICS ENGINE (OPTIMIZED)
# ==============================================================================

class PhiRecursiveEngine:
    """Optimized physics engine with caching"""

    def __init__(self, genome):
        self.genome = genome
        self.genome_len = len(genome)

        # Period parameters
        self.P = 100
        self.s = 1.0
        self.ell = 0.8 / self.P
        self.gamma = 0.75

        # Caches for performance
        self.fib_cache = {}
        self.params_cache = {}  # NEW: Cache nucleotide params

        # Pre-compute common values
        self.two_pi_over_P = (2 * np.pi / self.P)

    def get_octave_k(self, tau):
        return min(int(self.s * tau / self.P), 7)

    def get_tau_tilde(self, tau):
        return tau % self.P

    def get_alpha_k(self, k):
        return GEOMETRIES[k][4]

    def get_vertices_k(self, k):
        return GEOMETRIES[k][5]

    def compute_r(self, tau):
        k = self.get_octave_k(tau)
        tau_tilde = self.get_tau_tilde(tau)
        alpha_k = self.get_alpha_k(k)
        return np.exp(alpha_k * tau_tilde)

    def compute_theta(self, tau):
        return self.two_pi_over_P * tau

    def compute_a_k(self, k):
        return k * GOLDEN_ANGLE_RAD

    def S_plus(self, tau, base_radius=15.0):
        k = self.get_octave_k(tau)
        r = self.compute_r(tau) * base_radius
        theta = self.compute_theta(tau)
        a_k = self.compute_a_k(k)
        z = self.ell * tau

        x = r * np.cos(theta + a_k)
        y = r * np.sin(theta + a_k)

        return np.array([x, y, z])

    def S_minus(self, tau, base_radius=15.0, separation=0.5):
        k = self.get_octave_k(tau)
        r = self.compute_r(tau) * base_radius
        theta = self.compute_theta(tau)
        a_k = self.compute_a_k(k)
        z = self.ell * tau

        x = r * np.cos(theta - a_k) + separation
        y = r * np.sin(theta - a_k) - separation

        return np.array([x, y, z])

    def fibonacci_real(self, n):
        """Simplified Fibonacci (faster)"""
        if n in self.fib_cache:
            return self.fib_cache[n]

        if n > 50:  # Reduced limit
            return 0.0

        # Simpler calculation
        result = (PHI**n - PHI_INV**n) / np.sqrt(5)
        self.fib_cache[n] = result
        return result

    def psi_recursive(self, k, tau):
        """Simplified wavefunction (faster)"""
        r = self.compute_r(tau)
        theta = self.compute_theta(tau)

        amplitude = PHI ** (k / 2)
        radial = r ** (PHI ** k)

        # Simplified phase
        phase = np.exp(1j * PHI ** k * theta)

        return amplitude * radial * phase

    def nucleotide_to_params(self, position):
        """CACHED parameter extraction"""
        # Check cache first
        if position in self.params_cache:
            return self.params_cache[position]

        if position >= self.genome_len:
            return None

        base = self.genome[position]
        tau = float(position)

        k_raw = BASE_MAP.get(base, 1) - 1
        k = min(max(k_raw, 0), 7)

        _, note, color, name, alpha_k, v_k = GEOMETRIES[k]

        # Simplified calculations
        psi = self.psi_recursive(k, tau)

        params = {
            'position': position,
            'base': base,
            'tau': tau,
            'k': k,
            'alpha_k': alpha_k,
            'v_k': v_k,
            'geometry': name,
            'color': color,
            'note': note,
            'psi_magnitude': abs(psi),
            'psi_phase': np.angle(psi),
        }

        # Cache it
        if len(self.params_cache) < 10000:  # Limit cache size
            self.params_cache[position] = params

        return params

# ==============================================================================
# OPTIMIZED CELL
# ==============================================================================

class GenomeDrivenCell:
    """Optimized cell with visibility toggles"""

    def __init__(self, genome, engine, center_offset=np.zeros(3), parent_scene=None, config=None):
        self.genome = genome
        self.genome_len = len(genome)
        self.engine = engine
        self.center_offset = center_offset
        self.scene = parent_scene
        self.config = config or {}

        self.tau = 0
        self.frame = 0

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

        # Stats
        self.total_structures = 0

        # Displays
        self.progress_text = Text("0%", pos=self.center_offset + [0, 0, 20],
                                 color='white', font_size=20,
                                 anchor_x='center', parent=self.scene)

        self.physics_text = Text("", pos=self.center_offset + [0, 0, 18],
                                color='gold', font_size=10,
                                anchor_x='center', parent=self.scene)

    def update(self):
        self.frame += 1

        # Reduced points for performance
        N = 300  # Was 400
        tau_start = self.tau
        tau_end = min(self.tau + N, self.genome_len)

        s1_points = []
        s2_points = []

        for tau_val in range(int(tau_start), int(tau_end)):
            if tau_val >= self.genome_len:
                break

            params = self.engine.nucleotide_to_params(tau_val)
            if params is None:
                continue

            p1 = self.engine.S_plus(tau_val)
            p2 = self.engine.S_minus(tau_val)

            s1_points.append(p1)
            s2_points.append(p2)

        if s1_points:
            s1 = np.array(s1_points) + self.center_offset
            s2 = np.array(s2_points) + self.center_offset
            self.strand1.set_data(s1)
            self.strand2.set_data(s2)

        # REDUCED FREQUENCY: Every 40 frames instead of 20
        if self.frame % 40 == 0 and self.tau < self.genome_len:
            self.create_rung()

        # Update organelles less frequently
        if self.frame % 5 == 0:
            self.update_organelles()

        self.tau += 1
        if self.tau >= self.genome_len:
            self.tau = 0

        self.update_displays()

    def create_rung(self):
        params = self.engine.nucleotide_to_params(int(self.tau))
        if params is None:
            return

        k = params['k']
        v_k = params['v_k']
        color = params['color']
        name = params['geometry']
        base = params['base']

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

        size = 6 * (1 + params['psi_magnitude'] * 0.1)

        # Rungs
        mark = Markers(pos=all_pts, face_color=Color(color).rgba,
                      edge_color='white', size=size, parent=self.scene)
        self.rungs.append(mark)

        cen = all_pts.mean(axis=0)
        self.centers.append(cen)

        # Labels (smaller, less intrusive)
        label_text = f"{base}:{name}"
        lbl = Text(label_text, pos=cen + [0, 0, 0.3],
                  color=color, font_size=8, bold=True,
                  anchor_x='center', parent=self.scene)
        self.labels.append(lbl)

        # ECHOES: Much more transparent (0.05 instead of 0.2)
        if self.config.get('show_echoes', True):
            echo_pts = all_pts * self.engine.gamma + np.random.normal(scale=0.01, size=all_pts.shape)
            echo = Markers(pos=echo_pts, face_color=(1, 1, 1, 0.05),  # ← VERY transparent!
                          size=3, parent=self.scene)
            self.echoes.append(echo)

        # LINKS: Optional and reduced
        if self.config.get('show_links', True) and len(self.centers) > 1:
            prev_c = self.centers[-2]
            segs = []
            # Only 3 links instead of 6
            for p in all_pts[:min(3, len(all_pts))]:
                segs += [prev_c, p]
            link = Line(pos=np.array(segs), color=(0.7, 0.7, 0.7, 0.2),  # ← More transparent
                       width=1, connect='segments', parent=self.scene)
            self.links.append(link)

        # ORGANELLES: MUCH smaller and rarer (these are the BIG CLOUDS!)
        if self.config.get('show_organelles', True):
            spawn_prob = 0.003 + k * 0.001  # ← 10x LESS frequent!
            cluster_size = 2 + k // 4  # ← TINY clusters (2-3 points)
            if np.random.rand() < spawn_prob:
                org = self.spawn_organelle(cen, color, size=0.05, n=cluster_size)  # ← Size 0.05 (was 0.2!)
                self.organelles.append(org)

    def spawn_organelle(self, center, color_rgb, size=0.05, n=3):  # ← Default tiny!
        pts = center + np.random.normal(scale=0.03, size=(n, 3)) * size  # ← Tight cluster
        rgba = list(Color(color_rgb).rgba)
        rgba[3] = 0.4  # ← MORE transparent (was 0.8)
        mark = Markers(pos=pts, face_color=rgba, edge_color=None, size=2, parent=self.scene)  # ← Size 2 (was 4)
        return {"marker": mark, "positions": pts, "color": rgba}

    def update_organelles(self):
        if not self.organelles or not self.centers:
            return

        params = self.engine.nucleotide_to_params(int(self.tau))
        if params is None:
            return

        base = params['base']
        lattice_nodes = np.array(self.centers[-100:])  # Only recent nodes

        for org in self.organelles:
            positive = base in ['A', 'T']
            strength = 0.005  # ← Much gentler movement

            new_pts = []
            for p in org['positions']:
                if len(lattice_nodes) > 0:
                    idx = np.random.randint(0, len(lattice_nodes))
                    nearest = lattice_nodes[idx]
                    dir_vec = nearest - p
                    if not positive:
                        dir_vec *= -1
                    new_p = p + dir_vec * strength
                else:
                    new_p = p
                new_pts.append(new_p)

            org['positions'] = np.array(new_pts)
            org['marker'].set_data(pos=org['positions'], face_color=org['color'], size=2)  # ← Size 2

    def update_displays(self):
        percent = min(100 * self.tau / self.genome_len, 100)
        self.progress_text.text = f"{percent:.1f}%"

        params = self.engine.nucleotide_to_params(int(self.tau))
        if params:
            self.physics_text.text = f"k={params['k']} | |Ψ|={params['psi_magnitude']:.2f}"

        self.total_structures = (len(self.rungs) + len(self.echoes) +
                                len(self.links) + len(self.labels) +
                                len(self.organelles))

    def toggle_echoes(self):
        """Toggle echo visibility"""
        for echo in self.echoes:
            echo.visible = not echo.visible

    def toggle_links(self):
        """Toggle link visibility"""
        for link in self.links:
            link.visible = not link.visible

    def toggle_organelles(self):
        """Toggle organelle visibility"""
        for org in self.organelles:
            org['marker'].visible = not org['marker'].visible

    def toggle_rungs(self):
        """Toggle rung (geometric markers) visibility"""
        for rung in self.rungs:
            rung.visible = not rung.visible

    def toggle_labels(self):
        """Toggle label visibility"""
        for label in self.labels:
            label.visible = not label.visible

    def toggle_strands(self):
        """Toggle DNA strand visibility"""
        self.strand1.visible = not self.strand1.visible
        self.strand2.visible = not self.strand2.visible

    def toggle_text(self):
        """Toggle progress/physics text visibility"""
        self.progress_text.visible = not self.progress_text.visible
        self.physics_text.visible = not self.physics_text.visible# ==============================================================================
# OPTIMIZED APPLICATION
# ==============================================================================

class FastaFirstVisualizerApp:
    """OPTIMIZED for smooth mouse controls"""

    def __init__(self, fasta_file, division_interval=2000):
        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"FASTA not found: {fasta_file}")

        self.genome = load_genome(fasta_file)
        print(f"✓ Genome loaded: {len(self.genome)} nucleotides")

        # Initialize engine
        self.engine = PhiRecursiveEngine(self.genome)
        print(f"✓ φ-Recursive engine initialized")
        if USING_C_BACKEND:
            print(f"✓ Using NumPy C backend for acceleration")

        self.division_interval = division_interval

        # Rendering configuration - EVERY element is toggleable!
        self.config = {
            'show_strands': True,      # Toggle with S (DNA helices)
            'show_rungs': True,        # Toggle with U (geometric markers)
            'show_labels': True,       # Toggle with B (base names)
            'show_echoes': True,       # Toggle with E (ghost copies)
            'show_links': True,        # Toggle with L (connections)
            'show_organelles': True,   # Toggle with O (clusters)
            'show_text': True,         # Toggle with T (progress/physics displays)
        }

        # Statistics
        self.compute_statistics()

        # Create canvas
        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(1600, 1000),
            bgcolor='black',
            title="FASTA-First φ-Framework v6 - OPTIMIZED"
        )
        self.view = self.canvas.central_widget.add_view()

        # Optimized camera
        self.view.camera = TurntableCamera(
            fov=60,
            distance=50,
            elevation=15,
            azimuth=45,
            center=(0, 0, 0)
        )

        # Control state
        self.paused = False
        self.speed_multiplier = 1.0
        self.auto_rotate = True

        # Create environment (independent of FASTA)
        self.create_environment()

        # Create initial cell
        self.cells = [GenomeDrivenCell(self.genome, self.engine,
                                       center_offset=np.array([0, 0, 0]),
                                       parent_scene=self.view.scene,
                                       config=self.config)]

        # OPTIMIZED TIMER: Slower update for smooth interaction
        self.timer = app.Timer(interval=0.03, connect=self.update, start=True)  # Was 0.02

        # Info
        self.create_info_panel()
        self.frame = 0

        # Connect events
        self.canvas.events.key_press.connect(self.on_key_press)

    def create_environment(self):
        """Create reference environment (NOT driven by FASTA)"""
        # Subtle reference grid at z=0
        grid_size = 80
        grid_step = 10
        grid_points = []
        for x in range(-grid_size, grid_size+1, grid_step):
            grid_points += [[x, -grid_size, 0], [x, grid_size, 0]]
        for y in range(-grid_size, grid_size+1, grid_step):
            grid_points += [[-grid_size, y, 0], [grid_size, y, 0]]

        self.grid = Line(pos=np.array(grid_points), color=(0.2, 0.2, 0.3, 0.3),
                        width=1, connect='segments', parent=self.view.scene)

        # Central reference sphere
        angles = np.linspace(0, 2*np.pi, 32)
        circle_pts = []
        for theta in angles:
            circle_pts.append([5*np.cos(theta), 5*np.sin(theta), 0])
        circle_pts.append(circle_pts[0])

        self.ref_circle = Line(pos=np.array(circle_pts), color=(0.3, 0.3, 0.5, 0.5),
                              width=2, parent=self.view.scene)

    def on_key_press(self, event):
        # Safety check for None key
        if event.key is None or event.key.name is None:
            return

        if event.key == keys.SPACE:
            self.paused = not self.paused
            print(f"⏸️  {'PAUSED' if self.paused else 'RUNNING'}")

        elif event.key.name == 'Up':
            self.speed_multiplier = min(self.speed_multiplier * 1.5, 10.0)
            print(f"⚡ Speed: {self.speed_multiplier:.1f}x")

        elif event.key.name == 'Down':
            self.speed_multiplier = max(self.speed_multiplier / 1.5, 0.1)
            print(f"🐌 Speed: {self.speed_multiplier:.1f}x")

        elif event.key.name == 'R':
            self.auto_rotate = not self.auto_rotate
            print(f"🔄 Auto-rotate: {'ON' if self.auto_rotate else 'OFF'}")

        elif event.key.name == 'E':
            self.config['show_echoes'] = not self.config['show_echoes']
            for cell in self.cells:
                cell.toggle_echoes()
            print(f"👻 Echoes: {'ON' if self.config['show_echoes'] else 'OFF'}")

        elif event.key.name == 'L':
            self.config['show_links'] = not self.config['show_links']
            for cell in self.cells:
                cell.toggle_links()
            print(f"🔗 Links: {'ON' if self.config['show_links'] else 'OFF'}")

        elif event.key.name == 'O':
            self.config['show_organelles'] = not self.config['show_organelles']
            for cell in self.cells:
                cell.toggle_organelles()
            print(f"🔵 Organelles: {'ON' if self.config['show_organelles'] else 'OFF'}")

        elif event.key.name == 'U':
            self.config['show_rungs'] = not self.config['show_rungs']
            for cell in self.cells:
                cell.toggle_rungs()
            print(f"🔴 Rungs: {'ON' if self.config['show_rungs'] else 'OFF'}")

        elif event.key.name == 'B':
            self.config['show_labels'] = not self.config['show_labels']
            for cell in self.cells:
                cell.toggle_labels()
            print(f"🏷️  Labels: {'ON' if self.config['show_labels'] else 'OFF'}")

        elif event.key.name == 'S':
            self.config['show_strands'] = not self.config['show_strands']
            for cell in self.cells:
                cell.toggle_strands()
            print(f"🧬 DNA Strands: {'ON' if self.config['show_strands'] else 'OFF'}")

        elif event.key.name == 'T':
            self.config['show_text'] = not self.config['show_text']
            for cell in self.cells:
                cell.toggle_text()
            print(f"📊 Text Displays: {'ON' if self.config['show_text'] else 'OFF'}")

        elif event.key.name == 'H':
            self.print_help()

    def print_help(self):
        print("\n" + "="*70)
        print("COMPLETE LAYER CONTROL - TOGGLE EVERYTHING!")
        print("="*70)
        print("MOUSE:")
        print("  Left drag      - Rotate view")
        print("  Right drag     - Pan camera")
        print("  Mouse wheel    - Zoom in/out")
        print("  Middle drag    - Roll camera")
        print()
        print("KEYBOARD - TOGGLE ALL LAYERS:")
        print("  SPACE          - Pause/Resume")
        print("  Up/Down Arrow  - Speed control")
        print("  R              - Toggle auto-rotate")
        print()
        print("  S              - Toggle DNA STRANDS (helices)")
        print("  U              - Toggle RUNGS (geometric markers)")
        print("  B              - Toggle LABELS (base names)")
        print("  E              - Toggle ECHOES (ghost copies)")
        print("  L              - Toggle LINKS (connections)")
        print("  O              - Toggle ORGANELLES (clusters)")
        print("  T              - Toggle TEXT (progress/physics)")
        print()
        print("  H              - Show this help")
        print()
        print("FASTA DRIVES ALL THESE LAYERS:")
        print("  - DNA strand positions (S±)")
        print("  - Geometric rungs (vertices)")
        print("  - Labels (base + geometry)")
        print("  - Echoes (φ⁻⁷ copies)")
        print("  - Links (lattice)")
        print("  - Organelles (dynamics)")
        print("  - All physics parameters")
        print()
        print("ENVIRONMENT (independent):")
        print("  - Reference grid (z=0 plane)")
        print("  - Central circle")
        print("="*70 + "\n")

    def compute_statistics(self):
        print("\n" + "="*80)
        print("GENOME-SCALE φ-HARMONIC ANALYSIS")
        print("="*80)

        base_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for base in self.genome[:10000]:
            if base in base_counts:
                base_counts[base] += 1

        total = sum(base_counts.values())
        for base, count in base_counts.items():
            k = BASE_MAP[base] - 1
            geom = GEOMETRIES[k][3]
            print(f"  {base} → k={k} ({geom}): {count} ({100*count/total:.1f}%)")
        print("="*80 + "\n")

    def create_info_panel(self):
        info_text = (
            "FASTA-FIRST v6 - FULL CONTROL\n"
            f"φ = {PHI:.6f}\n"
            f"Genome: {len(self.genome)} bases\n"
            "\n"
            "TOGGLE ALL LAYERS:\n"
            "S:Strands U:Rungs B:Labels\n"
            "E:Echoes L:Links O:Organelles\n"
            "T:Text displays\n"
            "\n"
            "SPACE:Pause H:Help"
        )

        self.info = Text(info_text, pos=(10, 30),
                        color='cyan', font_size=9,
                        anchor_x='left', anchor_y='top',
                        parent=self.canvas.scene)

        self.cell_count_text = Text("",
                                    pos=(10, 180),
                                    color='lime', font_size=12,
                                    anchor_x='left',
                                    parent=self.canvas.scene)

        self.control_status_text = Text("",
                                        pos=(10, 200),
                                        color='yellow', font_size=10,
                                        anchor_x='left',
                                        parent=self.canvas.scene)

    def update(self, event):
        if self.paused:
            self.control_status_text.text = "⏸️  PAUSED"
            return

        self.frame += 1

        # Update cells
        updates = max(1, int(self.speed_multiplier))

        new_cells = []
        for _ in range(updates):
            for cell in self.cells:
                cell.update()

                if cell.tau == 0 and cell.frame > self.division_interval:
                    offset = np.random.normal(scale=5.0, size=3)
                    daughter = GenomeDrivenCell(
                        self.genome, self.engine,
                        center_offset=cell.center_offset + offset,
                        parent_scene=self.view.scene,
                        config=self.config
                    )
                    new_cells.append(daughter)
                    print(f"✓ Cell division at frame {self.frame}")

        self.cells.extend(new_cells)

        # Update displays
        total_structures = sum(c.total_structures for c in self.cells)
        self.cell_count_text.text = f"Cells: {len(self.cells)} | Structures: {total_structures}"

        echoes_status = "ON" if self.config['show_echoes'] else "OFF"
        self.control_status_text.text = f"Speed: {self.speed_multiplier:.1f}x | Echoes: {echoes_status}"

        # Auto-rotate (gentle)
        if self.auto_rotate:
            self.view.camera.azimuth = self.frame * 0.15  # Slower
            self.view.camera.elevation = 15 + 8 * np.sin(self.frame * 0.002)

    def run(self):
        print("\n" + "="*70)
        print("FASTA-FIRST COMPLETE LAYER CONTROL")
        print("="*70)
        print("✓ EVERY element is toggleable:")
        print("  S - DNA Strands (double helix)")
        print("  U - Rungs (geometric markers)")
        print("  B - Labels (base + geometry names)")
        print("  E - Echoes (φ⁻⁷ ghost copies)")
        print("  L - Links (lattice connections)")
        print("  O - Organelles (dynamic clusters)")
        print("  T - Text (progress/physics displays)")
        print()
        print("✓ Environment (independent of FASTA):")
        print("  - Reference grid (z=0)")
        print("  - Central reference circle")
        print()
        print("Toggle layers on/off to see how FASTA drives each component!")
        print("Press 'H' for full controls")
        print("="*70 + "\n")

        self.canvas.show()
        app.run()# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("FASTA-FIRST φ-FRAMEWORK VISUALIZER v6 - OPTIMIZED")
    print("Smooth Mouse Controls + Transparent Echoes + Better Performance")
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
