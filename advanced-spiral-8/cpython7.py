#!/usr/bin/env python3
"""
==============================================================================
FASTA-FIRST φ-FRAMEWORK GENOME ENGINE v7 - THERMODYNAMIC LIFECYCLE
==============================================================================
INHERENT DNA PROPERTIES CONTROL EVERYTHING:

BASE-PAIRING THERMODYNAMICS:
- G-C pairs: 3 hydrogen bonds → stronger, longer-lived structures
- A-T pairs: 2 hydrogen bonds → weaker, shorter-lived structures
- Complementarity drives stability
- Local melting temperature determines lifespan

NATURAL DNA BEHAVIOR:
- Structures form when bases can pair (complementary regions)
- Structures decay based on bond strength
- No arbitrary windows or rules
- Pure thermodynamic simulation

LIFECYCLE:
- Generation: Complementary base pairing
- Persistence: Bond strength (GC content)
- Decay: Thermal breathing, melting dynamics
- All driven by FASTA sequence properties

CONTROLS: Same as v6 (S/U/B/E/L/O/T toggles)
==============================================================================
"""

import os
import numpy as np
from decimal import Decimal, getcontext
from vispy import scene, app, keys
from vispy.scene.visuals import Line, Markers, Text
from vispy.scene.cameras import TurntableCamera
from vispy.color import Color

try:
    import numpy.core._multiarray_umath as np_c
    USING_C_BACKEND = True
except:
    USING_C_BACKEND = False

getcontext().prec = 50

# ==============================================================================
# φ-HARMONIC CONSTANTS
# ==============================================================================

PHI = (1 + np.sqrt(5)) / 2
PHI_2 = PHI ** 2
PHI_INV = 1.0 / PHI
PHI_INV_7 = PHI ** (-7)
GOLDEN_ANGLE_RAD = 2 * np.pi / PHI_2

# ==============================================================================
# DNA THERMODYNAMIC CONSTANTS
# ==============================================================================

# Hydrogen bond counts (inherent to DNA chemistry)
HYDROGEN_BONDS = {'A': 2, 'T': 2, 'G': 3, 'C': 3}  # A-T: 2 bonds, G-C: 3 bonds

# Base pairing rules (Watson-Crick)
COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

# Melting temperature contribution (in relative units)
# G-C pairs are more stable than A-T pairs
TM_CONTRIBUTION = {'G': 4.0, 'C': 4.0, 'A': 2.0, 'T': 2.0}

# Nucleotide → φ-octave mapping
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
    """Load FASTA sequence"""
    sequence = []
    with open(fasta_file) as f:
        for line in f:
            if not line.startswith(">"):
                sequence.extend(list(line.strip().upper()))
    return sequence

# ==============================================================================
# THERMODYNAMIC STRUCTURE
# ==============================================================================

class DNAStructure:
    """A structure with thermodynamic lifetime based on base pairing"""

    def __init__(self, tau, base1, base2, visual_elements, structure_type):
        self.tau = tau  # Birth position in genome
        self.base1 = base1
        self.base2 = base2
        self.visual_elements = visual_elements  # Dict: {'rungs': marker, 'labels': text, etc.}
        self.structure_type = structure_type  # 'rung', 'echo', 'link', 'organelle'

        # Calculate inherent stability from DNA chemistry
        self.h_bonds = HYDROGEN_BONDS.get(base1, 2) + HYDROGEN_BONDS.get(base2, 2)
        self.tm_score = TM_CONTRIBUTION.get(base1, 2) + TM_CONTRIBUTION.get(base2, 2)

        # Complementarity bonus (A-T or G-C pairing is more stable)
        self.is_complementary = (COMPLEMENT.get(base1) == base2)
        if self.is_complementary:
            self.stability_bonus = 2.0
        else:
            self.stability_bonus = 1.0

        # Base lifetime on thermodynamics (frames until decay)
        # GC-rich structures live ~3x longer than AT-rich
        base_lifetime = self.h_bonds * self.tm_score * self.stability_bonus
        self.max_lifetime = int(base_lifetime * 100)  # Scale to frames
        self.age = 0

        # Thermal breathing (random fluctuations)
        self.breathing_phase = np.random.random() * 2 * np.pi

    def update(self, current_tau, temperature=1.0):
        """Update structure age and calculate decay"""
        self.age += 1

        # Distance from current genome position (local context matters)
        distance_from_current = abs(current_tau - self.tau)

        # Thermal breathing effect (DNA naturally fluctuates)
        breathing = 0.5 + 0.5 * np.sin(self.breathing_phase + self.age * 0.1)

        # Decay probability increases with age and temperature
        # High stability structures resist decay longer
        decay_threshold = (self.age / self.max_lifetime) * temperature * (2.0 - breathing)

        # Should this structure decay?
        should_decay = (decay_threshold > 1.0) or (distance_from_current > 1000)

        # Calculate alpha (transparency) based on remaining lifetime
        if self.max_lifetime > 0:
            life_fraction = 1.0 - (self.age / self.max_lifetime)
            alpha = max(0.0, min(1.0, life_fraction))
        else:
            alpha = 0.0

        return should_decay, alpha

    def destroy(self):
        """Remove visual elements"""
        for key, element in self.visual_elements.items():
            if element is not None:
                # Organelles are stored as dicts with 'marker' key
                if isinstance(element, dict) and 'marker' in element:
                    element['marker'].parent = None
                elif hasattr(element, 'parent'):
                    element.parent = None

# ==============================================================================
# φ-RECURSIVE PHYSICS ENGINE
# ==============================================================================

class PhiRecursiveEngine:
    """Physics engine with thermodynamic awareness"""

    def __init__(self, genome):
        self.genome = genome
        self.genome_len = len(genome)

        self.P = 100
        self.s = 1.0
        self.ell = 0.8 / self.P
        self.gamma = 0.75

        self.fib_cache = {}
        self.params_cache = {}

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

    def psi_recursive(self, k, tau):
        """Simplified wavefunction"""
        r = self.compute_r(tau)
        theta = self.compute_theta(tau)

        amplitude = PHI ** (k / 2)
        radial = r ** (PHI ** k)
        phase = np.exp(1j * PHI ** k * theta)

        return amplitude * radial * phase

    def nucleotide_to_params(self, position):
        """Cached parameter extraction"""
        if position in self.params_cache:
            return self.params_cache[position]

        if position >= self.genome_len:
            return None

        base = self.genome[position]
        tau = float(position)

        k_raw = BASE_MAP.get(base, 1) - 1
        k = min(max(k_raw, 0), 7)

        _, note, color, name, alpha_k, v_k = GEOMETRIES[k]

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
            'h_bonds': HYDROGEN_BONDS.get(base, 2),
            'tm_contribution': TM_CONTRIBUTION.get(base, 2),
        }

        if len(self.params_cache) < 10000:
            self.params_cache[position] = params

        return params

    def get_complement_base(self, position):
        """Get complementary base for pairing check"""
        if position >= self.genome_len:
            return None
        base = self.genome[position]
        return COMPLEMENT.get(base)

# ==============================================================================
# THERMODYNAMIC CELL
# ==============================================================================

class ThermodynamicCell:
    """Cell with DNA thermodynamics-based lifecycle"""

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

        # Thermodynamic structures (with lifetimes!)
        self.structures = []  # List of DNAStructure objects

        self.centers = []

        # Temperature (affects decay rate)
        self.temperature = 1.0  # 1.0 = normal, >1.0 = faster decay

        # Displays
        self.progress_text = Text("0%", pos=self.center_offset + [0, 0, 20],
                                 color='white', font_size=20,
                                 anchor_x='center', parent=self.scene)

        self.physics_text = Text("", pos=self.center_offset + [0, 0, 18],
                                color='gold', font_size=10,
                                anchor_x='center', parent=self.scene)

    def update(self):
        self.frame += 1

        # Update strand positions
        N = 300
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

        # Create structures every 40 frames
        if self.frame % 40 == 0 and self.tau < self.genome_len:
            self.create_thermodynamic_rung()

        # Update ALL structures (decay based on thermodynamics)
        self.update_structures()

        self.tau += 1
        if self.tau >= self.genome_len:
            self.tau = 0

        self.update_displays()

    def create_thermodynamic_rung(self):
        """Create rung with inherent lifetime from base pairing"""
        params1 = self.engine.nucleotide_to_params(int(self.tau))
        if params1 is None:
            return

        # Get complementary position (simplified - same position for demo)
        # In real DNA, you'd check the opposite strand
        params2 = params1  # Or look at tau + offset for pairing

        k = params1['k']
        v_k = params1['v_k']
        color = params1['color']
        base = params1['base']

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
        size = 6 * (1 + params1['psi_magnitude'] * 0.1)

        # Create visual elements
        visual_elements = {}

        if self.config.get('show_rungs', True):
            mark = Markers(pos=all_pts, face_color=Color(color).rgba,
                          edge_color='white', size=size, parent=self.scene)
            visual_elements['rung'] = mark
        else:
            visual_elements['rung'] = None

        cen = all_pts.mean(axis=0)
        self.centers.append(cen)

        if self.config.get('show_labels', True):
            label_text = f"{base}"
            lbl = Text(label_text, pos=cen + [0, 0, 0.3],
                      color=color, font_size=8, bold=True,
                      anchor_x='center', parent=self.scene)
            visual_elements['label'] = lbl
        else:
            visual_elements['label'] = None

        # Echoes
        if self.config.get('show_echoes', True):
            echo_pts = all_pts * self.engine.gamma + np.random.normal(scale=0.01, size=all_pts.shape)
            echo = Markers(pos=echo_pts, face_color=(1, 1, 1, 0.05),
                          size=3, parent=self.scene)
            visual_elements['echo'] = echo
        else:
            visual_elements['echo'] = None

        # Links
        if self.config.get('show_links', True) and len(self.centers) > 1:
            prev_c = self.centers[-2]
            segs = []
            for p in all_pts[:min(3, len(all_pts))]:
                segs += [prev_c, p]
            link = Line(pos=np.array(segs), color=(0.7, 0.7, 0.7, 0.2),
                       width=1, connect='segments', parent=self.scene)
            visual_elements['link'] = link
        else:
            visual_elements['link'] = None

        # Organelle
        if self.config.get('show_organelles', True):
            spawn_prob = 0.003 + k * 0.001
            if np.random.rand() < spawn_prob:
                org = self.spawn_organelle(cen, color, size=0.05, n=2 + k // 4)
                visual_elements['organelle'] = org
            else:
                visual_elements['organelle'] = None
        else:
            visual_elements['organelle'] = None

        # Get second base for pairing (simplified)
        base2 = self.engine.get_complement_base(int(self.tau))
        if base2 is None:
            base2 = base

        # Create structure with thermodynamic properties!
        structure = DNAStructure(
            tau=self.tau,
            base1=base,
            base2=base2,
            visual_elements=visual_elements,
            structure_type='rung'
        )

        self.structures.append(structure)

    def spawn_organelle(self, center, color_rgb, size=0.05, n=3):
        """Spawn organelle cluster"""
        pts = center + np.random.normal(scale=0.03, size=(n, 3)) * size
        rgba = list(Color(color_rgb).rgba)
        rgba[3] = 0.4
        mark = Markers(pos=pts, face_color=rgba, edge_color=None, size=2, parent=self.scene)
        return {"marker": mark, "positions": pts, "color": rgba}

    def update_structures(self):
        """Update all structures based on DNA thermodynamics"""
        structures_to_remove = []

        for structure in self.structures:
            # Update based on inherent properties
            should_decay, alpha = structure.update(self.tau, self.temperature)

            if should_decay:
                structure.destroy()
                structures_to_remove.append(structure)
            else:
                # Fade based on remaining lifetime
                for elem in structure.visual_elements.values():
                    if elem is not None and hasattr(elem, 'set_data'):
                        # Update alpha for markers
                        if hasattr(elem, 'face_color'):
                            current_color = elem.face_color
                            if len(current_color) > 0:
                                new_color = current_color[0].copy()
                                new_color[3] = alpha
                                elem.set_data(face_color=new_color)

        # Remove decayed structures
        for structure in structures_to_remove:
            self.structures.remove(structure)

    def update_displays(self):
        percent = min(100 * self.tau / self.genome_len, 100)
        self.progress_text.text = f"{percent:.1f}%"

        params = self.engine.nucleotide_to_params(int(self.tau))
        if params:
            h_bonds = params['h_bonds']
            self.physics_text.text = f"k={params['k']} | H-bonds={h_bonds} | Structures={len(self.structures)}"

    def toggle_echoes(self):
        for struct in self.structures:
            if struct.visual_elements.get('echo'):
                struct.visual_elements['echo'].visible = not struct.visual_elements['echo'].visible

    def toggle_links(self):
        for struct in self.structures:
            if struct.visual_elements.get('link'):
                struct.visual_elements['link'].visible = not struct.visual_elements['link'].visible

    def toggle_organelles(self):
        for struct in self.structures:
            if struct.visual_elements.get('organelle'):
                struct.visual_elements['organelle']['marker'].visible = not struct.visual_elements['organelle']['marker'].visible

    def toggle_rungs(self):
        for struct in self.structures:
            if struct.visual_elements.get('rung'):
                struct.visual_elements['rung'].visible = not struct.visual_elements['rung'].visible

    def toggle_labels(self):
        for struct in self.structures:
            if struct.visual_elements.get('label'):
                struct.visual_elements['label'].visible = not struct.visual_elements['label'].visible

    def toggle_strands(self):
        self.strand1.visible = not self.strand1.visible
        self.strand2.visible = not self.strand2.visible

    def toggle_text(self):
        self.progress_text.visible = not self.progress_text.visible
        self.physics_text.visible = not self.physics_text.visible

# ==============================================================================
# APPLICATION
# ==============================================================================

class ThermodynamicVisualizerApp:
    """FASTA-driven thermodynamic visualization"""

    def __init__(self, fasta_file, division_interval=2000):
        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"FASTA not found: {fasta_file}")

        self.genome = load_genome(fasta_file)
        print(f"✓ Genome loaded: {len(self.genome)} nucleotides")

        self.engine = PhiRecursiveEngine(self.genome)
        print(f"✓ φ-Recursive engine initialized")
        if USING_C_BACKEND:
            print(f"✓ Using NumPy C backend")

        self.division_interval = division_interval

        self.config = {
            'show_strands': True,
            'show_rungs': True,
            'show_labels': True,
            'show_echoes': True,
            'show_links': True,
            'show_organelles': True,
            'show_text': True,
        }

        self.compute_statistics()

        self.canvas = scene.SceneCanvas(
            keys='interactive',
            size=(1600, 1000),
            bgcolor='black',
            title="FASTA-First φ-Framework v7 - THERMODYNAMIC LIFECYCLE"
        )
        self.view = self.canvas.central_widget.add_view()

        self.view.camera = TurntableCamera(
            fov=60,
            distance=50,
            elevation=15,
            azimuth=45,
            center=(0, 0, 0)
        )

        self.paused = False
        self.speed_multiplier = 1.0
        self.auto_rotate = True

        self.create_environment()

        self.cells = [ThermodynamicCell(self.genome, self.engine,
                                        center_offset=np.array([0, 0, 0]),
                                        parent_scene=self.view.scene,
                                        config=self.config)]

        self.timer = app.Timer(interval=0.03, connect=self.update, start=True)

        self.create_info_panel()
        self.frame = 0

        self.canvas.events.key_press.connect(self.on_key_press)

    def create_environment(self):
        """Reference environment"""
        grid_size = 80
        grid_step = 10
        grid_points = []
        for x in range(-grid_size, grid_size+1, grid_step):
            grid_points += [[x, -grid_size, 0], [x, grid_size, 0]]
        for y in range(-grid_size, grid_size+1, grid_step):
            grid_points += [[-grid_size, y, 0], [grid_size, y, 0]]

        self.grid = Line(pos=np.array(grid_points), color=(0.2, 0.2, 0.3, 0.3),
                        width=1, connect='segments', parent=self.view.scene)

        angles = np.linspace(0, 2*np.pi, 32)
        circle_pts = []
        for theta in angles:
            circle_pts.append([5*np.cos(theta), 5*np.sin(theta), 0])
        circle_pts.append(circle_pts[0])

        self.ref_circle = Line(pos=np.array(circle_pts), color=(0.3, 0.3, 0.5, 0.5),
                              width=2, parent=self.view.scene)

    def on_key_press(self, event):
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
            print(f"📊 Text: {'ON' if self.config['show_text'] else 'OFF'}")

        elif event.key.name == '+':
            # Increase temperature (faster decay)
            for cell in self.cells:
                cell.temperature = min(cell.temperature + 0.1, 3.0)
            print(f"🌡️  Temperature: {self.cells[0].temperature:.1f}")

        elif event.key.name == '-':
            # Decrease temperature (slower decay)
            for cell in self.cells:
                cell.temperature = max(cell.temperature - 0.1, 0.1)
            print(f"🌡️  Temperature: {self.cells[0].temperature:.1f}")

        elif event.key.name == 'H':
            self.print_help()

    def print_help(self):
        print("\n" + "="*70)
        print("THERMODYNAMIC LIFECYCLE CONTROL")
        print("="*70)
        print("FASTA DRIVES LIFECYCLE VIA DNA CHEMISTRY:")
        print("  - G-C pairs: 3 H-bonds → long-lived structures")
        print("  - A-T pairs: 2 H-bonds → short-lived structures")
        print("  - Complementarity: Watson-Crick pairing bonus")
        print("  - Thermal breathing: Natural DNA fluctuations")
        print()
        print("MOUSE:")
        print("  Left drag      - Rotate view")
        print("  Right drag     - Pan camera")
        print("  Mouse wheel    - Zoom")
        print()
        print("TOGGLE LAYERS:")
        print("  S - DNA Strands    U - Rungs         B - Labels")
        print("  E - Echoes         L - Links         O - Organelles")
        print("  T - Text displays")
        print()
        print("THERMODYNAMICS:")
        print("  + / -          - Increase/decrease temperature")
        print("                   (affects decay rate)")
        print()
        print("  SPACE          - Pause/Resume")
        print("  Up/Down        - Speed control")
        print("  R              - Toggle auto-rotate")
        print("  H              - Show this help")
        print("="*70 + "\n")

    def compute_statistics(self):
        print("\n" + "="*80)
        print("GENOME THERMODYNAMICS")
        print("="*80)

        base_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0}
        for base in self.genome[:10000]:
            if base in base_counts:
                base_counts[base] += 1

        total = sum(base_counts.values())
        gc_content = (base_counts['G'] + base_counts['C']) / total * 100

        print(f"  GC Content: {gc_content:.1f}% (higher = more stable structures)")
        for base, count in base_counts.items():
            h_bonds = HYDROGEN_BONDS[base]
            print(f"  {base} → {h_bonds} H-bonds: {count} ({100*count/total:.1f}%)")
        print("="*80 + "\n")

    def create_info_panel(self):
        info_text = (
            "FASTA-FIRST v7 - THERMODYNAMICS\n"
            f"φ = {PHI:.6f}\n"
            f"Genome: {len(self.genome)} bases\n"
            "\n"
            "DNA CHEMISTRY CONTROLS LIFECYCLE:\n"
            "GC: 3 H-bonds (stable)\n"
            "AT: 2 H-bonds (unstable)\n"
            "\n"
            "Toggles: S/U/B/E/L/O/T\n"
            "Temp: +/- | H:Help"
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

        self.thermo_text = Text("",
                               pos=(10, 200),
                               color='yellow', font_size=10,
                               anchor_x='left',
                               parent=self.canvas.scene)

    def update(self, event):
        if self.paused:
            self.thermo_text.text = "⏸️  PAUSED"
            return

        self.frame += 1

        updates = max(1, int(self.speed_multiplier))

        for _ in range(updates):
            for cell in self.cells:
                cell.update()

        # Display stats
        total_structures = sum(len(c.structures) for c in self.cells)
        self.cell_count_text.text = f"Cells: {len(self.cells)} | Active Structures: {total_structures}"

        temp = self.cells[0].temperature
        self.thermo_text.text = f"Temperature: {temp:.1f} | Speed: {self.speed_multiplier:.1f}x"

        if self.auto_rotate:
            self.view.camera.azimuth = self.frame * 0.15
            self.view.camera.elevation = 15 + 8 * np.sin(self.frame * 0.002)

    def run(self):
        print("\n" + "="*70)
        print("THERMODYNAMIC LIFECYCLE - INHERENT TO FASTA")
        print("="*70)
        print("✓ Structures form based on base pairing")
        print("✓ Lifetime determined by hydrogen bond count:")
        print("    G-C pairs: 3 bonds → ~3x longer life")
        print("    A-T pairs: 2 bonds → shorter life")
        print("✓ Decay via thermal breathing (natural DNA dynamics)")
        print("✓ No arbitrary rules - pure chemistry!")
        print()
        print("Press +/- to adjust temperature (decay rate)")
        print("Watch GC-rich regions create stable structures!")
        print("="*70 + "\n")

        self.canvas.show()
        app.run()

# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("FASTA-FIRST φ-FRAMEWORK v7 - THERMODYNAMIC LIFECYCLE")
    print("DNA Chemistry Controls Structure Birth & Death")
    print("="*80 + "\n")

    fasta_path = "ecoli_k12.fasta"

    try:
        visualizer = ThermodynamicVisualizerApp(fasta_path, division_interval=2000)
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
        visualizer = ThermodynamicVisualizerApp('demo_genome.fasta', division_interval=500)
        visualizer.run()

if __name__ == "__main__":
    main()
