"""
FASTA Universe 2.0 — Auto-detecting Human Genome visualization
Automatically finds and loads the Human Genome GRCh38 from ncbi_dataset
Multi-particle simulation with genome-driven holographic rendering
Install: pip install vispy pyqt6 numpy
Run: python fasta17_auto.py
"""

import os
import glob
import numpy as np
from vispy import scene, app
from vispy.scene.visuals import Markers, Text
from vispy.color import Color

# ---------- CONSTANTS ----------
phi = (1 + np.sqrt(5)) / 2
golden_angle_deg = 360 / (phi ** 2)

bases = ['A','T','G','C']
base_map = {b:i for i,b in enumerate(bases)}
geom_colors = ['red','green','blue','violet']

core_radius = 15.0
strand_sep = 0.5
MAX_POINTS = 100_000
MAX_ORG_POINTS = 50_000
BATCH_STEPS = 500

# ---------- AUTO-DETECT AND LOAD GENOME ----------
def find_human_fasta():
    """Automatically find the Human Genome FASTA file in the ncbi_dataset"""
    import glob
    possible_paths = [
        r"ncbi_dataset\ncbi_dataset\data\GCF_000001405.40\*.fna",
        r"ncbi_dataset\ncbi_dataset\data\GCA_000001405.29\*.fna",
        r"ncbi_dataset\ncbi_dataset\data\*\*.fna",
    ]

    for pattern in possible_paths:
        files = glob.glob(pattern)
        if files:
            return files[0]

    raise FileNotFoundError("Could not find Human Genome FASTA file in ncbi_dataset directory")


def load_genome(fasta_file, max_nucleotides=None, chromosome=None, start_position=0):
    """
    Load genome sequence from FASTA file with environment variable support

    Args:
        fasta_file: Path to FASTA file
        max_nucleotides: Maximum nucleotides to load (None = use GENOME_LIMIT env var, default 100000)
        chromosome: Specific chromosome to load (None = use GENOME_CHROMOSOME env var)
        start_position: Starting position in sequence (default 0, or GENOME_START env var)

    Returns:
        str: Genome sequence
    """
    import os

    # Get from environment if not specified
    if max_nucleotides is None:
        env_limit = os.environ.get('GENOME_LIMIT', '100000')
        if env_limit == 'all':
            max_nucleotides = None  # Load full genome
        else:
            try:
                max_nucleotides = int(env_limit)
            except ValueError:
                max_nucleotides = 100000

    if chromosome is None:
        chromosome = os.environ.get('GENOME_CHROMOSOME', None)

    if start_position == 0:
        env_start = os.environ.get('GENOME_START', '0')
        try:
            start_position = int(env_start)
        except ValueError:
            start_position = 0

    sequence = ""
    current_chromosome = None
    nucleotide_count = 0
    position_in_chromosome = 0
    skip_until_start = start_position > 0

    print(f"Loading genome from {fasta_file}...")
    if chromosome:
        print(f"  Filtering: Chromosome {chromosome}")
    if start_position > 0:
        print(f"  Starting at position: {start_position:,}")
    print(f"  Limit: {max_nucleotides:,} nucleotides")

    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith(">"):
                # New chromosome header
                header = line.strip()[1:].split()[0]
                current_chromosome = header
                position_in_chromosome = 0

                # If we're filtering by chromosome and this isn't it, skip
                if chromosome and current_chromosome != chromosome:
                    continue

                # Reset skip flag for new chromosome
                if chromosome and current_chromosome == chromosome:
                    skip_until_start = start_position > 0
                    print(f"Loading from {current_chromosome}...")
            else:
                # If filtering by chromosome and this isn't it, skip
                if chromosome and current_chromosome != chromosome:
                    continue

                bases = line.strip()

                # Handle start position skipping
                if skip_until_start:
                    if position_in_chromosome + len(bases) <= start_position:
                        position_in_chromosome += len(bases)
                        continue
                    else:
                        # Start is within this line
                        offset = start_position - position_in_chromosome
                        bases = bases[offset:]
                        position_in_chromosome = start_position
                        skip_until_start = False
                        print(f"  Started at position {start_position:,}")

                position_in_chromosome += len(bases)

                # Add nucleotides up to limit
                remaining = max_nucleotides - nucleotide_count

                if remaining <= 0:
                    break

                sequence += bases[:remaining]
                nucleotide_count += len(bases[:remaining])

                if nucleotide_count >= max_nucleotides:
                    break

            if nucleotide_count >= max_nucleotides:
                break

    print(f"  Loaded {nucleotide_count:,} nucleotides")
    return sequence

# Load genome
fasta_file = find_human_fasta()
genome_seq = load_genome(fasta_file)
genome_len = len(genome_seq)
print(f"Genome loaded: {genome_len:,} nucleotides")

class ViralParticle:
    def __init__(self, genome, origin=np.zeros(3), cell_id=0):
        self.genome = genome
        self.origin = np.array(origin, dtype=np.float32)
        self.cell_id = cell_id
        self.frame = 0
        self.lattice_positions = np.zeros((MAX_POINTS,3),dtype=np.float32)
        self.organelles_positions = np.zeros((MAX_ORG_POINTS,3),dtype=np.float32)
        self.lattice_count = 0
        self.organelles_count = 0

        # Dynamic coloring based on particle ID and position
        hue = (cell_id * 137.5) % 360  # Golden angle for color distribution
        sat = 0.7 + 0.3 * np.sin(cell_id * 0.1)
        val = 0.8 + 0.2 * np.cos(cell_id * 0.1)

        # Convert HSV to RGB
        c = val * sat
        x = c * (1 - abs((hue / 60) % 2 - 1))
        m = val - c

        if hue < 60:
            r, g, b = c, x, 0
        elif hue < 120:
            r, g, b = x, c, 0
        elif hue < 180:
            r, g, b = 0, c, x
        elif hue < 240:
            r, g, b = 0, x, c
        elif hue < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        base_color = np.array([r+m, g+m, b+m, 0.8])
        organelle_color = np.array([b+m, r+m, g+m, 0.6])

        self.lattice_marker = Markers(pos=self.lattice_positions[:0], face_color=base_color, size=2, parent=view.scene)
        self.organelles_marker = Markers(pos=self.organelles_positions[:0], face_color=organelle_color, size=4, parent=view.scene)

    def genome_to_vec(self, idx):
        base = self.genome[idx]
        dim = base_map.get(base, 0)

        # Enhanced viral genome spiral
        progress = idx / len(self.genome)
        theta = idx * np.radians(golden_angle_deg) + self.cell_id * 0.1

        # Adaptive radius based on genome position
        r = core_radius * (0.3 + 0.7 * (1 - progress**1.3))

        # Helical structure with viral-specific parameters
        z_freq = 8 + 2 * np.sin(self.cell_id * 0.1)  # Varied frequency per particle
        z = np.sin(progress * np.pi * z_freq) * 2 + progress * 8

        # Base-dependent angular offset
        a = np.radians(dim * 90 + self.frame * 0.1)

        # 3D position with slight wobble
        wobble = 0.1 * np.sin(idx * 0.01 + self.frame * 0.001)
        x = (r + wobble) * np.cos(theta) * np.cos(a) - (r + wobble) * np.sin(theta) * np.sin(a)
        y = (r + wobble) * np.sin(theta) * np.cos(a) + (r + wobble) * np.cos(theta) * np.sin(a)

        return np.array([x, y, z], dtype=np.float32) + self.origin

    def step(self):
        for _ in range(BATCH_STEPS):
            idx = self.frame % len(self.genome)
            pos = self.genome_to_vec(idx)

            # Add lattice point
            self.lattice_positions[self.lattice_count % MAX_POINTS] = pos
            self.lattice_count += 1

            # Viral proteins/complexes
            base = self.genome[idx]
            protein_probability = 0.03
            if base in ['G', 'C']:  # Higher probability for G/C rich regions
                protein_probability = 0.06

            if np.random.rand() < protein_probability:
                n = 2 + np.random.randint(5)
                protein_pts = pos + np.random.normal(scale=0.2, size=(n,3))

                end_idx = min(self.organelles_count + n, MAX_ORG_POINTS)
                actual_n = end_idx - self.organelles_count
                if actual_n > 0:
                    self.organelles_positions[self.organelles_count:end_idx] = protein_pts[:actual_n]
                    self.organelles_count = end_idx

            self.frame += 1

        # Update visual markers
        actual_lattice_count = min(self.lattice_count, MAX_POINTS)
        actual_organelle_count = min(self.organelles_count, MAX_ORG_POINTS)

        self.lattice_marker.set_data(self.lattice_positions[:actual_lattice_count])
        if actual_organelle_count > 0:
            self.organelles_marker.set_data(self.organelles_positions[:actual_organelle_count])

# ---------- VISPY SETUP ----------
canvas = scene.SceneCanvas(keys='interactive', size=(1600, 900), bgcolor='#000022', title='Human Genome GRCh38 Visualization')
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.distance = 70

# ---------- VIRAL SIMULATION SYSTEM ----------
particles = [ViralParticle(genome_seq, origin=np.zeros(3), cell_id=0)]
replication_interval = 12_000  # Faster viral replication
max_particles = 200  # Manageable number for visualization
particle_spacing = 12.0

# UI Text elements
title_text = Text("Human Genome GRCh38 Visualization", pos=[0, 50, core_radius*3],
                  color='white', font_size=24, anchor_x='center', parent=view.scene)

progress_text = Text("0%", pos=[0, 20, core_radius*3], color='yellow', font_size=18,
                     anchor_x='center', parent=view.scene)

info_text = Text(f"Genome: {genome_len:,} nucleotides", pos=[0, -10, core_radius*3],
                 color='cyan', font_size=14, anchor_x='center', parent=view.scene)

stats_text = Text("Particles: 1", pos=[0, -30, core_radius*3], color='lime', font_size=12,
                  anchor_x='center', parent=view.scene)

# ---------- MAIN UPDATE LOOP ----------
def update(ev):
    global particles

    # Update all viral particles
    for particle in particles:
        particle.step()

    # Viral replication
    if len(particles) < max_particles:
        total_frames = sum(p.frame for p in particles)
        if total_frames > replication_interval * len(particles):
            # New particle position with some clustering
            if len(particles) < 10:
                # Early stage: spread out
                new_origin = np.random.uniform(-particle_spacing*3, particle_spacing*3, 3)
            else:
                # Later stage: cluster around existing particles
                parent = particles[np.random.randint(len(particles))]
                new_origin = parent.origin + np.random.normal(scale=particle_spacing, size=3)

            particles.append(ViralParticle(genome_seq, origin=new_origin, cell_id=len(particles)))

    # Particle dynamics
    for i, p1 in enumerate(particles):
        for j, p2 in enumerate(particles[i+1:], start=i+1):
            delta = p2.origin - p1.origin
            dist = np.linalg.norm(delta)

            if 0 < dist < particle_spacing:
                # Weak repulsion to prevent overlap
                force = 0.005 * (particle_spacing - dist)
                direction = delta / dist
                p1.origin -= direction * force
                p2.origin += direction * force

    # Dynamic camera
    frame_total = sum(p.frame for p in particles)
    view.camera.azimuth = frame_total * 0.05
    view.camera.elevation = 25 + 10 * np.sin(frame_total * 0.0008)
    view.camera.distance = 50 + 20 * np.sin(frame_total * 0.0005)

    # Update UI
    progress = min(frame_total / (len(genome_seq) * max_particles) * 100, 100)
    progress_text.text = f"Progress: {progress:.1f}%"
    stats_text.text = f"Particles: {len(particles)} | Total Points: {sum(p.lattice_count for p in particles):,}"

    canvas.update()

# ---------- STARTUP ----------
timer = app.Timer(interval=0.02, connect=update, start=True)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Human Genome GRCh38.p14 Visualization")
    print("="*60)
    if 'genome_len' in dir():
        print(f"Genome length: {genome_len:,} nucleotides")
    print("="*60)

    print("\n" + "="*60)
    print("Human Genome GRCh38 Visualization")
    print("="*60)
    print(f"Genome length: {genome_len:,} nucleotides")
    print(f"Max particles: {max_particles}")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Mouse wheel: Zoom")
    print("  - ESC: Exit")
    print("="*60)

    canvas.show()
    app.run()