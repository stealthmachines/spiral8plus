"""
8 GEOMETRIES + φ SCALING APPLIED TO HUMAN GENOME (GRCh38)
================================================================

Combines eight geometries phi framework with human genome analysis:
- Loads human genome GRCh38.p14 data
- Applies 8-dimensional geometric analysis to DNA sequence
- Maps nucleotides to geometric octave structure
- Visualizes φ-harmonic relationships in genetic data
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from math import pi, sqrt, log, exp, sin, cos, tan
from pathlib import Path
from Bio import SeqIO
import os

def load_human_genome():
    """Load human genome sequence with GENOME_LIMIT support"""
    env_limit_str = os.environ.get('GENOME_LIMIT', '100000')
    genome_limit = None if env_limit_str == 'all' else int(env_limit_str)
    chromosome = os.environ.get('GENOME_CHROMOSOME', 'NC_000001.11')  # Chr 1 default
    start_pos = int(os.environ.get('GENOME_START', '0'))

    print(f"Loading human genome: {chromosome}")
    print(f"Limit: {genome_limit if genome_limit else 'Full chromosome'}")
    print(f"Start position: {start_pos}")

    # Find human genome FASTA files
    possible_paths = [
        Path(__file__).parent / "ncbi_dataset" / "data" / chromosome / f"{chromosome}_GRCh38.p14_genomic.fna",
        Path(__file__).parent / "ncbi_dataset" / "data" / "GCF_000001405.40" / f"{chromosome}.fna",
        Path(__file__).parent / "ncbi_dataset" / "data" / "GCA_000001405.29" / f"{chromosome}.fna"
    ]

    genome_path = None
    for path in possible_paths:
        if path.exists():
            genome_path = path
            break

    if not genome_path:
        raise FileNotFoundError(f"Could not find human genome file for {chromosome}")

    print(f"Reading from: {genome_path}")

    # Load sequence
    sequence = ""
    for record in SeqIO.parse(genome_path, "fasta"):
        seq_str = str(record.seq).upper()

        # Apply start position and limit
        if start_pos > 0:
            seq_str = seq_str[start_pos:]

        if genome_limit:
            seq_str = seq_str[:genome_limit]

        sequence += seq_str

        if genome_limit and len(sequence) >= genome_limit:
            sequence = sequence[:genome_limit]
            break

    print(f"Loaded {len(sequence):,} nucleotides")
    return sequence


def analyze_genome_eight_geometries(genome_seq):
    """Apply 8 geometries phi framework to genome sequence"""
    print("\n=== 8 GEOMETRIES + PHI ANALYSIS OF HUMAN GENOME ===")
    print("=" * 70)

    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
    PI = np.pi

    # Extended scaling constants - 8 geometric dimensions
    alpha_constants = {
        'n': 0.015269,      # 1D: Point geometry (C note)
        'β': 0.008262,      # 2D: Line/plane geometry (D note)
        'Ω': 0.110649,      # 3D: Spatial geometry (E note)
        'k': -0.083485,     # 4D: Spacetime geometry (F note)
        'Ψ': 0.025847,      # 5D: Hyperspace geometry (G note)
        'Χ': -0.045123,     # 6D: Complex manifold (A note)
        'Φ': 0.067891,      # 7D: String theory space (B note)
        'Θ': 0.012345       # 8D: Unified field space (C octave)
    }

    # Musical note mapping
    musical_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C']
    color_spectrum = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo', 'Violet', 'White']

    # Map nucleotides to geometric dimensions
    nucleotide_to_dimension = {
        'A': 0,  # 1D Point (C note, Red)
        'T': 2,  # 3D Spatial (E note, Yellow)
        'G': 4,  # 5D Hyperspace (G note, Blue)
        'C': 6,  # 7D String theory (B note, Violet)
        'N': 3   # 4D Spacetime (F note, Green) - unknown nucleotide
    }

    print("=== 1. NUCLEOTIDE -> GEOMETRIC DIMENSION MAPPING ===")
    print("-" * 60)
    print(f"{'Base':<4} {'Dimension':<5} {'Note':<4} {'Color':<7} {'Geometry'}")
    print("-" * 60)

    geometries = [
        'Point', 'Line', 'Triangle', 'Tetrahedron',
        'Pentachoron', 'Hexacross', 'Heptacube', 'Octahedron'
    ]

    for base, dim_idx in nucleotide_to_dimension.items():
        param = list(alpha_constants.keys())[dim_idx]
        print(f"{base:<4} {dim_idx+1}D   {musical_notes[dim_idx]:<4} {color_spectrum[dim_idx]:<7} {geometries[dim_idx]}")

    # Analyze nucleotide distribution
    print("\n📊 **2. GENOME GEOMETRIC DISTRIBUTION**")
    print("-" * 60)

    base_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0, 'N': 0}
    for base in genome_seq:
        if base in base_counts:
            base_counts[base] += 1
        else:
            base_counts['N'] += 1

    total = len(genome_seq)

    print("Nucleotide counts and geometric frequencies:")
    for base, count in base_counts.items():
        if count > 0:
            dim_idx = nucleotide_to_dimension[base]
            percentage = (count / total) * 100
            note = musical_notes[dim_idx]
            color = color_spectrum[dim_idx]

            print(f"  {base}: {count:>8,} ({percentage:>5.2f}%) -> {note} ({color})")

    # Calculate phi-harmonic analysis
    print("\n=== 3. PHI-HARMONIC GENOME ANALYSIS ===")
    print("-" * 60)

    # Map sequence to phi-harmonic series
    golden_divisions = 8
    harmonic_values = []

    for base in genome_seq:
        dim_idx = nucleotide_to_dimension.get(base, 3)
        phi_interval = dim_idx * (np.log(2) / np.log(PHI)) / golden_divisions
        frequency_ratio = PHI ** phi_interval
        harmonic_values.append(frequency_ratio)

    harmonic_array = np.array(harmonic_values)

    print(f"Mean phi-harmonic: {np.mean(harmonic_array):.6f}")
    print(f"Std deviation: {np.std(harmonic_array):.6f}")
    print(f"phi-entropy: {-np.sum(harmonic_array * np.log(harmonic_array + 1e-10)):.6f}")

    # K-mer geometric analysis
    print("\n=== 4. K-MER GEOMETRIC OCTAVE ANALYSIS ===")
    print("-" * 60)

    k = 3  # Triplet codons
    kmer_geometric_distribution = {i: 0 for i in range(8)}

    for i in range(len(genome_seq) - k + 1):
        kmer = genome_seq[i:i+k]

        # Map k-mer to geometric dimension (sum of individual bases)
        dim_sum = sum(nucleotide_to_dimension.get(base, 3) for base in kmer)
        dim_idx = dim_sum % 8  # Map to one of 8 dimensions

        kmer_geometric_distribution[dim_idx] += 1

    print(f"{k}-mer geometric octave distribution:")
    for dim_idx, count in kmer_geometric_distribution.items():
        param = list(alpha_constants.keys())[dim_idx]
        note = musical_notes[dim_idx]
        color = color_spectrum[dim_idx]
        percentage = (count / sum(kmer_geometric_distribution.values())) * 100

        print(f"  {dim_idx+1}D ({param}, {note}, {color}): {count:>8,} ({percentage:>5.2f}%)")

    # Spiral geometric coupling
    print("\n=== 5. GENOMIC SPIRAL COUPLING ANALYSIS ===")
    print("-" * 60)

    # Calculate inter-dimensional coupling in genome sequence
    window_size = 100
    coupling_strengths = {f"{i}↔{j}": [] for i in range(8) for j in range(i+1, 8)}

    for start_idx in range(0, len(genome_seq) - window_size, window_size):
        window = genome_seq[start_idx:start_idx + window_size]

        # Count dimension occurrences in window
        dim_counts = [0] * 8
        for base in window:
            dim_idx = nucleotide_to_dimension.get(base, 3)
            dim_counts[dim_idx] += 1

        # Calculate coupling between dimensions
        for i in range(8):
            for j in range(i+1, 8):
                alpha_i = list(alpha_constants.values())[i]
                alpha_j = list(alpha_constants.values())[j]
                phi_distance = abs(i - j)

                coupling = (alpha_i * alpha_j * dim_counts[i] * dim_counts[j]) / (PHI ** phi_distance)
                coupling_strengths[f"{i}↔{j}"].append(coupling)

    print("Average geometric coupling strengths in genome:")
    for pair, strengths in list(coupling_strengths.items())[:10]:  # Show first 10
        if strengths:
            avg_strength = np.mean(strengths)
            print(f"  Dim {pair}: {avg_strength:.6f}")

    # Create visualization
    print("\n📈 **6. VISUALIZATION GENERATION**")
    print("-" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('8 Geometries φ Framework Applied to Human Genome', fontsize=14, fontweight='bold')

    # Plot 1: Nucleotide geometric distribution
    ax1 = axes[0, 0]
    bases = list(base_counts.keys())
    counts = list(base_counts.values())
    colors_map = {'A': 'red', 'T': 'yellow', 'G': 'blue', 'C': 'violet', 'N': 'green'}
    bar_colors = [colors_map[b] for b in bases]

    ax1.bar(bases, counts, color=bar_colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Nucleotide → Geometric Dimension Distribution')
    ax1.set_xlabel('Nucleotide (Geometric Dimension)')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3)

    # Plot 2: K-mer geometric octave
    ax2 = axes[0, 1]
    dims = list(kmer_geometric_distribution.keys())
    kmer_counts = list(kmer_geometric_distribution.values())
    octave_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'white']

    ax2.bar([f"{d+1}D" for d in dims], kmer_counts, color=octave_colors, alpha=0.7, edgecolor='black')
    ax2.set_title(f'{k}-mer Geometric Octave Distribution')
    ax2.set_xlabel('Geometric Dimension')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)

    # Plot 3: φ-harmonic progression
    ax3 = axes[1, 0]
    sample_size = min(1000, len(harmonic_values))
    ax3.plot(harmonic_values[:sample_size], alpha=0.7, linewidth=0.5)
    ax3.axhline(y=PHI, color='gold', linestyle='--', label=f'φ = {PHI:.4f}')
    ax3.set_title('φ-Harmonic Progression in Genome')
    ax3.set_xlabel('Nucleotide Position')
    ax3.set_ylabel('φ-Harmonic Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Musical note distribution
    ax4 = axes[1, 1]
    note_counts = {}
    for base, count in base_counts.items():
        if count > 0:
            dim_idx = nucleotide_to_dimension[base]
            note = musical_notes[dim_idx]
            note_counts[note] = note_counts.get(note, 0) + count

    notes = list(note_counts.keys())
    note_vals = list(note_counts.values())
    ax4.bar(notes, note_vals, color='skyblue', alpha=0.7, edgecolor='black')
    ax4.set_title('Musical Octave Distribution in Genome')
    ax4.set_xlabel('Musical Note')
    ax4.set_ylabel('Count')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('human_eight_geometries_phi_analysis.png', dpi=150, bbox_inches='tight')
    print("[OK] Visualization saved: human_eight_geometries_phi_analysis.png")

    # Save analysis results
    analysis_results = {
        'genome_length': len(genome_seq),
        'nucleotide_distribution': base_counts,
        'kmer_geometric_distribution': kmer_geometric_distribution,
        'phi_harmonic_stats': {
            'mean': float(np.mean(harmonic_array)),
            'std': float(np.std(harmonic_array)),
            'entropy': float(-np.sum(harmonic_array * np.log(harmonic_array + 1e-10)))
        },
        'geometric_mapping': {
            base: {
                'dimension': nucleotide_to_dimension[base] + 1,
                'note': musical_notes[nucleotide_to_dimension[base]],
                'color': color_spectrum[nucleotide_to_dimension[base]],
                'geometry': geometries[nucleotide_to_dimension[base]]
            }
            for base in ['A', 'T', 'G', 'C', 'N']
        }
    }

    with open('human_eight_geometries_phi_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print("[OK] Analysis saved: human_eight_geometries_phi_results.json")

    print("\n=== 8 GEOMETRIES PHI ANALYSIS COMPLETE! ===")
    print(f"Analyzed {len(genome_seq):,} nucleotides from human genome")
    print("Geometric octave harmony mapped to DNA sequence")

    return analysis_results


def main():
    """Main execution function"""
    try:
        # Load human genome
        genome_seq = load_human_genome()

        # Run 8 geometries analysis
        results = analyze_genome_eight_geometries(genome_seq)

        print("\n" + "="*70)
        print("Analysis complete! Check output files:")
        print("  • human_eight_geometries_phi_analysis.png")
        print("  • human_eight_geometries_phi_results.json")
        print("="*70)

    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
