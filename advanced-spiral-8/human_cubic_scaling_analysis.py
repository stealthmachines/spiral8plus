"""
CUBIC SCALING ANALYSIS APPLIED TO HUMAN GENOME (GRCh38)
=======================================================

Applies cubic scaling law framework to human genome sequence:
- Loads human genome GRCh38.p14 data
- Maps nucleotides to cubic scaling parameters
- Analyzes φ-framework cubic patterns in DNA
- Visualizes scaling modulation across genome regions
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from Bio import SeqIO
import os
from scipy.optimize import curve_fit

def load_human_genome():
    """Load human genome sequence with GENOME_LIMIT support"""
    env_limit_str = os.environ.get('GENOME_LIMIT', '100000')
    genome_limit = None if env_limit_str == 'all' else int(env_limit_str)
    chromosome = os.environ.get('GENOME_CHROMOSOME', 'NC_000001.11')
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

    sequence = ""
    for record in SeqIO.parse(genome_path, "fasta"):
        seq_str = str(record.seq).upper()

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


def analyze_cubic_scaling_genome(genome_seq):
    """Apply cubic scaling law analysis to genome sequence"""
    print("\n📊 CUBIC SCALING LAW ANALYSIS OF HUMAN GENOME")
    print("=" * 70)

    # φ-framework cubic coefficients (discovered)
    phi_a3, phi_a2, phi_a1, phi_a0 = -0.067652, 0.460612, -0.915276, 0.537585

    print("🎯 **1. CUBIC SCALING LAW MAPPING TO GENOME**")
    print("-" * 60)

    print("PHI-Framework Cubic Scaling Law:")
    print(f"alpha(P) = {phi_a3:.6f}P^3 + {phi_a2:.6f}P^2 + {phi_a1:.6f}P + {phi_a0:.6f}")
    print()

    # Map nucleotides to parameter positions
    nucleotide_to_position = {
        'A': 1,  # alpha_n position
        'T': 2,  # α_β position
        'G': 3,  # α_Ω position
        'C': 4,  # α_k position
        'N': 2.5 # Midpoint for unknown
    }

    param_names = {1: 'α_n', 2: 'α_β', 3: 'α_Ω', 4: 'α_k'}

    print("Nucleotide → Parameter Position Mapping:")
    for base, pos in nucleotide_to_position.items():
        alpha_value = phi_a3*pos**3 + phi_a2*pos**2 + phi_a1*pos + phi_a0
        param_name = param_names.get(pos, 'α_mid')
        print(f"  {base} → P={pos} ({param_name}): α={alpha_value:.6f}")

    # Calculate cubic scaling values for entire sequence
    print("\n🔬 **2. GENOME-WIDE CUBIC SCALING ANALYSIS**")
    print("-" * 60)

    scaling_values = []
    position_values = []

    for base in genome_seq:
        P = nucleotide_to_position.get(base, 2.5)
        alpha = phi_a3*P**3 + phi_a2*P**2 + phi_a1*P + phi_a0
        scaling_values.append(alpha)
        position_values.append(P)

    scaling_array = np.array(scaling_values)
    position_array = np.array(position_values)

    print("Cubic Scaling Statistics:")
    print(f"  Mean α: {np.mean(scaling_array):.6f}")
    print(f"  Std Dev: {np.std(scaling_array):.6f}")
    print(f"  Min α: {np.min(scaling_array):.6f}")
    print(f"  Max α: {np.max(scaling_array):.6f}")
    print(f"  Range: {np.max(scaling_array) - np.min(scaling_array):.6f}")

    # Analyze base distribution
    base_counts = {'A': 0, 'T': 0, 'G': 0, 'C': 0, 'N': 0}
    for base in genome_seq:
        if base in base_counts:
            base_counts[base] += 1
        else:
            base_counts['N'] += 1

    total = len(genome_seq)

    print("\nNucleotide Distribution & Scaling:")
    for base, count in base_counts.items():
        if count > 0:
            P = nucleotide_to_position[base]
            alpha = phi_a3*P**3 + phi_a2*P**2 + phi_a1*P + phi_a0
            percentage = (count / total) * 100

            print(f"  {base}: {count:>8,} ({percentage:>5.2f}%) → α={alpha:>9.6f}")

    # Windowed cubic scaling analysis
    print("\n📈 **3. WINDOWED CUBIC MODULATION ANALYSIS**")
    print("-" * 60)

    window_size = 1000
    stride = 500

    window_stats = []

    for start_idx in range(0, len(genome_seq) - window_size, stride):
        window = genome_seq[start_idx:start_idx + window_size]

        window_scaling = []
        for base in window:
            P = nucleotide_to_position.get(base, 2.5)
            alpha = phi_a3*P**3 + phi_a2*P**2 + phi_a1*P + phi_a0
            window_scaling.append(alpha)

        window_stats.append({
            'start': start_idx,
            'mean_alpha': np.mean(window_scaling),
            'std_alpha': np.std(window_scaling),
            'min_alpha': np.min(window_scaling),
            'max_alpha': np.max(window_scaling)
        })

    print(f"Analyzed {len(window_stats)} windows of {window_size} nucleotides")
    print(f"Stride: {stride} nucleotides")

    mean_alphas = [w['mean_alpha'] for w in window_stats]
    print(f"\nWindow Mean α Statistics:")
    print(f"  Overall mean: {np.mean(mean_alphas):.6f}")
    print(f"  Std deviation: {np.std(mean_alphas):.6f}")
    print(f"  Variation range: {np.max(mean_alphas) - np.min(mean_alphas):.6f}")

    # Detect cubic modulation patterns
    print("\n=== 4. CUBIC MODULATION PATTERN DETECTION ===")
    print("-" * 60)

    # Try to fit modulation patterns to window statistics
    window_positions = np.array([w['start'] for w in window_stats])
    window_means = np.array(mean_alphas)

    # Detrend by removing overall mean
    detrended = window_means - np.mean(window_means)

    # Try sinusoidal modulation fit
    def sinusoidal_mod(x, A, B, C):
        return A * np.sin(B * x + C)

    try:
        popt_sin, _ = curve_fit(sinusoidal_mod, window_positions, detrended,
                               p0=[0.01, 0.0001, 0], maxfev=5000)
        A_sin, B_sin, C_sin = popt_sin

        sin_fitted = sinusoidal_mod(window_positions, *popt_sin)
        sin_r2 = 1 - np.sum((detrended - sin_fitted)**2) / np.sum(detrended**2)

        print("Sinusoidal Modulation Pattern:")
        print(f"  Δα(x) = {A_sin:.6f} × sin({B_sin:.8f}×x + {C_sin:.6f})")
        print(f"  R² = {sin_r2:.6f}")
        print(f"  Period: {2*np.pi/abs(B_sin):.1f} nucleotides" if B_sin != 0 else "  Period: undefined")

    except Exception as e:
        print(f"Sinusoidal fit failed: {e}")
        sin_fitted = None

    # K-mer cubic scaling
    print("\n🧬 **5. K-MER CUBIC SCALING ANALYSIS**")
    print("-" * 60)

    k = 3  # Triplet codons
    kmer_scaling_distribution = {}

    for i in range(len(genome_seq) - k + 1):
        kmer = genome_seq[i:i+k]

        # Calculate mean cubic scaling for k-mer
        kmer_alphas = []
        for base in kmer:
            P = nucleotide_to_position.get(base, 2.5)
            alpha = phi_a3*P**3 + phi_a2*P**2 + phi_a1*P + phi_a0
            kmer_alphas.append(alpha)

        mean_alpha = np.mean(kmer_alphas)

        # Bin by scaling value
        bin_key = round(mean_alpha, 3)
        kmer_scaling_distribution[bin_key] = kmer_scaling_distribution.get(bin_key, 0) + 1

    print(f"{k}-mer cubic scaling distribution:")
    print(f"  Total unique α bins: {len(kmer_scaling_distribution)}")
    print(f"  Most common α values:")

    top_bins = sorted(kmer_scaling_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
    for alpha_bin, count in top_bins:
        percentage = (count / (len(genome_seq) - k + 1)) * 100
        print(f"    α ≈ {alpha_bin:>7.3f}: {count:>8,} ({percentage:>5.2f}%)")

    # Visualization
    print("\n📊 **6. VISUALIZATION GENERATION**")
    print("-" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cubic Scaling Law Analysis of Human Genome', fontsize=14, fontweight='bold')

    # Plot 1: Cubic scaling progression
    ax1 = axes[0, 0]
    sample_size = min(2000, len(scaling_values))
    ax1.plot(scaling_values[:sample_size], alpha=0.7, linewidth=0.5)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='α=0')
    ax1.set_title('Cubic Scaling α Progression')
    ax1.set_xlabel('Nucleotide Position')
    ax1.set_ylabel('α Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Nucleotide position distribution
    ax2 = axes[0, 1]
    bases = list(base_counts.keys())
    alphas = [phi_a3*nucleotide_to_position[b]**3 + phi_a2*nucleotide_to_position[b]**2 +
              phi_a1*nucleotide_to_position[b] + phi_a0 for b in bases]
    counts = [base_counts[b] for b in bases]

    colors = ['red' if a > 0 else 'blue' for a in alphas]
    ax2.bar(bases, counts, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_title('Nucleotide Distribution (Color by α sign)')
    ax2.set_xlabel('Nucleotide')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Windowed mean α
    ax3 = axes[1, 0]
    window_x = [w['start'] for w in window_stats]
    ax3.plot(window_x, mean_alphas, alpha=0.7, linewidth=1, label='Windowed Mean α')
    ax3.axhline(y=np.mean(mean_alphas), color='green', linestyle='--', alpha=0.5, label=f'Overall Mean')

    if sin_fitted is not None:
        ax3.plot(window_positions, sin_fitted + np.mean(window_means),
                color='orange', linestyle='--', alpha=0.7, label='Sinusoidal Fit')

    ax3.set_title(f'Windowed Mean α (window={window_size}, stride={stride})')
    ax3.set_xlabel('Genome Position')
    ax3.set_ylabel('Mean α')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: α histogram
    ax4 = axes[1, 1]
    ax4.hist(scaling_values, bins=50, alpha=0.7, edgecolor='black', color='purple')
    ax4.axvline(x=np.mean(scaling_values), color='red', linestyle='--',
               label=f'Mean={np.mean(scaling_values):.4f}')
    ax4.set_title('Cubic Scaling α Distribution')
    ax4.set_xlabel('α Value')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('human_cubic_scaling_analysis.png', dpi=150, bbox_inches='tight')
    print("[OK] Visualization saved: human_cubic_scaling_analysis.png")

    # Save analysis results
    analysis_results = {
        'genome_length': len(genome_seq),
        'cubic_coefficients': {
            'a3': phi_a3,
            'a2': phi_a2,
            'a1': phi_a1,
            'a0': phi_a0
        },
        'nucleotide_mapping': {
            base: {
                'position': nucleotide_to_position[base],
                'alpha': float(phi_a3*nucleotide_to_position[base]**3 +
                             phi_a2*nucleotide_to_position[base]**2 +
                             phi_a1*nucleotide_to_position[base] + phi_a0),
                'count': base_counts[base]
            }
            for base in ['A', 'T', 'G', 'C', 'N']
        },
        'statistics': {
            'mean_alpha': float(np.mean(scaling_array)),
            'std_alpha': float(np.std(scaling_array)),
            'min_alpha': float(np.min(scaling_array)),
            'max_alpha': float(np.max(scaling_array))
        },
        'window_analysis': {
            'window_size': window_size,
            'stride': stride,
            'num_windows': len(window_stats),
            'mean_alpha_variation': float(np.std(mean_alphas))
        }
    }

    with open('human_cubic_scaling_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print("[OK] Analysis saved: human_cubic_scaling_results.json")

    print("\n=== CUBIC SCALING ANALYSIS COMPLETE! ===")
    print(f"Analyzed {len(genome_seq):,} nucleotides from human genome")
    print("PHI-framework cubic scaling patterns mapped to DNA sequence")

    return analysis_results


def main():
    """Main execution function"""
    try:
        genome_seq = load_human_genome()
        results = analyze_cubic_scaling_genome(genome_seq)

        print("\n" + "="*70)
        print("Analysis complete! Check output files:")
        print("  • human_cubic_scaling_analysis.png")
        print("  • human_cubic_scaling_results.json")
        print("="*70)

    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
