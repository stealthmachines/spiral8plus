"""
CROSS-CAVITY TUNING APPLIED TO HUMAN GENOME (GRCh38)
====================================================

Applies phi-attractor cavity tuning framework to human genome:
- Loads human genome GRCh38.p14 data
- Maps DNA regions to phi-attractor cavity layers
- Analyzes golden ratio dissipation patterns
- Visualizes multi-scale cavity resonances in genome
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from Bio import SeqIO
import os
from scipy.signal import find_peaks

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
SQRT5 = np.sqrt(5)
PHI_7 = PHI**7  # ≈ 29.03
PHI_NEG7 = PHI**(-7)  # ≈ 0.03445


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


def define_genome_cavities():
    """Define phi-attractor cavity layers for genome analysis"""
    return {
        'exon': {
            'Omega_base': 2.5,
            'n_cascade': 0,
            'Q_factor': 100,
            'time_speedup': 1.0,
            'description': 'Coding regions, high expression'
        },
        'intron': {
            'Omega_base': 1.0,
            'n_cascade': 1,
            'Q_factor': 1000,
            'time_speedup': PHI_7**1,
            'description': 'Non-coding, regulatory regions'
        },
        'promoter': {
            'Omega_base': 0.3,
            'n_cascade': 2,
            'Q_factor': 5000,
            'time_speedup': PHI_7**2,
            'description': 'Regulatory control, high conservation'
        },
        'heterochromatin': {
            'Omega_base': 0.05,
            'n_cascade': 3,
            'Q_factor': 10000,
            'time_speedup': PHI_7**3,
            'description': 'Compressed, silenced regions'
        }
    }


def calculate_phi_attenuation(n_cascade):
    """Calculate phi-attenuation: Ω_{n+1} = phi^(-7) * Ω_n"""
    return PHI_NEG7 ** n_cascade


def analyze_cross_cavity_genome(genome_seq):
    """Apply cross-cavity phi-tuning to genome sequence"""
    print("\n=== CROSS-CAVITY PHI-TUNING ANALYSIS OF HUMAN GENOME ===")
    print("=" * 70)

    print("PHI-ATTRACTOR CAVITY MODEL:")
    print(f"  phi = {PHI:.8f}")
    print(f"  phi^7 = {PHI_7:.6f}")
    print(f"  phi^(-7) = {PHI_NEG7:.8f}")
    print()

    cavities = define_genome_cavities()

    print("🔬 **1. GENOME CAVITY DEFINITIONS**")
    print("-" * 60)

    print(f"{'Cavity Type':<18} {'Ω_base':<8} {'n_cascade':<10} {'Attenuation':<12} {'Q-factor'}")
    print("-" * 70)

    for cavity_name, props in cavities.items():
        attenuation = calculate_phi_attenuation(props['n_cascade'])
        effective_omega = props['Omega_base'] * attenuation

        print(f"{cavity_name:<18} {props['Omega_base']:<8.2f} {props['n_cascade']:<10} "
              f"{attenuation:<12.6f} {props['Q_factor']:<10,}")

    # Map nucleotides to cavity types based on phi-properties
    nucleotide_to_cavity = {
        'A': 'exon',           # High expression
        'T': 'intron',          # Moderate regulation
        'G': 'promoter',        # High conservation
        'C': 'heterochromatin', # Compression
        'N': 'intron'           # Unknown → default
    }

    print("\n📊 **2. NUCLEOTIDE → CAVITY MAPPING**")
    print("-" * 60)

    for base, cavity_type in nucleotide_to_cavity.items():
        cavity = cavities[cavity_type]
        attenuation = calculate_phi_attenuation(cavity['n_cascade'])
        effective_omega = cavity['Omega_base'] * attenuation

        print(f"  {base} → {cavity_type:<18} (Ω_eff = {effective_omega:.6f})")

    # Calculate cavity distribution in genome
    print("\n🌈 **3. GENOME-WIDE CAVITY DISTRIBUTION**")
    print("-" * 60)

    cavity_counts = {name: 0 for name in cavities.keys()}
    omega_values = []

    for base in genome_seq:
        cavity_type = nucleotide_to_cavity.get(base, 'intron')
        cavity_counts[cavity_type] += 1

        # Calculate effective Omega
        cavity = cavities[cavity_type]
        attenuation = calculate_phi_attenuation(cavity['n_cascade'])
        omega_eff = cavity['Omega_base'] * attenuation
        omega_values.append(omega_eff)

    total = len(genome_seq)

    print("Cavity Type Distribution:")
    for cavity_name, count in cavity_counts.items():
        percentage = (count / total) * 100
        cavity = cavities[cavity_name]
        attenuation = calculate_phi_attenuation(cavity['n_cascade'])

        print(f"  {cavity_name:<18}: {count:>8,} ({percentage:>5.2f}%) "
              f"[phi^(-7×{cavity['n_cascade']}) = {attenuation:.6f}]")

    # Omega statistics
    omega_array = np.array(omega_values)

    print("\nEffective Ω Statistics:")
    print(f"  Mean Ω_eff: {np.mean(omega_array):.6f}")
    print(f"  Std Dev: {np.std(omega_array):.6f}")
    print(f"  Min: {np.min(omega_array):.6f}")
    print(f"  Max: {np.max(omega_array):.6f}")

    # Golden attenuation analysis
    print("\n🌀 **4. GOLDEN ATTENUATION CASCADE ANALYSIS**")
    print("-" * 60)

    print("phi-Recursive Law I: Ω_{n+1} = phi^(-7) × Ω_n")
    print()

    max_cascade = 5
    base_omega = 1.0

    print(f"{'n':<3} {'Ω_n':<12} {'phi^(-7n)':<12} {'Cumulative Σ':<15} {'% of Total'}")
    print("-" * 60)

    cumulative_omega = 0
    total_omega = base_omega / (1 - PHI_NEG7)  # Geometric series sum

    for n in range(max_cascade + 1):
        omega_n = base_omega * (PHI_NEG7 ** n)
        cumulative_omega += omega_n
        percentage = (cumulative_omega / total_omega) * 100

        print(f"{n:<3} {omega_n:<12.8f} {PHI_NEG7**n:<12.8f} "
              f"{cumulative_omega:<15.8f} {percentage:>6.2f}%")

    print(f"\nTheoretical Sum (n→∞): {total_omega:.8f}")
    print(f"LAW II Equilibrium: Σ Ω_n ≈ {total_omega:.4f}")

    # Windowed cavity resonance analysis
    print("\n📈 **5. WINDOWED CAVITY RESONANCE ANALYSIS**")
    print("-" * 60)

    window_size = 1000
    stride = 500

    window_resonances = []

    for start_idx in range(0, len(genome_seq) - window_size, stride):
        window = genome_seq[start_idx:start_idx + window_size]

        window_omegas = []
        for base in window:
            cavity_type = nucleotide_to_cavity.get(base, 'intron')
            cavity = cavities[cavity_type]
            attenuation = calculate_phi_attenuation(cavity['n_cascade'])
            omega_eff = cavity['Omega_base'] * attenuation
            window_omegas.append(omega_eff)

        # Calculate resonance (variance in Omega)
        resonance = np.std(window_omegas)
        mean_omega = np.mean(window_omegas)

        window_resonances.append({
            'start': start_idx,
            'mean_omega': mean_omega,
            'resonance': resonance,
            'Q_factor': mean_omega / resonance if resonance > 0 else float('inf')
        })

    print(f"Analyzed {len(window_resonances)} windows")

    mean_omegas = [w['mean_omega'] for w in window_resonances]
    resonances = [w['resonance'] for w in window_resonances]

    print(f"\nWindow Statistics:")
    print(f"  Mean Ω range: {np.min(mean_omegas):.6f} - {np.max(mean_omegas):.6f}")
    print(f"  Resonance range: {np.min(resonances):.6f} - {np.max(resonances):.6f}")

    # Detect resonance peaks (cavity boundaries)
    resonance_array = np.array(resonances)
    peaks, properties = find_peaks(resonance_array, prominence=0.01)

    print(f"  Detected {len(peaks)} resonance peaks (cavity boundaries)")

    # Visualization
    print("\n📊 **6. VISUALIZATION GENERATION**")
    print("-" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cross-Cavity phi-Tuning Analysis of Human Genome', fontsize=14, fontweight='bold')

    # Plot 1: Omega progression
    ax1 = axes[0, 0]
    sample_size = min(2000, len(omega_values))
    ax1.plot(omega_values[:sample_size], alpha=0.7, linewidth=0.5)
    ax1.axhline(y=1.0, color='gold', linestyle='--', label='Ω=1 (equilibrium)')
    ax1.set_title('Effective Ω Progression')
    ax1.set_xlabel('Nucleotide Position')
    ax1.set_ylabel('Ω_eff')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cavity distribution
    ax2 = axes[0, 1]
    cavity_names = list(cavity_counts.keys())
    counts = list(cavity_counts.values())
    colors = ['red', 'orange', 'yellow', 'blue']

    ax2.bar(cavity_names, counts, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_title('Cavity Type Distribution')
    ax2.set_xlabel('Cavity Type')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Window resonances
    ax3 = axes[1, 0]
    window_x = [w['start'] for w in window_resonances]
    ax3.plot(window_x, mean_omegas, alpha=0.7, label='Mean Ω', linewidth=1)
    ax3.plot(window_x, resonances, alpha=0.7, label='Resonance', linewidth=1, color='orange')

    # Mark peaks
    if len(peaks) > 0:
        peak_x = [window_resonances[p]['start'] for p in peaks]
        peak_y = [resonances[p] for p in peaks]
        ax3.scatter(peak_x, peak_y, color='red', s=50, zorder=5, label='Cavity Boundaries')

    ax3.set_title(f'Windowed Cavity Resonance (window={window_size})')
    ax3.set_xlabel('Genome Position')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: phi-attenuation cascade
    ax4 = axes[1, 1]
    n_values = np.arange(0, 6)
    omega_cascade = [PHI_NEG7**n for n in n_values]

    ax4.semilogy(n_values, omega_cascade, 'o-', linewidth=2, markersize=8, color='purple')
    ax4.set_title('phi-Attenuation Cascade: Ω_n = phi^(-7n)')
    ax4.set_xlabel('Cascade Level n')
    ax4.set_ylabel('Ω_n (log scale)')
    ax4.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('human_cross_cavity_tuning_analysis.png', dpi=150, bbox_inches='tight')
    print("[OK] Visualization saved: human_cross_cavity_tuning_analysis.png")

    # Save results
    analysis_results = {
        'genome_length': len(genome_seq),
        'phi_constants': {
            'phi': PHI,
            'phi_7': PHI_7,
            'phi_neg7': PHI_NEG7,
            'equilibrium_sum': float(1 / (1 - PHI_NEG7))
        },
        'cavity_distribution': {
            name: {
                'count': int(cavity_counts[name]),
                'percentage': float(cavity_counts[name] / total * 100),
                'omega_base': cavities[name]['Omega_base'],
                'n_cascade': cavities[name]['n_cascade'],
                'attenuation': float(calculate_phi_attenuation(cavities[name]['n_cascade']))
            }
            for name in cavity_names
        },
        'omega_statistics': {
            'mean': float(np.mean(omega_array)),
            'std': float(np.std(omega_array)),
            'min': float(np.min(omega_array)),
            'max': float(np.max(omega_array))
        },
        'resonance_peaks': {
            'count': len(peaks),
            'positions': [int(window_resonances[p]['start']) for p in peaks] if len(peaks) > 0 else []
        }
    }

    with open('human_cross_cavity_tuning_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print("[OK] Analysis saved: human_cross_cavity_tuning_results.json")

    print("\n[SUCCESS] **CROSS-CAVITY phi-TUNING ANALYSIS COMPLETE!** [SUCCESS]")
    print(f"Analyzed {len(genome_seq):,} nucleotides from human genome")
    print(f"Golden attenuation cascade: Ω_(n+1) = phi^(-7) × Ω_n")
    print(f"Detected {len(peaks)} cavity boundary resonances")

    return analysis_results


def main():
    """Main execution function"""
    try:
        genome_seq = load_human_genome()
        results = analyze_cross_cavity_genome(genome_seq)

        print("\n" + "="*70)
        print("Analysis complete! Check output files:")
        print("  • human_cross_cavity_tuning_analysis.png")
        print("  • human_cross_cavity_tuning_results.json")
        print("="*70)

    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
