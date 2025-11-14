"""
WATERFALL ANIMATION FOR HUMAN GENOME (GRCh38)
=============================================

Creates animated waterfall visualization of human genome:
- Loads human genome GRCh38.p14 data
- Maps nucleotides to frequency domain
- Generates phi-harmonic waterfall plot
- Animates spectral evolution across genome regions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
from pathlib import Path
from Bio import SeqIO
import os

PHI = (1 + np.sqrt(5)) / 2


def load_human_genome():
    """Load human genome sequence with GENOME_LIMIT support"""
    genome_limit = os.environ.get('GENOME_LIMIT', '50000')  # Smaller default for animation
    chromosome = os.environ.get('GENOME_CHROMOSOME', 'NC_000001.11')
    start_pos = int(os.environ.get('GENOME_START', '0'))

    if genome_limit.lower() == 'all':
        limit = None
    else:
        limit = int(genome_limit)

    print(f"Loading human genome: {chromosome}")
    print(f"Limit: {limit if limit else 'Full chromosome'}")

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

        if limit:
            seq_str = seq_str[:limit]

        sequence += seq_str

        if limit and len(sequence) >= limit:
            sequence = sequence[:limit]
            break

    print(f"Loaded {len(sequence):,} nucleotides")
    return sequence


def nucleotide_to_frequency(base):
    """Map nucleotide to phi-harmonic frequency"""
    freq_map = {
        'A': PHI ** 0,      # 1.0 Hz (fundamental)
        'T': PHI ** 1,      # 1.618 Hz
        'G': PHI ** 2,      # 2.618 Hz
        'C': PHI ** 3,      # 4.236 Hz
        'N': PHI ** 0.5     # sqrt(PHI)
    }
    return freq_map.get(base, PHI ** 0.5)


def compute_waterfall_data(genome_seq, window_size=512, hop_size=128):
    """Compute waterfall spectrogram data from genome"""
    print("\n📊 Computing waterfall spectrogram...")

    num_windows = (len(genome_seq) - window_size) // hop_size + 1
    freq_bins = window_size // 2

    waterfall = np.zeros((num_windows, freq_bins))
    time_points = []

    for i in range(num_windows):
        start_idx = i * hop_size
        end_idx = start_idx + window_size

        if end_idx > len(genome_seq):
            break

        window = genome_seq[start_idx:end_idx]

        # Convert window to frequency signal
        freq_signal = np.array([nucleotide_to_frequency(base) for base in window])

        # Compute FFT
        fft = np.fft.fft(freq_signal)
        magnitude = np.abs(fft[:freq_bins])

        waterfall[i, :] = magnitude
        time_points.append(start_idx)

    waterfall = waterfall[:i+1, :]  # Trim to actual size

    print(f"  Waterfall shape: {waterfall.shape}")
    print(f"  Time windows: {len(time_points)}")
    print(f"  Frequency bins: {freq_bins}")

    return waterfall, time_points


def create_static_waterfall(genome_seq):
    """Create static waterfall plot"""
    print("\n🌊 WATERFALL VISUALIZATION OF HUMAN GENOME")
    print("=" * 70)

    # Compute waterfall data
    window_size = 512
    hop_size = 128
    waterfall, time_points = compute_waterfall_data(genome_seq, window_size, hop_size)

    # Normalize
    waterfall_norm = waterfall / (np.max(waterfall) + 1e-10)
    waterfall_db = 20 * np.log10(waterfall_norm + 1e-10)

    # Create visualization
    fig = plt.figure(figsize=(14, 10))

    # 3D waterfall plot
    ax1 = fig.add_subplot(2, 2, (1, 3), projection='3d')

    X, Y = np.meshgrid(np.arange(waterfall.shape[1]), np.arange(waterfall.shape[0]))

    surf = ax1.plot_surface(X, Y, waterfall_db, cmap='viridis',
                           linewidth=0, antialiased=True, alpha=0.9)

    ax1.set_title('3D Waterfall: Human Genome phi-Harmonic Spectrogram', fontweight='bold')
    ax1.set_xlabel('Frequency Bin')
    ax1.set_ylabel('Time Window')
    ax1.set_zlabel('Magnitude (dB)')
    ax1.view_init(elev=30, azim=45)

    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

    # 2D heatmap
    ax2 = fig.add_subplot(2, 2, 2)

    im = ax2.imshow(waterfall_db, aspect='auto', origin='lower',
                   cmap='hot', interpolation='bilinear')

    ax2.set_title('2D Waterfall Heatmap')
    ax2.set_xlabel('Frequency Bin')
    ax2.set_ylabel('Time Window')

    fig.colorbar(im, ax=ax2, label='Magnitude (dB)')

    # Frequency spectrum at specific time
    ax3 = fig.add_subplot(2, 2, 4)

    mid_time = waterfall.shape[0] // 2
    spectrum = waterfall[mid_time, :]

    freq_axis = np.arange(len(spectrum)) * (PHI / window_size)

    ax3.plot(freq_axis, spectrum, linewidth=1.5, color='blue')
    ax3.set_title(f'Frequency Spectrum at Window {mid_time}')
    ax3.set_xlabel('Frequency (phi-harmonic units)')
    ax3.set_ylabel('Magnitude')
    ax3.grid(True, alpha=0.3)

    # Add phi-harmonic markers
    phi_harmonics = [PHI**n for n in range(5)]
    for harm in phi_harmonics:
        if harm < freq_axis[-1]:
            ax3.axvline(x=harm, color='red', linestyle='--', alpha=0.5, linewidth=0.5)

    plt.tight_layout()
    plt.savefig('human_waterfall_static.png', dpi=150, bbox_inches='tight')
    print("\n[OK] Static waterfall saved: human_waterfall_static.png")

    return waterfall, time_points


def create_animated_waterfall(genome_seq, output_file='human_waterfall_animation.mp4'):
    """Create animated waterfall visualization"""
    print("\n🎬 Creating animated waterfall...")

    # Compute data
    window_size = 512
    hop_size = 256
    waterfall, time_points = compute_waterfall_data(genome_seq, window_size, hop_size)

    waterfall_norm = waterfall / (np.max(waterfall) + 1e-10)
    waterfall_db = 20 * np.log10(waterfall_norm + 1e-10)

    # Setup figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Initialize plots
    num_display_windows = 50  # Show last 50 windows

    im = ax1.imshow(waterfall_db[:num_display_windows, :],
                   aspect='auto', origin='lower', cmap='hot',
                   vmin=np.min(waterfall_db), vmax=np.max(waterfall_db))

    ax1.set_title('Rolling Waterfall Spectrogram')
    ax1.set_xlabel('Frequency Bin')
    ax1.set_ylabel('Time Window')

    freq_axis = np.arange(waterfall.shape[1]) * (PHI / window_size)
    line, = ax2.plot(freq_axis, waterfall[0, :], linewidth=1.5, color='blue')

    ax2.set_title('Current Frequency Spectrum')
    ax2.set_xlabel('Frequency (phi-harmonic units)')
    ax2.set_ylabel('Magnitude')
    ax2.set_ylim(np.min(waterfall), np.max(waterfall))
    ax2.grid(True, alpha=0.3)

    # Add phi-harmonic markers
    phi_harmonics = [PHI**n for n in range(5)]
    for harm in phi_harmonics:
        if harm < freq_axis[-1]:
            ax2.axvline(x=harm, color='red', linestyle='--', alpha=0.5, linewidth=0.5)

    position_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                            verticalalignment='top', color='white',
                            fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    def animate(frame):
        """Animation update function"""
        if frame >= len(waterfall):
            return im, line, position_text

        # Update waterfall (rolling window)
        start_idx = max(0, frame - num_display_windows + 1)
        end_idx = frame + 1

        data_slice = waterfall_db[start_idx:end_idx, :]

        if len(data_slice) < num_display_windows:
            # Pad with zeros at the beginning
            padding = np.ones((num_display_windows - len(data_slice), waterfall.shape[1])) * np.min(waterfall_db)
            data_slice = np.vstack([padding, data_slice])

        im.set_array(data_slice)

        # Update spectrum
        line.set_ydata(waterfall[frame, :])

        # Update position text
        genome_pos = time_points[frame] if frame < len(time_points) else 0
        position_text.set_text(f'Genome Position: {genome_pos:,}\nWindow: {frame}/{len(waterfall)}')

        return im, line, position_text

    # Create animation
    num_frames = min(200, len(waterfall))  # Limit frames for file size
    interval = 50  # ms between frames

    print(f"  Animating {num_frames} frames...")

    anim = animation.FuncAnimation(fig, animate, frames=num_frames,
                                  interval=interval, blit=False, repeat=True)

    # Save animation
    try:
        Writer = animation.writers['pillow']
        writer = Writer(fps=20, metadata=dict(artist='Genome Analyzer'))
        anim.save('human_waterfall_animation.gif', writer=writer, dpi=100)
        print(f"[OK] Animation saved: human_waterfall_animation.gif")
    except Exception as e:
        print(f"[WARN]  Could not save as GIF: {e}")
        print("  Displaying animation instead...")
        plt.show()

    return anim


def main():
    """Main execution function"""
    try:
        print("🌊 WATERFALL VISUALIZATION FOR HUMAN GENOME")
        print("=" * 70)

        # Load genome
        genome_seq = load_human_genome()

        # Create static waterfall
        waterfall, time_points = create_static_waterfall(genome_seq)

        print("\n📈 **WATERFALL STATISTICS**")
        print("-" * 60)
        print(f"  Total time windows: {len(time_points)}")
        print(f"  Frequency bins: {waterfall.shape[1]}")
        print(f"  Peak magnitude: {np.max(waterfall):.2f}")
        print(f"  Mean magnitude: {np.mean(waterfall):.2f}")
        print(f"  phi-harmonic range: 1.0 to {PHI**4:.3f} Hz")

        # Ask user if they want animation
        print("\n" + "="*70)
        print("Static waterfall complete!")
        print("\nWould you like to create an animated GIF?")
        print("(This may take 30-60 seconds)")

        create_anim = input("Create animation? (y/n): ").strip().lower()

        if create_anim == 'y':
            create_animated_waterfall(genome_seq)

        print("\n" + "="*70)
        print("Waterfall visualization complete!")
        print("Files created:")
        print("  • human_waterfall_static.png")
        if create_anim == 'y':
            print("  • human_waterfall_animation.gif")
        print("="*70)

    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
