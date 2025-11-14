"""
LIGO Ringdown φ-Harmonic Analysis
==================================

Searches for φ-spaced overtones in gravitational wave ringdown data.
This is the IMMEDIATE next validation step.

Requirements:
    pip install gwosc scipy numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit

try:
    from gwosc.datasets import event_gps
    from gwosc import datasets
    GWOSC_AVAILABLE = True
except ImportError:
    print("⚠️ gwosc not installed. Install with: pip install gwosc")
    GWOSC_AVAILABLE = False

# Constants
PHI = (1 + np.sqrt(5)) / 2
PHI_INV_7 = PHI**(-7)

def analyze_ringdown_harmonics(event_name='GW150914', detector='H1'):
    """
    Analyze gravitational wave ringdown for φ-harmonic overtones

    Expected behavior:
    - Fundamental quasi-normal mode (QNM) at f₀
    - Overtones at f_n = f₀ × φ^n
    - Amplitudes A_n = A₀ × φ^(-7n)
    """

    if not GWOSC_AVAILABLE:
        print("Cannot proceed without gwosc package")
        return None

    print(f"Analyzing {event_name} from {detector}...")

    # Get event time
    try:
        gps_time = event_gps(event_name)
        print(f"Event GPS time: {gps_time}")
    except:
        print(f"Event {event_name} not found in catalog")
        return None

    # Download strain data around merger
    # Focus on ringdown (post-merger)
    from gwosc.datasets import event_strain

    try:
        # Get 0.5 seconds around merger (ringdown is ~0.01s after peak)
        strain, times = event_strain(
            event_name,
            detector,
            gps_time - 0.1,  # Start before merger
            gps_time + 0.4,  # Continue through ringdown
            sample_rate=4096  # High resolution
        )
        print(f"Downloaded {len(strain)} samples at {4096} Hz")
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

    # Isolate ringdown portion (after peak strain)
    peak_idx = np.argmax(np.abs(strain))
    peak_time = times[peak_idx]

    # Ringdown starts shortly after peak
    ringdown_start_idx = peak_idx + int(0.002 * 4096)  # 2ms after peak
    ringdown_end_idx = peak_idx + int(0.050 * 4096)    # 50ms window

    strain_ringdown = strain[ringdown_start_idx:ringdown_end_idx]
    times_ringdown = times[ringdown_start_idx:ringdown_end_idx] - peak_time

    print(f"Ringdown window: {times_ringdown[0]*1e3:.1f} to {times_ringdown[-1]*1e3:.1f} ms after peak")

    # Compute power spectral density
    fs = 4096  # Sample rate
    freqs, psd = signal.welch(
        strain_ringdown,
        fs=fs,
        nperseg=min(1024, len(strain_ringdown)//4),
        window='hann'
    )

    # Find fundamental QNM (expect ~250 Hz for GW150914)
    # Look in range 150-400 Hz
    freq_mask = (freqs > 150) & (freqs < 400)
    freqs_roi = freqs[freq_mask]
    psd_roi = psd[freq_mask]

    # Find peak (fundamental)
    fundamental_idx = np.argmax(psd_roi)
    f0 = freqs_roi[fundamental_idx]
    A0 = psd_roi[fundamental_idx]

    print(f"\nFundamental QNM detected: {f0:.1f} Hz")

    # Predict φ-harmonics
    predicted_harmonics = []
    detected_harmonics = []

    print("\nSearching for φ-harmonics:")
    print(f"{'n':<5} {'Predicted f (Hz)':<20} {'Expected Amplitude':<20} {'Detected?':<15}")
    print("-" * 70)

    for n in range(1, 6):
        f_n_predicted = f0 * PHI**n
        A_n_predicted = A0 * PHI**(-7*n)

        predicted_harmonics.append({
            'n': n,
            'freq': f_n_predicted,
            'amplitude': A_n_predicted
        })

        # Search within ±5% tolerance
        tolerance = 0.05
        search_mask = (freqs > f_n_predicted * (1-tolerance)) & (freqs < f_n_predicted * (1+tolerance))

        if np.any(search_mask):
            freqs_search = freqs[search_mask]
            psd_search = psd[search_mask]

            # Find local maximum
            if len(psd_search) > 0:
                local_peak_idx = np.argmax(psd_search)
                f_detected = freqs_search[local_peak_idx]
                A_detected = psd_search[local_peak_idx]

                # Calculate signal-to-noise ratio
                # Noise floor = median of surrounding frequencies
                noise_mask = (freqs > f_n_predicted * 0.8) & (freqs < f_n_predicted * 1.2)
                noise_floor = np.median(psd[noise_mask])
                snr = A_detected / noise_floor

                detected = snr > 2.0  # 2-sigma detection threshold

                detected_harmonics.append({
                    'n': n,
                    'freq_predicted': f_n_predicted,
                    'freq_detected': f_detected,
                    'amplitude': A_detected,
                    'snr': snr,
                    'detected': detected
                })

                status = "✅ YES" if detected else "⚠️ WEAK"
                print(f"{n:<5} {f_n_predicted:<20.1f} {A_n_predicted:<20.2e} {status:<15} (SNR={snr:.1f})")
            else:
                print(f"{n:<5} {f_n_predicted:<20.1f} {A_n_predicted:<20.2e} {'❌ NO':<15}")
        else:
            print(f"{n:<5} {f_n_predicted:<20.1f} {A_n_predicted:<20.2e} {'❌ NO':<15}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Ringdown strain
    ax1.plot(times_ringdown * 1e3, strain_ringdown, 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_xlabel('Time after peak (ms)')
    ax1.set_ylabel('Strain')
    ax1.set_title(f'{event_name} {detector} - Ringdown Strain')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Power spectrum with predicted harmonics
    ax2.loglog(freqs, psd, 'k-', linewidth=0.5, alpha=0.5, label='PSD')

    # Mark fundamental
    ax2.axvline(f0, color='blue', linestyle='--', linewidth=2, label=f'Fundamental ({f0:.0f} Hz)')

    # Mark predicted φ-harmonics
    colors = ['red', 'orange', 'green', 'purple', 'brown']
    for i, h in enumerate(predicted_harmonics):
        ax2.axvline(
            h['freq'],
            color=colors[i],
            linestyle=':',
            linewidth=1.5,
            alpha=0.7,
            label=f"φ^{h['n']} ({h['freq']:.0f} Hz)"
        )

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_title(f'{event_name} {detector} - φ-Harmonic Search')
    ax2.set_xlim(100, 3000)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    output_file = f'{event_name}_{detector}_phi_harmonics.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_file}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    n_detected = sum(1 for h in detected_harmonics if h['detected'])
    print(f"Fundamental: {f0:.1f} Hz")
    print(f"Harmonics detected: {n_detected}/5")

    if n_detected >= 2:
        print("✅ RESULT: φ-harmonic structure DETECTED")
    elif n_detected == 1:
        print("⚠️ RESULT: Weak evidence, needs stacking")
    else:
        print("❌ RESULT: No φ-harmonics detected in this event")

    print("\nNote: Single-event SNR may be too low.")
    print("Next step: Stack multiple events to increase sensitivity.")

    return {
        'event': event_name,
        'detector': detector,
        'fundamental': f0,
        'predicted_harmonics': predicted_harmonics,
        'detected_harmonics': detected_harmonics,
        'n_detected': n_detected
    }


if __name__ == "__main__":
    print("="*70)
    print("φ-HARMONIC RINGDOWN ANALYSIS")
    print("="*70)
    print()

    if GWOSC_AVAILABLE:
        # Test with GW150914 (first detection, well-studied)
        results = analyze_ringdown_harmonics('GW150914', 'H1')

        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print()
        print("1. Analyze multiple events: GW150914, GW151226, GW170814, GW190521")
        print("2. Stack power spectra to increase SNR")
        print("3. Perform statistical test: φ-harmonics vs random frequencies")
        print("4. If 3+ events show harmonics → Strong evidence")
        print("5. If no events show harmonics → Model rejected")
    else:
        print("\nPlease install gwosc to run analysis:")
        print("  pip install gwosc")
