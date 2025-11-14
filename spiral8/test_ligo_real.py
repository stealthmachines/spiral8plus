"""
Test LIGO Predictions Against Real Data
========================================

Takes the micro-tuned LIGO parameters (n=1.37, β=0.41, Ω=0.14)
and tests them against actual LIGO gravitational wave events.

Downloads real strain data and searches for 1% echoes at ~100 μs.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, find_peaks, butter, sosfiltfilt
import json
from pathlib import Path

# Load optimized parameters from comprehensive validation
results_file = Path('comprehensive_validation_results.json')
if results_file.exists():
    with open(results_file, 'r') as f:
        results = json.load(f)

    n_opt = results['ligo_scale']['n']
    beta_opt = results['ligo_scale']['beta']
    Omega_opt = results['ligo_scale']['Omega']
    k_opt = results['ligo_scale']['k']

    print("="*70)
    print("TESTING MICRO-TUNED LIGO PREDICTIONS AGAINST REAL DATA")
    print("="*70)
    print(f"\nOptimized Parameters (from comprehensive validation):")
    print(f"  n = {n_opt:.4f}")
    print(f"  β = {beta_opt:.4f}")
    print(f"  Ω = {Omega_opt:.4f}")
    print(f"  k = {k_opt:.4f}")
else:
    print("ERROR: Run comprehensive_validation.py first!")
    exit(1)

# Constants
PHI = (1 + np.sqrt(5)) / 2
SPEED_C = 299792458  # m/s
GRAV_G = 6.67430e-11  # m³/(kg·s²)
M_SUN = 1.98847e30  # kg

# Check if GWOSC is available
try:
    from gwosc.datasets import event_gps
    from gwosc import TimeSeries
    GWOSC_AVAILABLE = True
    print("\n✓ GWOSC available - will download real LIGO data")
except ImportError:
    GWOSC_AVAILABLE = False
    print("\n⚠ GWOSC not available - will use simulated data")

# LIGO events to test
EVENTS = [
    ('GW150914', 36, 29),   # First detection
    ('GW170814', 31, 25),   # First 3-detector
    ('GW151226', 14, 8),    # Light system
]

def calculate_echo_prediction(M_solar, n, beta, Omega, k):
    """
    Calculate echo properties using dimensional DNA framework

    USES SAME FORMULA AS MICRO AND BIGG:
    scale_factor = √(φ · F_n · 2^(n+β) · P_n · Ω)
    """
    M_kg = M_solar * M_SUN
    r_s = 2 * GRAV_G * M_kg / SPEED_C**2
    t_cross = 2 * r_s / SPEED_C

    # Dimensional DNA scaling
    PHI_7 = PHI**7
    F_n = PHI**n
    P_n = 2 + n
    scale_factor = np.sqrt(PHI * F_n * 2**(n+beta) * P_n * Omega)

    # Echo delay
    tau_echo = t_cross / PHI_7 * scale_factor

    # Echo amplitude
    A_base = 1.0 / (PHI_7**n)
    damping = np.exp(-beta * Omega / 10.0)
    A_echo = A_base * damping

    return tau_echo * 1e6, A_echo * 100  # μs, %

def generate_simulated_signal(t, M1, M2, add_echo=True):
    """
    Generate simulated GW signal with ringdown and optional echo
    """
    M_total = M1 + M2
    fs = 1 / (t[1] - t[0])

    # Calculate echo properties
    tau_echo_us, amp_echo_pct = calculate_echo_prediction(M_total, n_opt, beta_opt, Omega_opt, k_opt)
    tau_echo = tau_echo_us * 1e-6  # Convert to seconds

    # QNM parameters (approximate for M_total)
    M_geo = M_total * M_SUN * GRAV_G / SPEED_C**3
    f_qnm = (1 - 0.63 * 0.3) / (2 * np.pi * M_geo)
    tau_ringdown = 3.0 / (2 * np.pi * f_qnm)

    # Primary ringdown (starts at t=0)
    mask_ring = t > 0
    h_ring = np.zeros_like(t)
    h_ring[mask_ring] = np.exp(-t[mask_ring] / tau_ringdown) * np.sin(2 * np.pi * f_qnm * t[mask_ring])

    # Add echo if requested
    h_total = h_ring.copy()
    if add_echo:
        # Echo arrives at tau_echo after merger
        mask_echo = t > tau_echo
        if np.any(mask_echo):
            echo_amp = amp_echo_pct / 100.0
            t_shifted = t[mask_echo] - tau_echo
            h_echo = echo_amp * np.exp(-t_shifted / tau_ringdown) * np.sin(2 * np.pi * f_qnm * t_shifted)
            h_total[mask_echo] += h_echo

    # Add noise
    noise_level = 0.3 * np.max(np.abs(h_ring))
    noise = np.random.normal(0, noise_level, len(t))
    h_total += noise

    return h_total, tau_echo, amp_echo_pct, f_qnm

def analyze_for_echoes(t, h, M_total, tau_expected_us, amp_expected_pct):
    """
    Search for echo signals in strain data
    """
    fs = 1 / (t[1] - t[0])

    # Bandpass filter (100-600 Hz for stellar mass BH)
    sos = butter(4, [100, 600], btype='bandpass', fs=fs, output='sos')
    h_filt = sosfiltfilt(sos, h)

    # Calculate envelope
    analytic = hilbert(h_filt)
    envelope = np.abs(analytic)

    # Smooth envelope (1 ms window)
    window = int(0.001 * fs)
    if window % 2 == 0:
        window += 1
    window = max(5, window)  # Minimum window size
    if window >= len(envelope):
        window = len(envelope) // 2
        if window % 2 == 0:
            window += 1
        window = max(5, window)

    from scipy.signal import savgol_filter
    polyorder = min(3, window - 1)  # Ensure polyorder < window_length
    envelope_smooth = savgol_filter(envelope, window, polyorder)    # Find peaks
    prominence = 0.05 * np.max(envelope_smooth)
    min_distance = int(0.02 * fs)  # 20 ms minimum separation
    peaks, properties = find_peaks(envelope_smooth, prominence=prominence, distance=min_distance)

    if len(peaks) == 0:
        return None, envelope_smooth, peaks

    # Analyze peaks
    peak_times_us = t[peaks] * 1e6
    peak_amps = envelope_smooth[peaks]
    max_amp = np.max(envelope_smooth[:len(envelope_smooth)//4])  # Max in first quarter (ringdown)
    peak_amps_pct = (peak_amps / max_amp) * 100

    # Find echo candidate (peak near expected time)
    echo_candidate = None
    for i, (t_pk, amp_pk) in enumerate(zip(peak_times_us, peak_amps_pct)):
        # Check if near expected echo time (within 50%)
        if abs(t_pk - tau_expected_us) < 0.5 * tau_expected_us:
            # Check if amplitude matches (within factor of 3)
            if 0.3 * amp_expected_pct < amp_pk < 3.0 * amp_expected_pct:
                echo_candidate = {
                    'index': i,
                    'time_us': t_pk,
                    'amplitude_pct': amp_pk,
                    'time_error': abs(t_pk - tau_expected_us) / tau_expected_us,
                    'amp_error': abs(amp_pk - amp_expected_pct) / amp_expected_pct
                }
                break

    return echo_candidate, envelope_smooth, peaks

def create_ligo_test_plots():
    """
    Create comprehensive LIGO test plots
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    results_summary = []

    for idx, (event_name, M1, M2) in enumerate(EVENTS):
        M_total = M1 + M2

        print(f"\n{'='*70}")
        print(f"EVENT: {event_name} ({M1} + {M2} = {M_total} M☉)")
        print(f"{'='*70}")

        # Calculate predictions
        tau_pred_us, amp_pred_pct = calculate_echo_prediction(M_total, n_opt, beta_opt, Omega_opt, k_opt)

        print(f"\nPredictions:")
        print(f"  Echo delay:     {tau_pred_us:.1f} μs")
        print(f"  Echo amplitude: {amp_pred_pct:.2f}%")

        # Generate or download data
        if GWOSC_AVAILABLE:
            try:
                print(f"\nDownloading real data from GWOSC...")
                gps = event_gps(event_name)
                strain = TimeSeries.fetch_open_data('H1', gps - 0.5, gps + 0.5, sample_rate=4096)
                t = strain.times.value - gps
                h = strain.value
                data_source = "Real LIGO Data"
                print(f"  ✓ Downloaded {len(h)} samples")
            except Exception as e:
                print(f"  ✗ Download failed: {e}")
                print(f"  Using simulated data instead")
                t = np.linspace(-0.1, 0.5, 2048)
                h, _, _, _ = generate_simulated_signal(t, M1, M2, add_echo=True)
                data_source = "Simulated"
        else:
            t = np.linspace(-0.1, 0.5, 2048)
            h, _, _, _ = generate_simulated_signal(t, M1, M2, add_echo=True)
            data_source = "Simulated"

        # Analyze for echoes
        echo_found, envelope, peaks = analyze_for_echoes(t, h, M_total, tau_pred_us, amp_pred_pct)

        if echo_found:
            print(f"\n✓ ECHO CANDIDATE FOUND:")
            print(f"  Time:      {echo_found['time_us']:.1f} μs (expected {tau_pred_us:.1f} μs)")
            print(f"  Amplitude: {echo_found['amplitude_pct']:.2f}% (expected {amp_pred_pct:.2f}%)")
            print(f"  Time error:      {echo_found['time_error']*100:.1f}%")
            print(f"  Amplitude error: {echo_found['amp_error']*100:.1f}%")
            detection_status = "DETECTED"
        else:
            print(f"\n✗ No echo candidate found near {tau_pred_us:.1f} μs")
            detection_status = "NOT DETECTED"

        results_summary.append({
            'event': event_name,
            'mass': M_total,
            'predicted_tau': tau_pred_us,
            'predicted_amp': amp_pred_pct,
            'echo_found': echo_found,
            'detection': detection_status,
            'data_source': data_source
        })

        # PLOT THIS EVENT
        row = idx

        # Column 1: Raw strain + prediction marker
        ax1 = fig.add_subplot(gs[row, 0])
        t_ms = t * 1000  # Convert to ms
        ax1.plot(t_ms, h, 'b-', linewidth=0.5, alpha=0.7)
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Merger', alpha=0.7)
        ax1.axvline(tau_pred_us / 1000, color='orange', linestyle='--', linewidth=2,
                   label=f'Echo @ {tau_pred_us:.0f}μs', alpha=0.7)
        ax1.set_xlabel('Time (ms)', fontsize=10)
        ax1.set_ylabel('Strain', fontsize=10)
        ax1.set_title(f'{event_name}: Raw Strain\n{data_source}', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-50, 300)

        # Column 2: Envelope with peaks
        ax2 = fig.add_subplot(gs[row, 1])
        t_us = t * 1e6  # Convert to μs
        ax2.plot(t_us, envelope, 'g-', linewidth=1.5, label='Envelope', alpha=0.8)
        if peaks is not None and len(peaks) > 0:
            ax2.plot(t_us[peaks], envelope[peaks], 'ro', markersize=8, label='Peaks')
        ax2.axvline(tau_pred_us, color='orange', linestyle='--', linewidth=2,
                   label=f'Predicted echo', alpha=0.7)

        if echo_found:
            ax2.axvline(echo_found['time_us'], color='lime', linestyle='-', linewidth=3,
                       label='Echo candidate!', alpha=0.9)

        ax2.set_xlabel('Time (μs)', fontsize=10)
        ax2.set_ylabel('Amplitude', fontsize=10)
        ax2.set_title(f'{event_name}: Envelope Analysis\n{detection_status}',
                     fontsize=11, fontweight='bold',
                     color='green' if echo_found else 'red')
        ax2.legend(fontsize=8, loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 500)

        # Column 3: Frequency spectrum
        ax3 = fig.add_subplot(gs[row, 2])

        # Focus on ringdown region (0 to 0.3 s)
        mask_ring = (t > 0) & (t < 0.3)
        if np.any(mask_ring):
            h_ring = h[mask_ring]
            fs = 1 / (t[1] - t[0])

            # FFT
            n_fft = len(h_ring)
            freqs = fftfreq(n_fft, 1/fs)
            fft_vals = fft(h_ring)

            # Positive frequencies only
            pos_mask = freqs > 0
            freqs_pos = freqs[pos_mask]
            power = np.abs(fft_vals[pos_mask])**2

            ax3.semilogy(freqs_pos, power, 'b-', linewidth=1, alpha=0.7)
            ax3.set_xlabel('Frequency (Hz)', fontsize=10)
            ax3.set_ylabel('Power', fontsize=10)
            ax3.set_title(f'{event_name}: Frequency Spectrum', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(50, 1000)

            # Mark QNM region
            ax3.axvspan(200, 400, color='yellow', alpha=0.2, label='QNM region')
            ax3.legend(fontsize=8)

    # Add overall title and summary
    fig.suptitle('LIGO Echo Predictions: Testing Micro-Tuned φ-Framework\n' +
                 f'Parameters: n={n_opt:.3f}, β={beta_opt:.3f}, Ω={Omega_opt:.3f}',
                 fontsize=14, fontweight='bold')

    # Save
    output_path = Path('plots') if Path('plots').exists() else Path('.')
    save_path = output_path / 'ligo_echo_test.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved: {save_path}")

    return fig, results_summary

# Run analysis
print("\n" + "="*70)
print("GENERATING LIGO TEST PLOTS")
print("="*70)

fig, results = create_ligo_test_plots()

# Save results
output_path = Path('output') if Path('output').exists() else Path('.')
results_file = output_path / 'ligo_echo_test_results.json'
with open(results_file, 'w') as f:
    json.dump({
        'timestamp': '2025-11-05',
        'parameters': {
            'n': n_opt,
            'beta': beta_opt,
            'Omega': Omega_opt,
            'k': k_opt
        },
        'events': results
    }, f, indent=2)

print(f"✓ Saved: {results_file}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

for res in results:
    status_icon = "✓" if res['detection'] == "DETECTED" else "✗"
    print(f"\n{status_icon} {res['event']} ({res['mass']} M☉):")
    print(f"  Predicted: {res['predicted_tau']:.1f} μs @ {res['predicted_amp']:.2f}%")
    print(f"  Status: {res['detection']}")
    print(f"  Data: {res['data_source']}")

print("\n" + "="*70)
print(f"\nAll plots saved to: plots/ligo_echo_test.png")
print("Open this file to see the LIGO test results!")
print("="*70)

plt.show()
