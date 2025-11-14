# validate_unified_synthesis.py
# ==============================================================================
# Validation Script for φ-Framework Unified Synthesis
# Tests all components without requiring GUI
# ==============================================================================

import json
import numpy as np
import sys
import os

# ==============================================================================
# CONSTANTS
# ==============================================================================

PHI = (1 + np.sqrt(5)) / 2
PHI_2 = PHI ** 2
PHI_7 = PHI ** 7
PHI_INV_7 = 1.0 / PHI_7

print("="*70)
print("φ-FRAMEWORK UNIFIED SYNTHESIS VALIDATION")
print("="*70)
print(f"φ = {PHI:.10f}")
print(f"φ⁷ = {PHI_7:.10f}")
print(f"φ⁻⁷ = {PHI_INV_7:.10f}")
print()

# ==============================================================================
# TEST 1: Framework Files
# ==============================================================================

print("TEST 1: Framework Files")
print("-" * 70)

files_to_check = [
    'ecoli_unified_phi_synthesis.py',
    'ecoli_k12.fasta',
    'complete_phi_framework_final.json',
    'codata_2022.json'
]

for fname in files_to_check:
    exists = os.path.exists(fname)
    status = "✅" if exists else "⚠️ "
    required = "Required" if fname.endswith('.py') or fname.endswith('.fasta') else "Optional"
    print(f"{status} {fname:<40} {required}")

print()

# ==============================================================================
# TEST 2: Load Framework Data
# ==============================================================================

print("TEST 2: Framework Data Loading")
print("-" * 70)

try:
    with open('complete_phi_framework_final.json', 'r') as f:
        framework = json.load(f)
    print("✅ φ-Framework loaded successfully")
    print(f"   Golden ratio: {framework['golden_ratio']:.10f}")
    print(f"   Base equation: {framework['base_equation'][:50]}...")

    # Validate coefficients
    coeffs = framework['scaling_law']['coefficients']
    print(f"   Cubic coefficients: a₃={coeffs['a3']:.6f}, a₂={coeffs['a2']:.6f}")
except Exception as e:
    print(f"⚠️  Could not load framework: {e}")
    framework = None

print()

try:
    with open('codata_2022.json', 'r') as f:
        codata = json.load(f)
    print("✅ CODATA 2022 loaded successfully")
    print(f"   Speed of light: {codata['speed_of_light_c']['value']:,} m/s")
    print(f"   Planck constant: {codata['planck_constant_h']['value']:.6e} J·s")
except Exception as e:
    print(f"⚠️  Could not load CODATA: {e}")
    codata = None

print()

# ==============================================================================
# TEST 3: Cubic Scaling Law
# ==============================================================================

print("TEST 3: Cubic Scaling Law")
print("-" * 70)

if framework:
    coeffs = framework['scaling_law']['coefficients']
    a3, a2, a1, a0 = coeffs['a3'], coeffs['a2'], coeffs['a1'], coeffs['a0']
else:
    # Fallback to φ-derived
    a3 = -PHI_2 / 50
    a2 = PHI / 3
    a1 = -PHI
    a0 = PHI / 3
    print("Using φ-derived fallback coefficients")

def compute_alpha(P):
    return a3*P**3 + a2*P**2 + a1*P + a0

print(f"α(P) = {a3:.6f}P³ + {a2:.6f}P² + {a1:.6f}P + {a0:.6f}")
print()
print("P    α(P)")
print("-" * 20)
for P in range(1, 9):
    alpha = compute_alpha(P)
    print(f"{P}    {alpha:+.6f}")

print()

# ==============================================================================
# TEST 4: φ-Harmonic Detection
# ==============================================================================

print("TEST 4: φ-Harmonic Detection")
print("-" * 70)

def compute_phi_harmonic(value, reference=1.0):
    ratio = value / reference
    n = round(np.log(ratio) / np.log(PHI))
    phi_n = PHI ** n
    error = abs(ratio - phi_n) / phi_n if phi_n != 0 else float('inf')
    return n, phi_n, error

test_values = [1.0, PHI, PHI**2, 2.5, PHI**3, 5.0, PHI**(-1)]
print("Value    Closest φⁿ      n      Error")
print("-" * 50)
for val in test_values:
    n, phi_n, error = compute_phi_harmonic(val)
    print(f"{val:6.3f}   {phi_n:8.4f}     {n:+3d}    {error*100:6.2f}%")

print()

# ==============================================================================
# TEST 5: DNA to Physics Mapping
# ==============================================================================

print("TEST 5: DNA to Physics Mapping")
print("-" * 70)

BASE_MAP = {'A': 5, 'T': 2, 'G': 4, 'C': 1}

test_codons = ['ATG', 'AAA', 'GGG', 'CCC', 'ACG', 'TAC']

print("Codon  Dims      P      α(P)    φⁿ   Error")
print("-" * 60)
for codon in test_codons:
    dims = [BASE_MAP.get(b, 1) for b in codon]
    P = np.mean(dims)
    alpha = compute_alpha(P)
    n, phi_n, error = compute_phi_harmonic(P)
    print(f"{codon}    {dims}   {P:.2f}   {alpha:+.4f}   φ^{n:+2d}  {error*100:5.2f}%")

print()

# ==============================================================================
# TEST 6: Cavity Resonance
# ==============================================================================

print("TEST 6: Cavity Resonance Properties")
print("-" * 70)

CAVITY_STRUCTURE = {
    'deep_interior': {'n_cascade': 3, 'Omega_base': 0.05, 'Q_range': (80, 100)},
    'photon_shell':  {'n_cascade': 2, 'Omega_base': 0.5,  'Q_range': (50, 80)},
    'weak_field':    {'n_cascade': 1, 'Omega_base': 1.0,  'Q_range': (20, 50)},
    'accretion_disk':{'n_cascade': 0, 'Omega_base': 2.5,  'Q_range': (5, 20)},
}

print("Cavity Type       n_cascade  Ω_base   Q_range    φ⁻⁷/√Q_avg")
print("-" * 70)
for name, props in CAVITY_STRUCTURE.items():
    Q_avg = np.mean(props['Q_range'])
    echo_amp = PHI_INV_7 / np.sqrt(Q_avg)
    print(f"{name:<17} {props['n_cascade']:^10} {props['Omega_base']:^8.2f} "
          f"{str(props['Q_range']):<12} {echo_amp:.6f}")

print()

# ==============================================================================
# TEST 7: Genome Analysis (if available)
# ==============================================================================

print("TEST 7: Sample Genome Analysis")
print("-" * 70)

fasta_file = 'ecoli_k12.fasta'

if os.path.exists(fasta_file):
    print(f"✅ Found {fasta_file}")

    # Load first 1000 bases
    sequence = []
    with open(fasta_file) as f:
        for line in f:
            if line.startswith(">"):
                continue
            sequence.extend(list(line.strip().upper()))
            if len(sequence) >= 1000:
                break

    print(f"   Loaded {len(sequence)} bases for testing")

    # Analyze codons
    phi_aligned = 0
    total_codons = 0
    alphas = []

    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i+3]
        dims = [BASE_MAP.get(b, 1) for b in codon]
        P = np.mean(dims)
        alpha = compute_alpha(P)
        n, phi_n, error = compute_phi_harmonic(P)

        alphas.append(alpha)
        total_codons += 1
        if error < 0.10:
            phi_aligned += 1

    print(f"   Total codons: {total_codons}")
    print(f"   φ-aligned (<10% error): {phi_aligned}")
    print(f"   φ-resonance: {phi_aligned/total_codons*100:.1f}%")
    print(f"   Mean α: {np.mean(alphas):.6f}")
    print(f"   Std α: {np.std(alphas):.6f}")
else:
    print(f"⚠️  {fasta_file} not found - skipping genome analysis")

print()

# ==============================================================================
# TEST 8: Dependencies Check
# ==============================================================================

print("TEST 8: Python Dependencies")
print("-" * 70)

dependencies = [
    ('numpy', 'Core numerical library'),
    ('vispy', 'GPU-accelerated visualization'),
    ('PyQt6', 'GUI backend for VisPy'),
]

all_ok = True
for module, desc in dependencies:
    try:
        __import__(module)
        print(f"✅ {module:<15} - {desc}")
    except ImportError:
        print(f"❌ {module:<15} - {desc} (NOT INSTALLED)")
        all_ok = False

print()

# ==============================================================================
# SUMMARY
# ==============================================================================

print("="*70)
print("VALIDATION SUMMARY")
print("="*70)

checks = [
    ("Framework files", os.path.exists('ecoli_unified_phi_synthesis.py')),
    ("FASTA file", os.path.exists('ecoli_k12.fasta')),
    ("φ-Framework data", framework is not None),
    ("CODATA constants", codata is not None),
    ("Python dependencies", all_ok),
]

total_passed = sum(1 for _, passed in checks if passed)
total_critical = 2  # Script + FASTA

print()
for check_name, passed in checks:
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {check_name}")

print()
print(f"Critical checks passed: {sum(1 for c in checks[:2] if c[1])}/{total_critical}")
print(f"Total checks passed: {total_passed}/{len(checks)}")
print()

if checks[0][1] and checks[1][1]:
    print("🎉 Ready to run: python ecoli_unified_phi_synthesis.py")
else:
    print("⚠️  Missing critical files. Check above for details.")

if not all_ok:
    print("\n📦 Install missing dependencies:")
    print("   pip install vispy pyqt6 numpy")

print()
print("="*70)
print(f"φ = {PHI:.10f} | φ⁷ = {PHI_7:.4f} | φ⁻⁷ = {PHI_INV_7:.6f}")
print("="*70)
