"""
Additional tests for scale_dependent_model.py edge cases
=======================================================
"""

import numpy as np
from scale_dependent_model import *

def test_edge_cases():
    print("TESTING EDGE CASES")
    print("=" * 50)

    # Test 1: Very small and very large masses
    print("\n1. Extreme mass scales:")
    extreme_masses = [1e-20, 1e-10, 1e10, 1e20]

    for M in extreme_masses:
        try:
            params = interpolate_parameters_for_scale(M)
            print(f"M={M:.0e}: Success - t={params['interpolation_factor']:.3f}")
        except Exception as e:
            print(f"M={M:.0e}: Error - {e}")

    # Test 2: Framework consistency at anchor points
    print("\n2. Framework at anchor points:")

    # Micro scale
    micro = SCALE_PARAMETERS['micro']
    f_micro = framework_frequency(1e-15, micro['n'], micro['beta'], micro['Omega'], micro['k'])
    print(f"Micro framework output: {f_micro:.6e}")

    # Cosmic scale
    cosmic = SCALE_PARAMETERS['cosmic']
    f_cosmic = framework_frequency(50.0, cosmic['n'], cosmic['beta'], cosmic['Omega'], cosmic['k'])
    print(f"Cosmic framework output: {f_cosmic:.6e}")

    # Test 3: φ-ratio calculations
    print("\n3. φ-ratio calculations:")

    phi_powers = [PHI**n for n in range(1, 15)]
    test_ratios = [1.5, 2.7, 4.2, 6.9, 11.1]

    for ratio in test_ratios:
        error = phi_normalized_error(ratio, 1.0)
        closest_n = int(round(np.log(ratio) / np.log(PHI)))
        print(f"Ratio {ratio:.1f}: closest φ^{closest_n}, error={error:.6f}")

    # Test 4: Parameter bounds
    print("\n4. Parameter interpolation bounds:")

    # Test interpolation at exact anchor points
    micro_interp = interpolate_parameters_for_scale(1e-15)
    cosmic_interp = interpolate_parameters_for_scale(50.0)

    print(f"At micro mass (1e-15): t={micro_interp['interpolation_factor']:.6f}")
    print(f"At cosmic mass (50): t={cosmic_interp['interpolation_factor']:.6f}")

    # Test 5: Mathematical stability
    print("\n5. Mathematical stability:")

    try:
        # Test factorial handling
        test_n = [1.5, 5.7, 15.2, 25.0]
        for n in test_n:
            D0 = framework_D0(n, 0.5, 1.0)
            print(f"n={n}: D0={D0:.6e} (finite: {np.isfinite(D0)})")
    except Exception as e:
        print(f"Stability test failed: {e}")

def test_physics_consistency():
    print("\n\nPHYSICS CONSISTENCY CHECKS")
    print("=" * 50)

    # Test 1: Dimensional analysis
    print("\n1. Dimensional consistency:")

    # Framework should give frequency in Hz when M is in solar masses
    M_sun = 1.0  # Solar masses
    bh_data = load_black_hole_data()
    bh_params = fit_black_hole_parameters(bh_data)

    f_pred = framework_frequency(M_sun, bh_params['n'], bh_params['beta'],
                                bh_params['Omega'], bh_params['k'])

    print(f"1 M☉ predicted frequency: {f_pred:.2f} Hz")
    print(f"Reasonable range for stellar BH QPO: {10 <= f_pred <= 1000}")

    # Test 2: Scaling behavior
    print("\n2. Scaling behavior:")

    masses = [1, 10, 100]
    for M in masses:
        f = framework_frequency(M, bh_params['n'], bh_params['beta'],
                               bh_params['Omega'], bh_params['k'])
        print(f"M={M:3.0f} M☉: f={f:8.3f} Hz")

    # Should roughly follow f ∝ M^(-k)
    f1 = framework_frequency(1, bh_params['n'], bh_params['beta'],
                            bh_params['Omega'], bh_params['k'])
    f10 = framework_frequency(10, bh_params['n'], bh_params['beta'],
                             bh_params['Omega'], bh_params['k'])

    expected_ratio = 10**bh_params['k']
    actual_ratio = f1 / f10

    print(f"f(1M☉)/f(10M☉): expected={expected_ratio:.3f}, actual={actual_ratio:.3f}")
    print(f"Scaling consistent: {abs(expected_ratio - actual_ratio) < 0.1}")

if __name__ == '__main__':
    test_edge_cases()
    test_physics_consistency()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)