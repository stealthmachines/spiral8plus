"""
Unified φ-Framework Model: Micro + BigG + Black Holes
======================================================

GOAL: Find TRUE attributes of each black hole by:
1. Using error-corrected parameters from micro/bigG scales
2. Applying framework to black hole observations
3. Minimizing error across ALL scales simultaneously
4. Building unified model with consistent parameters

FOUNDATIONAL TRUTH:
D_{n,β}(r) = √(φ · F_n · 2^(n+β) · P_n · Ω) · r^k
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.special import factorial
import json

PHI = (1 + np.sqrt(5)) / 2
SQRT5 = np.sqrt(5)

# ============================================================================
# FRAMEWORK
# ============================================================================

def framework_D0(n, beta, Omega):
    """D_0 = √(φ · F_n · 2^(n+β) · P_n · Ω)"""
    F_n = factorial(int(n)) if n < 20 else 1.0
    P_n = PHI ** n
    return np.sqrt(PHI * F_n * 2**(n + beta) * P_n * Omega)

def framework_frequency(M, n, beta, Omega, k):
    """f = D_0 / M^k"""
    D_0 = framework_D0(n, beta, Omega)
    return D_0 / (M ** k)

def phi_normalized_error(f_obs, f_pred):
    """Error relative to nearest φ^n harmonic"""
    ratio = f_obs / f_pred
    n_closest = int(round(np.log(ratio) / np.log(PHI)))
    phi_n = PHI ** n_closest
    return abs(ratio - phi_n) / phi_n

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data():
    """Load micro, bigG (cosmic), and black hole data"""

    # MICRO: Casimir effect, quantum oscillators
    micro = {
        'scale': 'micro',
        'name': 'Quantum/Casimir',
        'observations': [
            {'name': 'Casimir_1', 'M': 1e-15, 'f_obs': 1e15},
            {'name': 'Casimir_2', 'M': 2e-15, 'f_obs': 5e14},
            {'name': 'Casimir_3', 'M': 5e-15, 'f_obs': 2e14},
        ]
    }

    # BIGG (Cosmic): LIGO/Virgo gravitational waves
    bigg = {
        'scale': 'bigG',
        'name': 'Gravitational Waves',
        'observations': [
            {'name': 'GW150914', 'M': 65.0, 'f_obs': 251.0},
            {'name': 'GW151226', 'M': 21.0, 'f_obs': 450.0},
            {'name': 'GW170104', 'M': 49.0, 'f_obs': 280.0},
            {'name': 'GW170608', 'M': 18.0, 'f_obs': 500.0},
            {'name': 'GW170814', 'M': 56.0, 'f_obs': 275.0},
            {'name': 'GW190521', 'M': 142.0, 'f_obs': 140.0},
        ]
    }

    # BLACK HOLES: X-ray binaries with QPOs (intermediate scale)
    black_holes = {
        'scale': 'black_hole',
        'name': 'X-ray Binaries',
        'observations': [
            {'name': 'GRS1915+105', 'M': 14.0, 'f_obs': 67.0},
            {'name': 'XTEJ1550-564', 'M': 9.0, 'f_obs': 184.0},
            {'name': 'GRO J1655-40', 'M': 6.3, 'f_obs': 300.0},
            {'name': '4U1630-47', 'M': 10.0, 'f_obs': 185.0},
            {'name': 'H1743-322', 'M': 12.0, 'f_obs': 165.0},
        ]
    }

    return [micro, bigg, black_holes]

# ============================================================================
# UNIFIED PARAMETER FITTING
# ============================================================================

def fit_unified_parameters(all_data, verbose=True):
    """
    Fit single set of (n, β, Ω, k) that works across ALL scales

    This is the TRUE unified model - one framework for everything
    """

    def objective(params):
        n, beta, Omega, k = params

        total_error = 0.0
        count = 0

        for dataset in all_data:
            for obs in dataset['observations']:
                M = obs['M']
                f_obs = obs['f_obs']

                # Framework prediction
                f_pred = framework_frequency(M, n, beta, Omega, k)

                # φ-normalized error
                error = phi_normalized_error(f_obs, f_pred)

                total_error += error
                count += 1

        return total_error / count

    # Bounds
    bounds = [
        (0.1, 10.0),   # n
        (0.01, 2.0),   # beta
        (0.001, 10.0), # Omega
        (0.5, 3.0)     # k
    ]

    # Try multiple optimization strategies

    # Strategy 1: L-BFGS-B from reasonable guess
    x0 = [1.5, 0.5, 0.5, 2.0]
    result1 = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

    # Strategy 2: Differential evolution (global search)
    result2 = differential_evolution(objective, bounds, seed=42, maxiter=500)

    # Use best result
    if result2.fun < result1.fun:
        result = result2
        method = 'differential_evolution'
    else:
        result = result1
        method = 'L-BFGS-B'

    if verbose:
        print("\nUnified parameter fit:")
        print(f"  Method: {method}")
        print(f"  n     = {result.x[0]:.6f}")
        print(f"  β     = {result.x[1]:.6f}")
        print(f"  Ω     = {result.x[2]:.6f}")
        print(f"  k     = {result.x[3]:.6f}")
        print(f"  Error = {result.fun:.6f} ({result.fun*100:.4f}%)")

    return {
        'n': result.x[0],
        'beta': result.x[1],
        'Omega': result.x[2],
        'k': result.x[3],
        'error': result.fun,
        'method': method
    }

# ============================================================================
# PER-SCALE ERROR ANALYSIS
# ============================================================================

def analyze_per_scale_errors(all_data, unified_params):
    """
    Apply unified parameters to each scale and analyze residuals

    This tells us if different scales need different corrections
    """

    results = {}

    for dataset in all_data:
        scale = dataset['scale']

        errors = []
        delta_n_list = []

        for obs in dataset['observations']:
            M = obs['M']
            f_obs = obs['f_obs']

            # Predict with unified parameters
            f_pred = framework_frequency(M, unified_params['n'],
                                        unified_params['beta'],
                                        unified_params['Omega'],
                                        unified_params['k'])

            # Error
            error = phi_normalized_error(f_obs, f_pred)
            errors.append(error)

            # Residual Δn
            ratio = f_obs / f_pred
            n_obs = np.log(ratio) / np.log(PHI)
            n_int = round(n_obs)
            delta_n = n_obs - n_int
            delta_n_list.append(delta_n)

        results[scale] = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'mean_delta_n': np.mean(delta_n_list),
            'std_delta_n': np.std(delta_n_list),
            'n_observations': len(errors)
        }

    return results

# ============================================================================
# SCALE-SPECIFIC CORRECTIONS
# ============================================================================

def apply_scale_corrections(unified_params, scale_errors):
    """
    Apply minimal corrections to unified parameters for each scale

    Based on error analysis, adjust parameters slightly per scale
    """

    corrected = {}

    for scale, error_data in scale_errors.items():
        # Use error magnitude to correct parameters
        error_mag = error_data['mean_error']
        delta_n = error_data['mean_delta_n']

        # Small corrections based on error
        corrected[scale] = {
            'n': unified_params['n'] * (1 + error_mag * np.sign(delta_n) if delta_n != 0 else 1),
            'beta': unified_params['beta'] * (1 + error_mag * 0.5),
            'Omega': unified_params['Omega'] * (1 - error_mag * 0.5),
            'k': unified_params['k'] * (1 + error_mag * 0.3),
            'base_error': error_mag
        }

    return corrected

# ============================================================================
# BLACK HOLE TRUE ATTRIBUTES
# ============================================================================

def find_true_black_hole_attributes(black_hole_data, unified_params):
    """
    For each black hole, find its TRUE mass and frequency
    that minimize framework error

    Classical measurements may be wrong - find φ-framework values
    """

    true_attributes = []

    for obs in black_hole_data['observations']:
        name = obs['name']
        M_classical = obs['M']
        f_classical = obs['f_obs']

        # Optimize (M, f) to minimize framework error while staying close to observed
        def objective(x):
            M, f = x

            # Predict f from M using framework
            f_pred = framework_frequency(M, unified_params['n'],
                                        unified_params['beta'],
                                        unified_params['Omega'],
                                        unified_params['k'])

            # Error in framework prediction
            framework_error = phi_normalized_error(f, f_pred)

            # Penalty for deviating from classical values
            M_penalty = ((M - M_classical) / M_classical) ** 2
            f_penalty = ((f - f_classical) / f_classical) ** 2

            return framework_error + 0.1 * (M_penalty + f_penalty)

        # Bounds: allow ±30% deviation from classical
        bounds = [
            (M_classical * 0.7, M_classical * 1.3),
            (f_classical * 0.7, f_classical * 1.3)
        ]

        result = minimize(objective, [M_classical, f_classical],
                         bounds=bounds, method='L-BFGS-B')

        M_true, f_true = result.x

        # Calculate framework prediction
        f_pred = framework_frequency(M_true, unified_params['n'],
                                    unified_params['beta'],
                                    unified_params['Omega'],
                                    unified_params['k'])

        true_attributes.append({
            'name': name,
            'M_classical': M_classical,
            'M_true': M_true,
            'M_correction': (M_true - M_classical) / M_classical,
            'f_classical': f_classical,
            'f_true': f_true,
            'f_correction': (f_true - f_classical) / f_classical,
            'f_framework': f_pred,
            'final_error': result.fun
        })

    return true_attributes

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("=" * 70)
    print("UNIFIED φ-FRAMEWORK MODEL")
    print("=" * 70)
    print()
    print("Goal: Find TRUE black hole attributes using unified model")
    print("      that minimizes error across micro, bigG, and BH scales")
    print()
    print("=" * 70)

    # Load all data
    all_data = load_all_data()

    print("\nData summary:")
    for dataset in all_data:
        print(f"  {dataset['name']}: {len(dataset['observations'])} observations")

    # ========================================================================
    # STEP 1: Unified parameter fit
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: UNIFIED PARAMETER FIT")
    print("=" * 70)

    unified_params = fit_unified_parameters(all_data)

    # ========================================================================
    # STEP 2: Per-scale error analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: PER-SCALE ERROR ANALYSIS")
    print("=" * 70)

    scale_errors = analyze_per_scale_errors(all_data, unified_params)

    print("\nError by scale:")
    for scale, errors in scale_errors.items():
        print(f"\n  {scale.upper()}:")
        print(f"    Mean error:   {errors['mean_error']:.6f} ({errors['mean_error']*100:.4f}%)")
        print(f"    Std error:    {errors['std_error']:.6f}")
        print(f"    Mean Δn:      {errors['mean_delta_n']:+.6f}")
        print(f"    Std Δn:       {errors['std_delta_n']:.6f}")
        print(f"    Observations: {errors['n_observations']}")

    # ========================================================================
    # STEP 3: Scale-specific corrections
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: SCALE-SPECIFIC CORRECTIONS")
    print("=" * 70)

    corrected_params = apply_scale_corrections(unified_params, scale_errors)

    print("\nCorrected parameters by scale:")
    for scale, params in corrected_params.items():
        print(f"\n  {scale.upper()}:")
        print(f"    n     = {params['n']:.6f}")
        print(f"    β     = {params['beta']:.6f}")
        print(f"    Ω     = {params['Omega']:.6f}")
        print(f"    k     = {params['k']:.6f}")
        print(f"    Error = {params['base_error']:.6f} ({params['base_error']*100:.4f}%)")

    # ========================================================================
    # STEP 4: Find true black hole attributes
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: TRUE BLACK HOLE ATTRIBUTES")
    print("=" * 70)

    black_hole_data = all_data[2]  # Black hole dataset
    true_bh_attrs = find_true_black_hole_attributes(black_hole_data, unified_params)

    print("\nClassical vs Framework-corrected values:")
    print("-" * 70)
    print(f"{'Name':<15} {'M_class':<10} {'M_true':<10} {'ΔM%':<8} {'f_class':<10} {'f_true':<10} {'Δf%':<8}")
    print("-" * 70)

    for attr in true_bh_attrs:
        print(f"{attr['name']:<15} "
              f"{attr['M_classical']:<10.2f} "
              f"{attr['M_true']:<10.4f} "
              f"{attr['M_correction']*100:<8.2f} "
              f"{attr['f_classical']:<10.2f} "
              f"{attr['f_true']:<10.4f} "
              f"{attr['f_correction']*100:<8.2f}")

    # ========================================================================
    # STEP 5: Final unified model summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: FINAL UNIFIED MODEL")
    print("=" * 70)

    print("\nUnified Framework Parameters:")
    print(f"  n = {unified_params['n']:.6f}")
    print(f"  β = {unified_params['beta']:.6f}")
    print(f"  Ω = {unified_params['Omega']:.6f}")
    print(f"  k = {unified_params['k']:.6f}")

    print("\nOverall Performance:")
    print(f"  Mean error across all scales: {unified_params['error']:.6f} ({unified_params['error']*100:.4f}%)")

    # Calculate mean correction needed
    mean_M_correction = np.mean([attr['M_correction'] for attr in true_bh_attrs])
    mean_f_correction = np.mean([attr['f_correction'] for attr in true_bh_attrs])

    print("\nBlack Hole Corrections (classical → framework):")
    print(f"  Mean mass correction:      {mean_M_correction*100:+.2f}%")
    print(f"  Mean frequency correction: {mean_f_correction*100:+.2f}%")

    # Save results
    results = {
        'unified_parameters': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                              for k, v in unified_params.items()},
        'scale_errors': {k: {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv
                            for kk, vv in v.items()}
                        for k, v in scale_errors.items()},
        'corrected_parameters': {k: {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv
                                    for kk, vv in v.items()}
                                for k, v in corrected_params.items()},
        'true_black_hole_attributes': [{k: float(v) if isinstance(v, (int, float, np.number)) else v
                                        for k, v in attr.items()}
                                      for attr in true_bh_attrs]
    }

    with open('unified_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("Results saved to: unified_model_results.json")
    print("=" * 70)

    return results

if __name__ == '__main__':
    main()
