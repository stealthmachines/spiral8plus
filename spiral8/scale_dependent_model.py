"""
Scale-Dependent φ-Framework Model: Micro + BigG + Black Holes
==============================================================

CRITICAL UNDERSTANDING:
Parameters (n, β, Ω, k) MUST change with scale - this is the framework!

From clean_phi_validation.py results:
- MICRO:  n=1.400, β=0.512, Ω=0.114, k=2.135  (error: 6.68%)
- COSMIC: n=1.676, β=0.662, Ω=2.031, k=0.670  (error: 4.69%)

GOAL:
1. Use proven micro/bigG parameters as anchors
2. Interpolate to find black hole scale parameters
3. Find TRUE black hole attributes using scale-appropriate parameters

FOUNDATIONAL TRUTH:
D_{n,β}(r) = √(φ · F_n · 2^(n+β) · P_n · Ω) · r^k
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import json

PHI = (1 + np.sqrt(5)) / 2
SQRT5 = np.sqrt(5)

# ============================================================================
# PROVEN SCALE PARAMETERS (from clean_phi_validation.py)
# ============================================================================

SCALE_PARAMETERS = {
    'micro': {
        'n': 1.400033,
        'beta': 0.512254,
        'Omega': 0.113869,
        'k': 2.134879,
        'error': 0.066822,
        'typical_M': 1e-15,  # kg scale
        'log_M': -15
    },
    'cosmic': {
        'n': 1.676307,
        'beta': 0.662358,
        'Omega': 2.031371,
        'k': 0.670294,
        'error': 0.046862,
        'typical_M': 50.0,  # solar masses
        'log_M': np.log10(50)
    }
}

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
# SCALE INTERPOLATION
# ============================================================================

def interpolate_parameters_for_scale(typical_M):
    """
    Interpolate parameters for intermediate scale

    Uses log-linear interpolation between micro and cosmic scales
    """
    micro = SCALE_PARAMETERS['micro']
    cosmic = SCALE_PARAMETERS['cosmic']

    # Log scale position
    log_M = np.log10(typical_M) if typical_M > 0 else 0

    # Interpolation factor (0=micro, 1=cosmic)
    if cosmic['log_M'] != micro['log_M']:
        t = (log_M - micro['log_M']) / (cosmic['log_M'] - micro['log_M'])
        t = np.clip(t, 0, 1)  # Stay within bounds
    else:
        t = 0.5

    # Linear interpolation of each parameter
    n_interp = micro['n'] + t * (cosmic['n'] - micro['n'])
    beta_interp = micro['beta'] + t * (cosmic['beta'] - micro['beta'])
    Omega_interp = micro['Omega'] + t * (cosmic['Omega'] - micro['Omega'])
    k_interp = micro['k'] + t * (cosmic['k'] - micro['k'])

    return {
        'n': n_interp,
        'beta': beta_interp,
        'Omega': Omega_interp,
        'k': k_interp,
        'interpolation_factor': t,
        'log_M': log_M
    }

# ============================================================================
# DATA
# ============================================================================

def load_black_hole_data():
    """X-ray binaries with QPOs"""
    return [
        {'name': 'GRS1915+105', 'M': 14.0, 'f_obs': 67.0},
        {'name': 'XTEJ1550-564', 'M': 9.0, 'f_obs': 184.0},
        {'name': 'GRO J1655-40', 'M': 6.3, 'f_obs': 300.0},
        {'name': '4U1630-47', 'M': 10.0, 'f_obs': 185.0},
        {'name': 'H1743-322', 'M': 12.0, 'f_obs': 165.0},
    ]

# ============================================================================
# FIT BLACK HOLE SCALE PARAMETERS
# ============================================================================

def fit_black_hole_parameters(bh_data):
    """
    Fit parameters for black hole scale using interpolation as starting point

    Start from interpolated values, then optimize
    """

    # Get typical mass scale for black holes
    typical_M = np.mean([bh['M'] for bh in bh_data])

    # Interpolate as starting point
    initial = interpolate_parameters_for_scale(typical_M)

    print(f"Black hole typical mass: {typical_M:.2f} M☉")
    print(f"Interpolation starting point (t={initial['interpolation_factor']:.3f}):")
    print(f"  n = {initial['n']:.6f}")
    print(f"  β = {initial['beta']:.6f}")
    print(f"  Ω = {initial['Omega']:.6f}")
    print(f"  k = {initial['k']:.6f}")

    # Optimize from this starting point
    def objective(params):
        n, beta, Omega, k = params

        total_error = 0.0
        for bh in bh_data:
            f_pred = framework_frequency(bh['M'], n, beta, Omega, k)
            error = phi_normalized_error(bh['f_obs'], f_pred)
            total_error += error

        return total_error / len(bh_data)

    # Bounds: allow deviation from interpolated values
    bounds = [
        (initial['n'] * 0.8, initial['n'] * 1.2),
        (initial['beta'] * 0.8, initial['beta'] * 1.2),
        (initial['Omega'] * 0.5, initial['Omega'] * 2.0),
        (initial['k'] * 0.8, initial['k'] * 1.2)
    ]

    x0 = [initial['n'], initial['beta'], initial['Omega'], initial['k']]

    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

    return {
        'n': result.x[0],
        'beta': result.x[1],
        'Omega': result.x[2],
        'k': result.x[3],
        'error': result.fun,
        'interpolated': initial,
        'success': result.success
    }

# ============================================================================
# FIND TRUE BLACK HOLE ATTRIBUTES
# ============================================================================

def find_true_attributes(bh_data, bh_params):
    """
    Find TRUE (M, f) for each black hole using scale-appropriate parameters
    """

    results = []

    for bh in bh_data:
        name = bh['name']
        M_classical = bh['M']
        f_classical = bh['f_obs']

        # Optimize (M, f) to minimize framework error
        def objective(x):
            M, f = x

            # Framework prediction
            f_pred = framework_frequency(M, bh_params['n'], bh_params['beta'],
                                        bh_params['Omega'], bh_params['k'])

            # Error
            framework_error = phi_normalized_error(f, f_pred)

            # Penalty for deviating from classical
            M_penalty = ((M - M_classical) / M_classical) ** 2
            f_penalty = ((f - f_classical) / f_classical) ** 2

            return framework_error + 0.1 * (M_penalty + f_penalty)

        # Bounds: ±30% from classical
        bounds = [
            (M_classical * 0.7, M_classical * 1.3),
            (f_classical * 0.7, f_classical * 1.3)
        ]

        result = minimize(objective, [M_classical, f_classical],
                         bounds=bounds, method='L-BFGS-B')

        M_true, f_true = result.x

        # Framework prediction at true values
        f_framework = framework_frequency(M_true, bh_params['n'], bh_params['beta'],
                                         bh_params['Omega'], bh_params['k'])

        results.append({
            'name': name,
            'M_classical': M_classical,
            'M_true': M_true,
            'M_correction_%': (M_true - M_classical) / M_classical * 100,
            'f_classical': f_classical,
            'f_true': f_true,
            'f_correction_%': (f_true - f_classical) / f_classical * 100,
            'f_framework': f_framework,
            'final_error': phi_normalized_error(f_true, f_framework)
        })

    return results

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("=" * 70)
    print("SCALE-DEPENDENT φ-FRAMEWORK MODEL")
    print("=" * 70)
    print()
    print("Framework principle: Parameters change with scale!")
    print()
    print("Proven anchors:")
    print(f"  MICRO:  n={SCALE_PARAMETERS['micro']['n']:.3f}, "
          f"β={SCALE_PARAMETERS['micro']['beta']:.3f}, "
          f"Ω={SCALE_PARAMETERS['micro']['Omega']:.3f}, "
          f"k={SCALE_PARAMETERS['micro']['k']:.3f}")
    print(f"  COSMIC: n={SCALE_PARAMETERS['cosmic']['n']:.3f}, "
          f"β={SCALE_PARAMETERS['cosmic']['beta']:.3f}, "
          f"Ω={SCALE_PARAMETERS['cosmic']['Omega']:.3f}, "
          f"k={SCALE_PARAMETERS['cosmic']['k']:.3f}")
    print()
    print("=" * 70)

    # Load black hole data
    bh_data = load_black_hole_data()

    print(f"\nBlack hole observations: {len(bh_data)}")

    # ========================================================================
    # STEP 1: Fit black hole scale parameters
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: FIT BLACK HOLE SCALE PARAMETERS")
    print("=" * 70)
    print()

    bh_params = fit_black_hole_parameters(bh_data)

    print(f"\nOptimized black hole parameters:")
    print(f"  n = {bh_params['n']:.6f}")
    print(f"  β = {bh_params['beta']:.6f}")
    print(f"  Ω = {bh_params['Omega']:.6f}")
    print(f"  k = {bh_params['k']:.6f}")
    print(f"  Error = {bh_params['error']:.6f} ({bh_params['error']*100:.4f}%)")

    # ========================================================================
    # STEP 2: Find true black hole attributes
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: TRUE BLACK HOLE ATTRIBUTES")
    print("=" * 70)
    print()

    true_attrs = find_true_attributes(bh_data, bh_params)

    print("Classical vs Framework-corrected:")
    print("-" * 70)
    print(f"{'Name':<15} {'M_class':<10} {'M_true':<10} {'ΔM%':<8} "
          f"{'f_class':<10} {'f_true':<10} {'Δf%':<8} {'Error%':<8}")
    print("-" * 70)

    for attr in true_attrs:
        print(f"{attr['name']:<15} "
              f"{attr['M_classical']:<10.2f} "
              f"{attr['M_true']:<10.4f} "
              f"{attr['M_correction_%']:<8.2f} "
              f"{attr['f_classical']:<10.2f} "
              f"{attr['f_true']:<10.4f} "
              f"{attr['f_correction_%']:<8.2f} "
              f"{attr['final_error']*100:<8.4f}")

    # ========================================================================
    # STEP 3: Scale comparison
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: PARAMETER SCALING ACROSS SCALES")
    print("=" * 70)
    print()

    print(f"{'Scale':<12} {'n':<12} {'β':<12} {'Ω':<12} {'k':<12} {'Error%':<10}")
    print("-" * 70)
    print(f"{'Micro':<12} "
          f"{SCALE_PARAMETERS['micro']['n']:<12.6f} "
          f"{SCALE_PARAMETERS['micro']['beta']:<12.6f} "
          f"{SCALE_PARAMETERS['micro']['Omega']:<12.6f} "
          f"{SCALE_PARAMETERS['micro']['k']:<12.6f} "
          f"{SCALE_PARAMETERS['micro']['error']*100:<10.4f}")

    print(f"{'Black Hole':<12} "
          f"{bh_params['n']:<12.6f} "
          f"{bh_params['beta']:<12.6f} "
          f"{bh_params['Omega']:<12.6f} "
          f"{bh_params['k']:<12.6f} "
          f"{bh_params['error']*100:<10.4f}")

    print(f"{'Cosmic':<12} "
          f"{SCALE_PARAMETERS['cosmic']['n']:<12.6f} "
          f"{SCALE_PARAMETERS['cosmic']['beta']:<12.6f} "
          f"{SCALE_PARAMETERS['cosmic']['Omega']:<12.6f} "
          f"{SCALE_PARAMETERS['cosmic']['k']:<12.6f} "
          f"{SCALE_PARAMETERS['cosmic']['error']*100:<10.4f}")

    # Calculate trends
    print("\nParameter trends (micro → BH → cosmic):")
    dn_micro_bh = bh_params['n'] - SCALE_PARAMETERS['micro']['n']
    dn_bh_cosmic = SCALE_PARAMETERS['cosmic']['n'] - bh_params['n']

    dOmega_micro_bh = bh_params['Omega'] - SCALE_PARAMETERS['micro']['Omega']
    dOmega_bh_cosmic = SCALE_PARAMETERS['cosmic']['Omega'] - bh_params['Omega']

    dk_micro_bh = bh_params['k'] - SCALE_PARAMETERS['micro']['k']
    dk_bh_cosmic = SCALE_PARAMETERS['cosmic']['k'] - bh_params['k']

    print(f"  Δn:     {dn_micro_bh:+.6f} (micro→BH),  {dn_bh_cosmic:+.6f} (BH→cosmic)")
    print(f"  ΔΩ:     {dOmega_micro_bh:+.6f} (micro→BH),  {dOmega_bh_cosmic:+.6f} (BH→cosmic)")
    print(f"  Δk:     {dk_micro_bh:+.6f} (micro→BH),  {dk_bh_cosmic:+.6f} (BH→cosmic)")

    # ========================================================================
    # Save results
    # ========================================================================
    results = {
        'scale_parameters': SCALE_PARAMETERS,
        'black_hole_parameters': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                                  for k, v in bh_params.items() if k != 'interpolated'},
        'true_black_hole_attributes': [{k: float(v) if isinstance(v, (int, float, np.number)) else v
                                        for k, v in attr.items()}
                                       for attr in true_attrs]
    }

    with open('scale_dependent_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("Results saved to: scale_dependent_results.json")
    print("=" * 70)

if __name__ == '__main__':
    main()
