import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
from tqdm import tqdm
from joblib import Parallel, delayed
import logging
import time
import matplotlib.pyplot as plt
import os
import signal
import sys

# Set up logging
logging.basicConfig(filename='symbolic_cosmo_fit.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

# Generate 10,000 primes
def generate_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(np.sqrt(n)) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(n + 1) if sieve[i]]

PRIMES = generate_primes(104729)[:10000]  # First 10,000 primes, up to ~104,729

phi = (1 + np.sqrt(5)) / 2
fib_cache = {}

def fib_real(n):
    if n in fib_cache:
        return fib_cache[n]
    if n > 100:
        return 0.0
    term1 = phi**n / np.sqrt(5)
    term2 = ((1/phi)**n) * np.cos(np.pi * n)
    result = term1 - term2
    fib_cache[n] = result
    return result

def D(n, beta, r=1.0, k=1.0, Omega=1.0, base=2, scale=1.0):
    Fn_beta = fib_real(n + beta)
    idx = int(np.floor(n + beta)) % len(PRIMES)
    Pn_beta = PRIMES[idx]
    dyadic = base ** (n + beta)
    val = scale * phi * Fn_beta * dyadic * Pn_beta * Omega
    val = np.maximum(val, 1e-30)
    return np.sqrt(val) * (r ** k)

def invert_D(value, r=1.0, k=1.0, Omega=1.0, base=2, scale=1.0, max_n=10000, steps=5000):
    candidates = []
    log_val = np.log10(max(abs(value), 1e-30))
    scale_factors = np.logspace(max(log_val - 5, -20), min(log_val + 5, 20), num=20)
    max_n = min(50000, max(1000, int(1000 * abs(log_val))))
    steps = min(10000, max(1000, int(500 * abs(log_val))))
    n_values = np.logspace(0, np.log10(max_n), steps) if log_val > 3 else np.linspace(0, max_n, steps)
    r_values = [0.1, 0.5, 1.0, 2.0, 4.0]
    k_values = [0.1, 0.5, 1.0, 2.0, 4.0]
    try:
        # Regular D for positive exponents
        for n in n_values:
            for beta in np.linspace(0, 1, 10):
                for dynamic_scale in scale_factors:
                    for r_val in r_values:
                        for k_val in k_values:
                            val = D(n, beta, r_val, k_val, Omega, base, scale * dynamic_scale)
                            if val is not None and np.isfinite(val):
                                diff = abs(val - abs(value))
                                candidates.append((diff, n, beta, dynamic_scale, r_val, k_val))
        # Inverse D for negative exponents (e.g., G)
        for n in n_values:
            for beta in np.linspace(0, 1, 10):
                for dynamic_scale in scale_factors:
                    for r_val in r_values:
                        for k_val in k_values:
                            val = 1 / D(n, beta, r_val, k_val, Omega, base, scale * dynamic_scale)
                            if val is not None and np.isfinite(val):
                                diff = abs(val - abs(value))
                                candidates.append((diff, n, beta, dynamic_scale, r_val, k_val))
        if not candidates:
            logging.error(f"invert_D: No valid candidates for value {value}")
            return None, None, None, None, None, None
        candidates = sorted(candidates, key=lambda x: x[0])[:10]
        valid_vals = [D(n, beta, r, k, Omega, base, scale * s) if x[0] < 1e-10 else 1/D(n, beta, r, k, Omega, base, scale * s)
                      for x, n, beta, s, r, k in candidates]
        valid_vals = [v for v in valid_vals if v is not None and np.isfinite(v)]
        emergent_uncertainty = np.std(valid_vals) if len(valid_vals) > 1 else abs(valid_vals[0]) * 0.01 if valid_vals else 1e-10
        best = candidates[0]
        return best[1], best[2], best[3], emergent_uncertainty, best[4], best[5]
    except Exception as e:
        logging.error(f"invert_D failed for value {value}: {e}")
        return None, None, None, None, None, None

def parse_categorized_codata(filename):
    try:
        df = pd.read_csv(filename, sep='\t', header=0,
                         names=['name', 'value', 'uncertainty', 'unit', 'category'],
                         dtype={'name': str, 'value': float, 'uncertainty': float, 'unit': str, 'category': str},
                         na_values=['exact'])
        df['uncertainty'] = df['uncertainty'].fillna(0.0)
        required_columns = ['name', 'value', 'uncertainty', 'unit']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Missing required columns in {filename}: {missing}")
        logging.info(f"Successfully parsed {len(df)} constants from {filename}")
        return df
    except FileNotFoundError:
        logging.error(f"Input file {filename} not found")
        raise
    except Exception as e:
        logging.error(f"Error parsing {filename}: {e}")
        raise

def generate_emergent_constants(n_max=10000, beta_steps=20, r_values=[0.1, 0.5, 1.0, 2.0, 4.0], k_values=[0.1, 0.5, 1.0, 2.0, 4.0], Omega=1.0, base=2, scale=1.0):
    candidates = []
    n_values = np.linspace(0, n_max, 1000)
    beta_values = np.linspace(0, 1, beta_steps)
    for n in tqdm(n_values, desc="Generating emergent constants"):
        for beta in beta_values:
            for r in r_values:
                for k in k_values:
                    val = D(n, beta, r, k, Omega, base, scale)
                    if val is not None and np.isfinite(val):
                        candidates.append({
                            'n': n, 'beta': beta, 'value': val, 'r': r, 'k': k, 'scale': scale
                        })
                    val_inv = 1 / D(n, beta, r, k, Omega, base, scale)
                    if val_inv is not None and np.isfinite(val_inv):
                        candidates.append({
                            'n': n, 'beta': beta, 'value': val_inv, 'r': r, 'k': k, 'scale': scale
                        })
    return pd.DataFrame(candidates)

def match_to_codata(df_emergent, df_codata, tolerance=0.05, batch_size=100):
    matches = []
    output_file = "emergent_constants.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        pd.DataFrame(columns=['name', 'codata_value', 'emergent_value', 'n', 'beta', 'r', 'k', 'scale', 'error', 'rel_error', 'codata_uncertainty', 'bad_data', 'bad_data_reason']).to_csv(f, sep="\t", index=False)

    for start in range(0, len(df_codata), batch_size):
        batch = df_codata.iloc[start:start + batch_size]
        for _, codata_row in tqdm(batch.iterrows(), total=len(batch), desc=f"Matching constants batch {start//batch_size + 1}"):
            value = codata_row['value']
            mask = abs(df_emergent['value'] - value) / max(abs(value), 1e-30) < tolerance
            matched = df_emergent[mask]
            for _, emergent_row in matched.iterrows():
                error = abs(emergent_row['value'] - value)
                rel_error = error / max(abs(value), 1e-30)
                matches.append({
                    'name': codata_row['name'],
                    'codata_value': value,
                    'emergent_value': emergent_row['value'],
                    'n': emergent_row['n'],
                    'beta': emergent_row['beta'],
                    'r': emergent_row['r'],
                    'k': emergent_row['k'],
                    'scale': emergent_row['scale'],
                    'error': error,
                    'rel_error': rel_error,
                    'codata_uncertainty': codata_row['uncertainty'],
                    'bad_data': rel_error > 0.5 or (codata_row['uncertainty'] is not None and abs(codata_row['uncertainty'] - error) > 10 * codata_row['uncertainty']),
                    'bad_data_reason': f"High rel_error ({rel_error:.2e})" if rel_error > 0.5 else f"Uncertainty deviation ({codata_row['uncertainty']:.2e} vs. {error:.2e})" if (codata_row['uncertainty'] is not None and abs(codata_row['uncertainty'] - error) > 10 * codata_row['uncertainty']) else ""
                })
        try:
            with open(output_file, 'a', encoding='utf-8') as f:
                pd.DataFrame(matches).to_csv(f, sep="\t", index=False, header=False, lineterminator='\n')
                f.flush()
            matches = []
        except Exception as e:
            logging.error(f"Failed to save batch {start//batch_size + 1} to {output_file}: {e}")
    return pd.DataFrame(pd.read_csv(output_file, sep='\t'))

def check_physical_consistency(df_results):
    bad_data = []
    relations = [
        ('Planck constant', 'reduced Planck constant', lambda x, y: abs(x['scale'] / y['scale'] - 2 * np.pi), 0.1, 'scale ratio vs. 2π'),
        ('proton mass', 'proton-electron mass ratio', lambda x, y: abs(x['n'] - y['n'] - np.log10(1836)), 0.5, 'n difference vs. log(proton-electron ratio)'),
        ('Fermi coupling constant', 'weak mixing angle', lambda x, y: abs(x['scale'] - y['scale'] / np.sqrt(2)), 0.1, 'scale vs. sin²θ_W/√2'),
        ('tau energy equivalent', 'tau mass energy equivalent in MeV', lambda x, y: abs(x['codata_value'] - y['codata_value']), 0.01, 'value consistency'),
        ('proton mass', 'electron mass', 'proton-electron mass ratio',
         lambda x, y, z: abs(z['n'] - abs(x['n'] - y['n'])), 10.0, 'n inconsistency for mass ratio'),
        ('fine-structure constant', 'elementary charge', 'Planck constant',
         lambda x, y, z: abs(x['codata_value'] - y['codata_value']**2 / (4 * np.pi * 8.854187817e-12 * z['codata_value'] * 299792458)), 0.01, 'fine-structure vs. e²/(4πε₀hc)'),
        ('Bohr magneton', 'elementary charge', 'Planck constant',
         lambda x, y, z: abs(x['codata_value'] - y['codata_value'] * z['codata_value'] / (2 * 9.1093837e-31)), 0.01, 'Bohr magneton vs. eh/(2m_e)'),
        ('speed of light in vacuum', None, lambda x: abs(x['codata_value'] - 299792458), 0.01, 'speed of light deviation'),
        ('Newtonian constant of gravitation', None, lambda x: abs(x['codata_value'] - 6.6743e-11), 1e-12, 'G deviation')
    ]
    for relation in relations:
        try:
            if len(relation) == 5:
                name1, name2, check_func, threshold, reason = relation
                if name1 in df_results['name'].values and name2 in df_results['name'].values:
                    row1 = df_results[df_results['name'] == name1].iloc[0]
                    row2 = df_results[df_results['name'] == name2].iloc[0]
                    if check_func(row1, row2) > threshold:
                        bad_data.append((name1, f"Physical inconsistency: {reason}"))
                        bad_data.append((name2, f"Physical inconsistency: {reason}"))
            elif len(relation) == 6:
                name1, name2, name3, check_func, threshold, reason = relation
                if all(name in df_results['name'].values for name in [name1, name2, name3]):
                    row1 = df_results[df_results['name'] == name1].iloc[0]
                    row2 = df_results[df_results['name'] == name2].iloc[0]
                    row3 = df_results[df_results['name'] == name3].iloc[0]
                    if check_func(row1, row2, row3) > threshold:
                        bad_data.append((name3, f"Physical inconsistency: {reason}"))
            elif len(relation) == 4:
                name, _, check_func, threshold, reason = relation
                if name in df_results['name'].values:
                    row = df_results[df_results['name'] == name].iloc[0]
                    if check_func(row) > threshold:
                        bad_data.append((name, f"Physical inconsistency: {reason}"))
        except Exception as e:
            logging.warning(f"Physical consistency check failed for {relation}: {e}")
            continue
    return bad_data

def total_error(params, df_subset):
    r, k, Omega, base, scale = params
    df_results = symbolic_fit_all_constants(df_subset, base=base, Omega=Omega, r=r, k=k, scale=scale)
    if df_results.empty:
        return np.inf
    valid_errors = df_results['rel_error'].dropna()
    return valid_errors.mean() if not valid_errors.empty else np.inf

def process_constant(row, r, k, Omega, base, scale):
    try:
        name, value, uncertainty, unit = row['name'], row['value'], row['uncertainty'], row['unit']
        abs_value = abs(value)
        sign = np.sign(value)
        result = invert_D(abs_value, r=r, k=k, Omega=Omega, base=base, scale=scale)
        if result[0] is None:
            logging.warning(f"No valid fit for {name}")
            return {
                'name': name, 'codata_value': value, 'unit': unit, 'n': None, 'beta': None, 'emergent_value': None,
                'error': None, 'rel_error': None, 'codata_uncertainty': uncertainty, 'emergent_uncertainty': None,
                'scale': None, 'bad_data': True, 'bad_data_reason': 'No valid fit found', 'r': None, 'k': None
            }
        n, beta, dynamic_scale, emergent_uncertainty, r_local, k_local = result
        approx = D(n, beta, r_local, k_local, Omega, base, scale * dynamic_scale) if value > 0 else 1 / D(n, beta, r_local, k_local, Omega, base, scale * dynamic_scale)
        if approx is None:
            logging.warning(f"D returned None for {name}")
            return {
                'name': name, 'codata_value': value, 'unit': unit, 'n': None, 'beta': None, 'emergent_value': None,
                'error': None, 'rel_error': None, 'codata_uncertainty': uncertainty, 'emergent_uncertainty': None,
                'scale': None, 'bad_data': True, 'bad_data_reason': 'D function returned None', 'r': None, 'k': None
            }
        approx *= sign
        error = abs(approx - value)
        rel_error = error / max(abs(value), 1e-30) if abs(value) > 0 else np.inf
        bad_data = False
        bad_data_reason = ""
        if rel_error > 0.5:
            bad_data = True
            bad_data_reason += f"High relative error ({rel_error:.2e} > 0.5); "
        if emergent_uncertainty is not None and uncertainty is not None:
            if emergent_uncertainty > uncertainty * 20 or emergent_uncertainty < uncertainty / 20:
                bad_data = True
                bad_data_reason += f"Uncertainty deviates from emergent ({emergent_uncertainty:.2e} vs. {uncertainty:.2e}); "
        return {
            'name': name, 'codata_value': value, 'unit': unit, 'n': n, 'beta': beta, 'emergent_value': approx,
            'error': error, 'rel_error': rel_error, 'codata_uncertainty': uncertainty,
            'emergent_uncertainty': emergent_uncertainty, 'scale': scale * dynamic_scale,
            'bad_data': bad_data, 'bad_data_reason': bad_data_reason, 'r': r_local, 'k': k_local
        }
    except Exception as e:
        logging.error(f"process_constant failed for {row['name']}: {e}")
        return {
            'name': row['name'], 'codata_value': row['value'], 'unit': row['unit'], 'n': None, 'beta': None,
            'emergent_value': None, 'error': None, 'rel_error': None, 'codata_uncertainty': row['uncertainty'],
            'emergent_uncertainty': None, 'scale': None, 'bad_data': True, 'bad_data_reason': f"Processing error: {str(e)}",
            'r': None, 'k': None
        }

def symbolic_fit_all_constants(df, base=2, Omega=1.0, r=1.0, k=1.0, scale=1.0, batch_size=100):
    logging.info("Starting symbolic fit for all constants...")
    results = []
    output_file = "symbolic_fit_results_emergent_fixed.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        pd.DataFrame(columns=['name', 'codata_value', 'unit', 'n', 'beta', 'emergent_value', 'error', 'rel_error',
                              'codata_uncertainty', 'emergent_uncertainty', 'scale', 'bad_data', 'bad_data_reason', 'r', 'k']).to_csv(f, sep="\t", index=False)

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        try:
            batch_results = Parallel(n_jobs=-1, timeout=120, backend='loky', maxtasksperchild=20)(
                delayed(process_constant)(row, r, k, Omega, base, scale)
                for row in tqdm(batch.to_dict('records'), total=len(batch), desc=f"Fitting constants batch {start//batch_size + 1}")
            )
            batch_results = [r for r in batch_results if r is not None]
            results.extend(batch_results)
            try:
                with open(output_file, 'a', encoding='utf-8') as f:
                    pd.DataFrame(batch_results).to_csv(f, sep="\t", index=False, header=False, lineterminator='\n')
                    f.flush()
            except Exception as e:
                logging.error(f"Failed to save batch {start//batch_size + 1} to {output_file}: {e}")
        except Exception as e:
            logging.error(f"Parallel processing failed for batch {start//batch_size + 1}: {e}")
            continue
        fib_cache.clear()  # Clear cache to manage memory
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results['bad_data'] = df_results.get('bad_data', False)
        df_results['bad_data_reason'] = df_results.get('bad_data_reason', '')
        for name in df_results['name'].unique():
            mask = df_results['name'] == name
            if df_results.loc[mask, 'codata_uncertainty'].notnull().any():
                uncertainties = df_results.loc[mask, 'codata_uncertainty'].dropna()
                if not uncertainties.empty:
                    Q1, Q3 = np.percentile(uncertainties, [25, 75])
                    IQR = Q3 - Q1
                    outlier_mask = (uncertainties < Q1 - 1.5 * IQR) | (uncertainties > Q3 + 1.5 * IQR)
                    if outlier_mask.any():
                        df_results.loc[mask & df_results['codata_uncertainty'].isin(uncertainties[outlier_mask]), 'bad_data'] = True
                        df_results.loc[mask & df_results['codata_uncertainty'].isin(uncertainties[outlier_mask]), 'bad_data_reason'] += 'Uncertainty outlier; '
        high_rel_error_mask = df_results['rel_error'] > 0.5
        df_results.loc[high_rel_error_mask, 'bad_data'] = True
        df_results.loc[high_rel_error_mask, 'bad_data_reason'] += df_results.loc[high_rel_error_mask, 'rel_error'].apply(lambda x: f"High relative error ({x:.2e} > 0.5); ")
        high_uncertainty_mask = (df_results['emergent_uncertainty'].notnull()) & (
            (df_results['codata_uncertainty'] > 20 * df_results['emergent_uncertainty']) |
            (df_results['codata_uncertainty'] < 0.05 * df_results['emergent_uncertainty'])
        )
        df_results.loc[high_uncertainty_mask, 'bad_data'] = True
        df_results.loc[high_uncertainty_mask, 'bad_data_reason'] += df_results.loc[high_uncertainty_mask].apply(
            lambda row: f"Uncertainty deviates from emergent ({row['codata_uncertainty']:.2e} vs. {row['emergent_uncertainty']:.2e}); ", axis=1)
        bad_data = check_physical_consistency(df_results)
        for name, reason in bad_data:
            df_results.loc[df_results['name'] == name, 'bad_data'] = True
            df_results.loc[df_results['name'] == name, 'bad_data_reason'] += reason + '; '
    logging.info("Symbolic fit completed.")
    return df_results

def select_worst_names(df, n_select=20):
    categories = df['category'].unique()
    n_per_category = max(1, n_select // len(categories))
    selected = []
    for category in categories:
        cat_df = df[df['category'] == category]
        if len(cat_df) > 0:
            n_to_select = min(n_per_category, len(cat_df))
            selected.extend(np.random.choice(cat_df['name'], size=n_to_select, replace=False))
    if len(selected) < n_select:
        remaining = df[~df['name'].isin(selected)]
        if len(remaining) > 0:
            selected.extend(np.random.choice(remaining['name'], size=n_select - len(selected), replace=False))
    return selected[:n_select]

def a_of_z(z):
    return 1 / (1 + z)

def Omega(z, Omega0, alpha):
    return Omega0 / (a_of_z(z) ** alpha)

def s(z, s0, beta):
    return s0 * (1 + z) ** (-beta)

def G(z, k, r0, Omega0, s0, alpha, beta):
    return Omega(z, Omega0, alpha) * k**2 * r0 / s(z, s0, beta)

def H(z, k, r0, Omega0, s0, alpha, beta):
    Om_m = 0.3
    Om_de = 0.7
    Gz = G(z, k, r0, Omega0, s0, alpha, beta)
    Hz_sq = (H0 ** 2) * (Om_m * Gz * (1 + z) ** 3 + Om_de)
    return np.sqrt(Hz_sq)

def emergent_c(z, Omega0, alpha, gamma):
    return c0_emergent * (Omega(z, Omega0, alpha) / Omega0) ** gamma * lambda_scale

def compute_luminosity_distance_grid(z_max, params, n=500):
    k, r0, Omega0, s0, alpha, beta, gamma = params
    z_grid = np.linspace(0, z_max, n)
    c_z = emergent_c(z_grid, Omega0, alpha, gamma)
    H_z = H(z_grid, k, r0, Omega0, s0, alpha, beta)
    integrand_values = c_z / H_z
    integral_grid = np.cumsum((integrand_values[:-1] + integrand_values[1:]) / 2 * np.diff(z_grid))
    integral_grid = np.insert(integral_grid, 0, 0)
    d_c = interp1d(z_grid, integral_grid, kind='cubic', fill_value="extrapolate")
    return lambda z: (1 + z) * d_c(z)

def model_mu(z_arr, params):
    d_L_func = compute_luminosity_distance_grid(np.max(z_arr), params)
    d_L_vals = d_L_func(z_arr)
    return 5 * np.log10(d_L_vals) + 25

def signal_handler(sig, frame):
    print("\nKeyboardInterrupt detected. Saving partial results...")
    logging.info("KeyboardInterrupt detected. Exiting gracefully.")
    for output_file in ["emergent_constants.txt", "symbolic_fit_results_emergent_fixed.txt", "cosmo_fit_results.txt"]:
        try:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.flush()
        except Exception as e:
            logging.error(f"Failed to flush {output_file} on interrupt: {e}")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    start_time = time.time()
    stages = ['Parsing data', 'Generating emergent constants', 'Optimizing CODATA parameters', 'Fitting CODATA constants', 'Fitting supernova data', 'Generating plots']
    progress = tqdm(stages, desc="Overall progress")

    # Stage 1: Parse CODATA
    input_file = "categorized_allascii.txt"
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found in the current directory")
    df = parse_categorized_codata(input_file)
    logging.info(f"Parsed {len(df)} constants")
    progress.update(1)

    # Stage 2: Generate emergent constants
    emergent_df = generate_emergent_constants(n_max=10000, beta_steps=20)
    matched_df = match_to_codata(emergent_df, df, tolerance=0.05, batch_size=100)
    logging.info("Saved emergent constants to emergent_constants.txt")
    progress.update(1)

    # Stage 3: Optimize CODATA parameters
    worst_names = select_worst_names(df, n_select=20)
    print(f"Selected constants for optimization: {worst_names}")
    subset_df = df[df['name'].isin(worst_names)]
    if subset_df.empty:
        subset_df = df.head(50)
    init_params = [0.5, 0.5, 0.5, 2.0, 0.1]
    bounds = [(1e-10, 100), (1e-10, 100), (1e-10, 100), (1.5, 20), (1e-10, 1000)]
    try:
        res = differential_evolution(total_error, bounds, args=(subset_df,), maxiter=100, popsize=15)
        if res.success:
            res = minimize(total_error, res.x, args=(subset_df,), bounds=bounds, method='SLSQP', options={'maxiter': 500})
        if not res.success:
            logging.warning(f"Optimization failed: {res.message}")
            r_opt, k_opt, Omega_opt, base_opt, scale_opt = init_params
        else:
            r_opt, k_opt, Omega_opt, base_opt, scale_opt = res.x
        print(f"CODATA Optimization complete. Found parameters:\nr = {r_opt:.6f}, k = {k_opt:.6f}, Omega = {Omega_opt:.6f}, base = {base_opt:.6f}, scale = {scale_opt:.6f}")
    except Exception as e:
        logging.error(f"CODATA Optimization failed: {e}")
        r_opt, k_opt, Omega_opt, base_opt, scale_opt = init_params
        print(f"CODATA Optimization failed: {e}. Using default parameters.")
    progress.update(1)

    # Stage 4: Fit CODATA constants
    df_results = symbolic_fit_all_constants(df, base=base_opt, Omega=Omega_opt, r=r_opt, k=k_opt, scale=scale_opt, batch_size=100)
    if not df_results.empty:
        with open("symbolic_fit_results.txt", 'w', encoding='utf-8') as f:
            df_results.to_csv(f, sep="\t", index=False)
            f.flush()
        logging.info(f"Saved CODATA results to symbolic_fit_results.txt")
    else:
        logging.error("No CODATA results to save")
    progress.update(1)

    # Stage 5: Fit supernova data
    supernova_file = 'hlsp_ps1cosmo_panstarrs_gpc1_all_model_v1_lcparam-full.txt'
    if not os.path.exists(supernova_file):
        raise FileNotFoundError(f"{supernova_file} not found")
    lc_data = np.genfromtxt(supernova_file, delimiter=' ', names=True, comments='#', dtype=None, encoding=None)
    z = lc_data['zcmb']
    mb = lc_data['mb']
    dmb = lc_data['dmb']

    # Reconstruct cosmological parameters
    fitted_params = {
        'k': 1.049342, 'r0': 1.049676, 'Omega0': 1.049675, 's0': 0.994533,
        'alpha': 0.340052, 'beta': 0.360942, 'gamma': 0.993975, 'H0': 70.0,
        'c0': phi ** (2.5 * 6), 'M': -19.3
    }
    params_reconstructed = {}
    for name, val in fitted_params.items():
        if name == 'M':
            params_reconstructed[name] = val
            continue
        n, beta, _, _, r, k = invert_D(val)
        params_reconstructed[name] = D(n, beta, r, k) if name != 'c0' else phi ** (2.5 * n)

    global H0, c0_emergent, lambda_scale, lambda_G
    H0 = params_reconstructed['H0']
    c0_emergent = params_reconstructed['c0']
    lambda_scale = 299792.458 / c0_emergent
    lambda_G = 6.6743e-11 / G(0, params_reconstructed['k'], params_reconstructed['r0'],
                               params_reconstructed['Omega0'], params_reconstructed['s0'],
                               params_reconstructed['alpha'], params_reconstructed['beta'])

    param_list = [
        params_reconstructed['k'], params_reconstructed['r0'], params_reconstructed['Omega0'],
        params_reconstructed['s0'], params_reconstructed['alpha'], params_reconstructed['beta'],
        params_reconstructed['gamma']
    ]
    mu_fit = model_mu(z, param_list)
    residuals = (mb - params_reconstructed['M']) - mu_fit

    cosmo_results = pd.DataFrame({
        'z': z, 'mu_obs': mb - params_reconstructed['M'], 'mu_fit': mu_fit,
        'residuals': residuals, 'dmb': dmb
    })
    with open("cosmo_fit_results.txt", 'w', encoding='utf-8') as f:
        cosmo_results.to_csv(f, sep="\t", index=False)
        f.flush()
    logging.info("Saved supernova fit results to cosmo_fit_results.txt")
    progress.update(1)

    # Stage 6: Generate plots
    df_results_sorted = df_results.sort_values("rel_error", na_position='last')
    print("\nTop 20 best CODATA fits:")
    print(df_results_sorted.head(20)[['name', 'codata_value', 'unit', 'n', 'beta', 'emergent_value', 'error', 'rel_error', 'codata_uncertainty', 'scale', 'bad_data', 'bad_data_reason']].to_string(index=False))
    print("\nTop 20 worst CODATA fits:")
    print(df_results_sorted.tail(20)[['name', 'codata_value', 'unit', 'n', 'beta', 'emergent_value', 'error', 'rel_error', 'codata_uncertainty', 'scale', 'bad_data', 'bad_data_reason']].to_string(index=False))
    print("\nPotentially bad data constants:")
    bad_data_df = df_results[df_results['bad_data'] == True][['name', 'codata_value', 'error', 'rel_error', 'codata_uncertainty', 'emergent_uncertainty', 'bad_data_reason']]
    print(bad_data_df.to_string(index=False))
    print("\nTop 20 emergent constants matches:")
    matched_df_sorted = matched_df.sort_values('rel_error', na_position='last')
    print(matched_df_sorted.head(20)[['name', 'codata_value', 'emergent_value', 'n', 'beta', 'error', 'rel_error', 'codata_uncertainty', 'bad_data', 'bad_data_reason']].to_string(index=False))

    plt.figure(figsize=(10, 5))
    plt.hist(df_results_sorted['rel_error'].dropna(), bins=50, color='skyblue', edgecolor='black')
    plt.title('Histogram of Relative Errors in CODATA Fit')
    plt.xlabel('Relative Error')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('histogram_rel_errors.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.scatter(df_results_sorted['n'], df_results_sorted['rel_error'], alpha=0.5, s=15, c='orange', edgecolors='black')
    plt.title('Relative Error vs Symbolic Dimension n (CODATA)')
    plt.xlabel('n')
    plt.ylabel('Relative Error')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('scatter_n_rel_error.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(matched_df_sorted.head(20)['name'], matched_df_sorted.head(20)['rel_error'], color='purple', edgecolor='black')
    plt.xticks(rotation=90)
    plt.title('Relative Errors for Top 20 Emergent Constants')
    plt.xlabel('Constant Name')
    plt.ylabel('Relative Error')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('bar_emergent_errors.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.errorbar(z, mb - params_reconstructed['M'], yerr=dmb, fmt='.', alpha=0.5, label='Pan-STARRS1 SNe')
    plt.plot(z, mu_fit, 'r-', label='Symbolic Emergent Gravity Model')
    plt.xlabel('Redshift (z)')
    plt.ylabel('Distance Modulus (μ)')
    plt.title('Supernova Distance Modulus with Emergent G(z) and c(z)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('supernova_fit.png')
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.errorbar(z, residuals, yerr=dmb, fmt='.', alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Redshift (z)')
    plt.ylabel('Residuals (μ_data - μ_model)')
    plt.title('Residuals of Symbolic Model with Emergent G(z) and c(z)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('supernova_residuals.png')
    plt.close()

    z_grid = np.linspace(0, max(z), 300)
    c_z = emergent_c(z_grid, params_reconstructed['Omega0'], params_reconstructed['alpha'], params_reconstructed['gamma'])
    G_z = G(z_grid, params_reconstructed['k'], params_reconstructed['r0'], params_reconstructed['Omega0'],
            params_reconstructed['s0'], params_reconstructed['alpha'], params_reconstructed['beta']) * lambda_G
    G_z_norm = G_z / G_z[0]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(z_grid, c_z, label='c(z) (km/s)')
    plt.axhline(299792.458, color='red', linestyle='--', label='Local c')
    plt.xlabel('Redshift z')
    plt.ylabel('Speed of Light c(z) [km/s]')
    plt.title('Emergent Speed of Light Variation')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(z_grid, G_z_norm, label='G(z) / G_0')
    plt.axhline(1.0, color='red', linestyle='--', label='Local G')
    plt.xlabel('Redshift z')
    plt.ylabel('Normalized Gravitational Constant G(z)/G_0')
    plt.title('Emergent Gravitational Constant Variation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('emergent_c_G.png')
    plt.close()

    logging.info(f"Total runtime: {time.time() - start_time:.2f} seconds")
    progress.update(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        signal_handler(None, None)