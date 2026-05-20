"""
DGP Generator for JMR Horse Race
5 DGP Worlds x K=3 States x 3-Segment Heterogeneity
Worlds: independent, correlated, structural, mixed, hidden_structure
"""

import numpy as np
import argparse
import json
import sys
from pathlib import Path
from scipy.stats import poisson, norm
from scipy.linalg import cholesky


def parse_segment_config(config_path=None, n_segments=3):
    """Parse segment configuration from JSON or use defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        return cfg
    
    return {
        "n_segments": n_segments,
        "probs": [0.50, 0.35, 0.15],
        "labels": ["Low", "Medium", "Whale"],
        "mu_theta": [-1.2, 0.3, 2.1],
        "sigma_theta": [0.8, 1.0, 1.4],
        "sigma_spend": [0.6, 0.8, 1.2],
        "spend_intercept": [0.8, 1.5, 2.5]
    }


def build_transition_matrix(K, persistence):
    """Build KxK transition matrix from self-persistence vector."""
    if isinstance(persistence, (int, float)):
        persistence = [persistence] * K
    
    P = np.zeros((K, K))
    for i in range(K):
        p_self = persistence[i]
        n_other = K - 1
        p_other = (1 - p_self) / n_other if n_other > 0 else 0
        for j in range(K):
            P[i, j] = p_self if i == j else p_other
    return P


def stationary_distribution(P):
    """Compute stationary distribution of Markov chain P."""
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    stationary = np.real(eigvecs[:, idx])
    stationary = stationary / stationary.sum()
    return np.abs(stationary)


def calibrate_sparsity_full(world, target_sparsity, N_cal, T_cal, K, P, cfg,
                            gamma_h, gamma_m, rho, state_timing_coef, 
                            state_spend_coef, cov_beta, max_iter=30, tol=0.005):
    """
    Iterative calibration using FULL emission logic of target world.
    Adjusts base timing intercept until observed purchase rate converges.
    """
    low, high = -4.0, 4.0
    best_intercept = 0.0
    best_error = 1.0
    
    for iteration in range(max_iter):
        mid = (low + high) / 2.0
        
        data_test = generate_world_core(
            world=world, N=N_cal, T=T_cal, seed=999 + iteration,
            K=K, P=P, cfg=cfg, base_intercept=mid,
            gamma_h=gamma_h, gamma_m=gamma_m, rho=rho,
            state_timing_coef=state_timing_coef,
            state_spend_coef=state_spend_coef,
            cov_beta=cov_beta, include_covariates=(cov_beta is not None),
            verbose=False
        )
        
        actual = (data_test['y'] > 0).mean()
        error = abs(actual - target_sparsity)
        
        if error < best_error:
            best_error = error
            best_intercept = mid
        
        if error < tol:
            return mid
        
        if actual < target_sparsity:
            low = mid
        else:
            high = mid
    
    return best_intercept


def generate_world_core(world, N, T, seed, K, P, cfg, base_intercept,
                        gamma_h, gamma_m, rho, state_timing_coef,
                        state_spend_coef, cov_beta, include_covariates,
                        verbose=False):
    """
    Core DGP generation. Returns dict with standardized keys.
    This is the FULL emission logic used by both calibration and final generation.
    """
    np.random.seed(seed)
    n_seg = cfg["n_segments"]
    
    segments = np.random.choice(n_seg, size=N, p=cfg["probs"])
    
    theta = np.zeros(N)
    for c in range(n_seg):
        mask = (segments == c)
        theta[mask] = np.random.normal(cfg["mu_theta"][c], 
                                        cfg["sigma_theta"][c], 
                                        size=mask.sum())
    
    S = np.zeros((N, T), dtype=int)
    init_dist = stationary_distribution(P)
    S[:, 0] = np.random.choice(K, size=N, p=init_dist)
    
    for t in range(1, T):
        for i in range(N):
            S[i, t] = np.random.choice(K, p=P[S[i, t-1]])
    
    y = np.zeros((N, T), dtype=int)
    z = np.zeros((N, T))
    
    covariates = None
    if include_covariates:
        covariates = generate_covariates_skeleton(N, T)
    
    for i in range(N):
        c = segments[i]
        th = theta[i]
        sigma_s = cfg["sigma_spend"][c]
        spend_base = cfg["spend_intercept"][c]
        
        for t in range(T):
            s = S[i, t]
            
            log_lam = base_intercept + state_timing_coef * s
            log_mu = spend_base + state_spend_coef * s
            
            if world in ["structural", "mixed", "hidden_structure"]:
                log_lam += gamma_h * th
            
            if world in ["structural", "mixed"]:
                log_mu += gamma_m * th
            
            if include_covariates and cov_beta is not None:
                cov_t = covariates[i, t, :]
                log_lam += cov_beta[0] * cov_t[0] + cov_beta[1] * cov_t[1]
                log_mu += cov_beta[2] * cov_t[2]
            
            lam = np.exp(log_lam)
            y[i, t] = poisson.rvs(lam)
            
            if y[i, t] > 0:
                mu = np.exp(log_mu)
                z[i, t] = np.random.lognormal(np.log(mu), sigma_s)
    
    if world in ["correlated", "mixed"]:
        z = apply_copula_residual(world, y, z, S, theta, segments, cfg, 
                                     base_intercept, state_timing_coef,
                                     state_spend_coef, gamma_h, gamma_m,
                                     cov_beta, rho)
    
    if include_covariates:
        covariates = finalize_covariates(y, z, covariates)
    
    actual_sparsity = (y > 0).mean()
    
    data = {
        'y': y.astype(np.float32),
        'z': z.astype(np.float32),
        'S_true': S.astype(np.int32),
        'theta': theta.astype(np.float32),
        'segments': segments.astype(np.int32),
        'covariates': covariates.astype(np.float32) if covariates is not None else None,
        'N': N,
        'T': T,
        'K': K,
        'world': world,
        'seed': seed,
        'base_intercept': float(base_intercept),
        'actual_sparsity': float(actual_sparsity)
    }
    
    return data


def generate_covariates_skeleton(N, T):
    """Generate placeholder covariates (filled properly after y,z known)."""
    return np.zeros((N, T, 3))


def finalize_covariates(y, z, cov_raw):
    """Compute proper lagged RFM covariates after data generation."""
    N, T = y.shape
    cov = np.zeros((N, T, 3))
    
    for i in range(N):
        last_purchase = -999
        freq = 0
        cum_monetary = 0.0
        
        for t in range(T):
            cov[i, t, 0] = t - last_purchase if last_purchase >= 0 else 999
            cov[i, t, 1] = freq
            cov[i, t, 2] = cum_monetary / (freq + 1e-6) if freq > 0 else 0
            
            if y[i, t] > 0:
                last_purchase = t
                freq += 1
                cum_monetary += z[i, t]
    
    for j in range(3):
        m, s = cov[:, :, j].mean(), cov[:, :, j].std() + 1e-6
        cov[:, :, j] = (cov[:, :, j] - m) / s
    
    return cov


def apply_copula_residual(world, y, z, S, theta, segments, cfg, base_intercept,
                          state_timing_coef, state_spend_coef, gamma_h, gamma_m,
                          cov_beta, rho):
    """
    Apply Gaussian copula residual correlation to spend.
    Generates correlated latent normals first, then transforms to spend.
    Only modifies z where y > 0.
    """
    N, T = y.shape
    
    Sigma = np.array([[1.0, rho], [rho, 1.0]])
    L = cholesky(Sigma, lower=True)
    
    for i in range(N):
        c = segments[i]
        th = theta[i]
        sigma_s = cfg["sigma_spend"][c]
        spend_base = cfg["spend_intercept"][c]
        
        for t in range(T):
            if y[i, t] == 0:
                continue
            
            s = S[i, t]
            
            log_lam = base_intercept + state_timing_coef * s
            if world_includes_theta("timing", gamma_h=gamma_h):
                log_lam += gamma_h * th
            
            lam = np.exp(log_lam)
            u_timing = poisson.cdf(y[i, t], lam)
            u_timing = np.clip(u_timing, 0.001, 0.999)
            z1 = norm.ppf(u_timing)
            
            eps = np.random.normal()
            w_spend = L[1, 0] * z1 + L[1, 1] * eps
            u_spend = norm.cdf(w_spend)
            u_spend = np.clip(u_spend, 0.001, 0.999)
            
            log_mu = spend_base + state_spend_coef * s
            if world_includes_theta("spend", gamma_m=gamma_m):
                log_mu += gamma_m * th
            
            mu = np.exp(log_mu)
            z[i, t] = np.exp(norm.ppf(u_spend) * sigma_s + np.log(mu))
    
    return z


def world_includes_theta(margin, gamma_h=0, gamma_m=0):
    """Helper to check if theta enters a given margin."""
    if margin == "timing":
        return gamma_h != 0
    elif margin == "spend":
        return gamma_m != 0
    return False


def generate_world(world, N, T, seed, K, target_sparsity, state_persistence,
                   gamma_h, gamma_m, rho, state_timing_coef, state_spend_coef,
                   segment_config, include_covariates=True, verbose=True,
                   cov_beta=None):
    """
    Full DGP generation with calibration.
    """
    np.random.seed(seed)
    cfg = parse_segment_config(segment_config)
    n_seg = cfg["n_segments"]
    
    P = build_transition_matrix(K, state_persistence)
    
    if include_covariates and cov_beta is None:
        cov_beta = np.array([0.15, 0.10, 0.05])
    
    N_cal = min(200, N)
    T_cal = min(52, T)
    
    calibrated_intercept = calibrate_sparsity_full(
        world, target_sparsity, N_cal, T_cal, K, P, cfg,
        gamma_h, gamma_m, rho, state_timing_coef, state_spend_coef, cov_beta
    )
    
    data = generate_world_core(
        world=world, N=N, T=T, seed=seed, K=K, P=P, cfg=cfg,
        base_intercept=calibrated_intercept,
        gamma_h=gamma_h, gamma_m=gamma_m, rho=rho,
        state_timing_coef=state_timing_coef,
        state_spend_coef=state_spend_coef,
        cov_beta=cov_beta, include_covariates=include_covariates,
        verbose=verbose
    )
    
    data['calibrated_intercept'] = float(calibrated_intercept)
    data['target_sparsity'] = float(target_sparsity)
    
    true_params = {
        'world': world,
        'N': N, 'T': T, 'K': K, 'seed': seed,
        'target_sparsity': target_sparsity,
        'state_persistence': state_persistence,
        'gamma_h': gamma_h, 'gamma_m': gamma_m,
        'rho': rho,
        'state_timing_coef': state_timing_coef,
        'state_spend_coef': state_spend_coef,
        'segment_config': cfg,
        'transition_matrix': P.tolist(),
        'cov_beta': cov_beta.tolist() if cov_beta is not None else None
    }
    data['true_params'] = json.dumps(true_params)
    
    if verbose:
        print("\n" + "="*50)
        print("DGP Generated: {}".format(world))
        print("  N={}, T={}, K={}, seed={}".format(N, T, K, seed))
        print("  Target sparsity: {:.1%}".format(target_sparsity))
        print("  Actual sparsity: {:.1%}".format(data['actual_sparsity']))
        print("  Calibrated intercept: {:.3f}".format(calibrated_intercept))
        print("  Segment counts: {}".format(np.bincount(data['segments'], minlength=n_seg)))
        print("  State distribution: {}".format(np.bincount(data['S_true'].flatten(), minlength=K)))
        print("  Theta range: [{:.2f}, {:.2f}]".format(data['theta'].min(), data['theta'].max()))
        z_pos = data['z'][data['z'] > 0]
        print("  Spend (y>0): mean={:.2f}, std={:.2f}".format(z_pos.mean(), z_pos.std()))
        print("  Covariates: {}".format("Yes" if include_covariates else "No"))
        print("="*50)
    
    return data



def save_csv(data, output_path):
    import pandas as pd
    N, T = data['N'], data['T']
    rows = []
    for i in range(N):
        for t in range(T):
            row = {
                'customer_id': i,
                't': t,
                'y': int(data['y'][i, t]),
                'z': float(data['z'][i, t]),
                'true_state': int(data['S_true'][i, t]),
                'segment': int(data['segments'][i]),
                'theta': float(data['theta'][i]),
            }
            if data['covariates'] is not None:
                row['recency_lag'] = float(data['covariates'][i, t, 0])
                row['frequency_lag'] = float(data['covariates'][i, t, 1])
                row['monetary_lag'] = float(data['covariates'][i, t, 2])
            else:
                row['recency_lag'] = 0.0
                row['frequency_lag'] = 0.0
                row['monetary_lag'] = 0.0
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description='JMR DGP Generator')
    parser.add_argument('--world', type=str, required=True,
                        choices=['independent', 'correlated', 'structural', 'mixed', 'hidden_structure'],
                        help='DGP world')
    parser.add_argument('--N', type=int, default=500, help='Number of customers')
    parser.add_argument('--T', type=int, default=104, help='Number of time periods')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--K', type=int, default=3, help='Number of latent states')
    parser.add_argument('--sparsity', type=float, default=0.30, help='Target purchase rate')
    parser.add_argument('--state_persistence', type=str, default='0.85,0.70,0.90',
                        help='Self-persistence for each state (comma-separated)')
    parser.add_argument('--gamma_h', type=float, default=0.8, help='Timing loading on theta')
    parser.add_argument('--gamma_m', type=float, default=0.7, help='Spend loading on theta')
    parser.add_argument('--rho', type=float, default=0.4, help='Copula correlation')
    parser.add_argument('--state_timing_coef', type=float, default=0.5,
                        help='State effect on timing emission')
    parser.add_argument('--state_spend_coef', type=float, default=0.8,
                        help='State effect on spend emission')
    parser.add_argument('--segment_config', type=str, default=None,
                        help='Path to JSON segment config file')
    parser.add_argument('--no_covariates', action='store_true',
                        help='Exclude lagged RFM covariates')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    parser.add_argument('--format', type=str, default='npz',
                        choices=['npz', 'csv'],
                        help='Output format: npz or csv')
    parser.add_argument('--out_dir', type=str, default='.',
                        help='Output directory for .npz/.csv files')
    parser.add_argument('--output', type=str, default=None,
                        help='Output .npz filename')
    
    args = parser.parse_args()
    
    persistence = [float(x) for x in args.state_persistence.split(',')]
    if len(persistence) == 1:
        persistence = persistence[0]
    
    data = generate_world(
        world=args.world,
        N=args.N,
        T=args.T,
        seed=args.seed,
        K=args.K,
        target_sparsity=args.sparsity,
        state_persistence=persistence,
        gamma_h=args.gamma_h,
        gamma_m=args.gamma_m,
        rho=args.rho,
        state_timing_coef=args.state_timing_coef,
        state_spend_coef=args.state_spend_coef,
        segment_config=args.segment_config,
        include_covariates=not args.no_covariates,
        verbose=not args.quiet
    )
    
    if args.output is None:
        if args.format == 'csv':
            args.output = "dgp_{}_N{}_T{}_seed{}.csv".format(
                args.world, args.N, args.T, args.seed)
        else:
            args.output = "dgp_{}_N{}_T{}_seed{}.npz".format(
                args.world, args.N, args.T, args.seed)
    
    if args.format == 'csv':
        df = save_csv(data, args.output)
        if not args.quiet:
            print("\nSaved to: {}".format(args.output))
            print("Rows: {}, Customers: {}, Periods: {}".format(
                len(df), df['customer_id'].nunique(), df['t'].nunique()))
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / args.output
    np.savez(output_path, **data)
    if not args.quiet:
        print("\nSaved to: {}".format(args.output))
        print("File size: {:.1f} MB".format(Path(args.output).stat().st_size / 1024**2))


if __name__ == "__main__":
    main()
