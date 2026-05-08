#!/usr/bin/env python3
"""
DGP for batched SMC pilot.
Global theta (single shared factor), auto-calibrated sparsity.
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
from scipy.optimize import minimize_scalar


def calibrate_alpha_h_for_sparsity(pi0_target, gamma_h, theta_global, r_nb_0=1.0):
    """Find alpha_h[0] that achieves target sparsity."""
    def sparsity_error(alpha_h_0):
        lam = np.exp(alpha_h_0 + gamma_h * theta_global)
        p_zero = (r_nb_0 / (r_nb_0 + lam)) ** r_nb_0
        return (p_zero - pi0_target) ** 2
    
    result = minimize_scalar(sparsity_error, bounds=(-5, 3), method='bounded')
    return float(result.x)


def generate_dgp(N=100, T=52, K=2, pi0_target=0.30, psi=5, rho=0.4, seed=42):
    """
    Generate coupled HMM data with GLOBAL theta.
    Sparsity is auto-calibrated via alpha_h[0].
    """
    rng = np.random.default_rng(seed)
    
    # Transition matrix - moderate persistence
    stickiness = 0.85
    Gamma = np.array([
        [stickiness, 1 - stickiness],
        [(1 - stickiness) * 0.7, 1 - (1 - stickiness) * 0.7]
    ])
    
    # Stationary initial distribution
    pi0_vec = np.array([0.5, 0.5])
    
    # State-specific magnitude parameters
    beta_m = np.array([1.0, 2.5])        # magnitude intercepts (log scale)
    alpha_gamma = np.array([2.0, 5.0])   # Gamma shape
    
    # GLOBAL theta - single draw for ALL customers
    theta_global = float(rng.normal(0, 1))
    theta = np.full((N, 1), theta_global)
    
    # Global coupling parameters
    gamma_h = float(rho * 0.6)
    gamma_m = float(rho * 1.0)
    
    # NB dispersion
    r_nb = np.array([1.0, 2.0])
    
    # AUTO-CALIBRATE alpha_h[0] for target sparsity
    alpha_h_1 = 0.5
    alpha_h_0 = calibrate_alpha_h_for_sparsity(pi0_target, gamma_h, theta_global, r_nb[0])
    alpha_h = np.array([alpha_h_0, alpha_h_1])
    
    # Generate latent states
    Z = np.zeros((N, T), dtype=int)
    for i in range(N):
        Z[i, 0] = rng.choice(K, p=pi0_vec)
        for t in range(1, T):
            Z[i, t] = rng.choice(K, p=Gamma[Z[i, t-1], :])
    
    # Generate observations
    Y = np.zeros((N, T), dtype=np.float32)
    
    for i in range(N):
        for t in range(T):
            k = Z[i, t]
            lam = np.exp(alpha_h[k] + gamma_h * theta_global)
            p_zero = (r_nb[k] / (r_nb[k] + lam)) ** r_nb[k]
            
            if rng.random() > p_zero:
                mu_spend = np.exp(beta_m[k] + gamma_m * theta_global)
                scale = mu_spend / alpha_gamma[k]
                Y[i, t] = rng.gamma(alpha_gamma[k], scale)
    
    actual_sparsity = float(np.mean(Y == 0))
    
    return {
        'Y': Y, 'Z': Z, 'Gamma': Gamma, 'pi0_vec': pi0_vec,
        'theta_global': theta_global, 'theta': theta,
        'gamma_h': gamma_h, 'gamma_m': gamma_m,
        'alpha_h': alpha_h, 'beta_m': beta_m,
        'alpha_gamma': alpha_gamma, 'r_nb': r_nb,
        'N': N, 'T': T, 'K': K, 'seed': seed,
        'pi0_target': pi0_target, 'psi': psi, 'rho': rho,
        'actual_sparsity': actual_sparsity,
    }


def save_dgp(dgp, out_dir='data', prefix='dgp'):
    """Save DGP to disk."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    fname = f"{prefix}_N{dgp['N']}_T{dgp['T']}_K{dgp['K']}_pi0{dgp['pi0_target']}_rho{dgp['rho']}_seed{dgp['seed']}.pkl"
    fpath = out_path / fname
    
    with open(fpath, 'wb') as f:
        pickle.dump(dgp, f)
    
    return fpath


def main():
    parser = argparse.ArgumentParser(description='Generate DGP for batched SMC')
    parser.add_argument('--N', type=int, default=100, help='Number of customers')
    parser.add_argument('--T', type=int, default=52, help='Number of periods')
    parser.add_argument('--K', type=int, default=2, help='Number of states')
    parser.add_argument('--pi0', type=float, default=0.30, help='Target sparsity')
    parser.add_argument('--psi', type=float, default=5, help='Volatility scaling')
    parser.add_argument('--rho', type=float, default=0.4, help='Coupling strength')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--out', type=str, default='data', help='Output directory')
    
    args = parser.parse_args()
    
    print(f"Generating DGP: N={args.N}, T={args.T}, K={args.K}")
    print(f"  pi0_target={args.pi0}, psi={args.psi}, rho={args.rho}, seed={args.seed}")
    
    dgp = generate_dgp(
        N=args.N, T=args.T, K=args.K,
        pi0_target=args.pi0, psi=args.psi, rho=args.rho,
        seed=args.seed
    )
    
    fpath = save_dgp(dgp, out_dir=args.out)
    
    print(f"\nSaved to: {fpath}")
    print(f"  theta_global: {dgp['theta_global']:.4f}")
    print(f"  gamma_h: {dgp['gamma_h']:.4f}, gamma_m: {dgp['gamma_m']:.4f}")
    print(f"  alpha_h: [{dgp['alpha_h'][0]:.4f}, {dgp['alpha_h'][1]:.4f}]")
    print(f"  Target sparsity: {dgp['pi0_target']:.2%}")
    print(f"  Actual sparsity: {dgp['actual_sparsity']:.2%}")
    print(f"  Sparsity error: {abs(dgp['actual_sparsity'] - dgp['pi0_target']):.2%}")
    print(f"  Y non-zero mean: {dgp['Y'][dgp['Y']>0].mean():.2f}")
    print(f"  Y max: {dgp['Y'].max():.2f}")
    
    if abs(dgp['actual_sparsity'] - dgp['pi0_target']) > 0.05:
        print(f"\nWARNING: Sparsity calibration may need tuning")

if __name__ == '__main__':
    main()


# Test it
python3 scripts/run_dgp.py --N 100 --T 52 --K 2 --pi0 0.30 --rho 0.4 --seed 42
