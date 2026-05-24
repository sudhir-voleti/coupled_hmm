#!/usr/bin/env python3
"""
DGP Generator with LogNormal theta_i and stronger state separation.
"""

import numpy as np
import argparse
from pathlib import Path

def generate_dgp(N, T, K, world, seed=42, theta_mu=-0.223, theta_sigma=0.5):
    """
    theta_i ~ LogNormal(mu, sigma) for positive, moderate-skew heterogeneity.
    Stronger state separation via wider beta_m spread.
    """
    np.random.seed(seed)
    
    configs = {
        'independent': {'rho': 0.0, 'gamma_h': 0.0, 'gamma_m': 0.0},
        'correlated': {'rho': 0.8, 'gamma_h': 0.0, 'gamma_m': 0.0},
        'structural': {'rho': 0.0, 'gamma_h': 0.3, 'gamma_m': 0.5},
        'mixed': {'rho': 0.4, 'gamma_h': 0.3, 'gamma_m': 0.5},
    }
    cfg = configs[world]
    rho, gamma_h, gamma_m = cfg['rho'], cfg['gamma_h'], cfg['gamma_m']
    
    # Theta_i: LogNormal distributed
    theta = np.random.lognormal(mean=theta_mu, sigma=theta_sigma, size=N)
    # Center to mean ≈ 0 for the linear predictor
    theta = theta - np.exp(theta_mu + theta_sigma**2 / 2)
    
    # State assignment
    pi0_true = np.array([0.4, 0.35, 0.25])
    Gamma_true = np.array([
        [0.85, 0.10, 0.05],
        [0.15, 0.75, 0.10],
        [0.05, 0.15, 0.80],
    ])
    
    S_true = np.zeros((N, T), dtype=int)
    for i in range(N):
        S_true[i, 0] = np.random.choice(K, p=pi0_true)
        for t in range(1, T):
            S_true[i, t] = np.random.choice(K, p=Gamma_true[S_true[i, t-1], :])
    
    # STRONGER STATE SEPARATION: wider beta_m spread
    alpha_h_true = np.sort(np.array([-2.0, -0.5, 1.0]))      # Wider spread in NBD
    beta_m_true = np.sort(np.array([0.5, 2.5, 5.0]))          # Wider spread in spend
    log_r_true = np.sort(np.array([0.5, 1.0, 1.5]))
    log_alpha_gamma_true = np.sort(np.array([0.5, 1.0, 1.5]))
    
    y = np.zeros((N, T), dtype=int)
    z = np.zeros((N, T))
    
    for i in range(N):
        for t in range(T):
            k = S_true[i, t]
            
            log_lam = alpha_h_true[k] + gamma_h * theta[i]
            lam = np.exp(np.clip(log_lam, -10, 10))
            r = np.exp(log_r_true[k])
            
            p_nb = r / (r + lam)
            n_nb = np.random.poisson(lam=lam)
            y[i, t] = n_nb
            
            if y[i, t] > 0:
                log_mu = beta_m_true[k] + gamma_m * theta[i]
                mu = np.exp(np.clip(log_mu, -10, 10))
                alpha_g = np.exp(log_alpha_gamma_true[k])
                beta_g = alpha_g / mu
                z[i, t] = np.random.gamma(shape=alpha_g, scale=1.0/beta_g)
    
    return {
        'y': y, 'z': z, 'S_true': S_true, 'theta': theta,
        'segments': (theta > np.median(theta)).astype(int),
        'N': N, 'T': T, 'K': K, 'world': world,
        'gamma_h': gamma_h, 'gamma_m': gamma_m, 'rho': rho,
        'alpha_h_true': alpha_h_true, 'beta_m_true': beta_m_true,
        'log_r_true': log_r_true, 'log_alpha_gamma_true': log_alpha_gamma_true,
        'Gamma_true': Gamma_true, 'pi0_true': pi0_true,
        'seed': seed, 'theta_mu': theta_mu, 'theta_sigma': theta_sigma,
    }

def save_dgp(data, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"dgp_{data['world']}_N{data['N']}_T{data['T']}_seed{data['seed']}.npz"
    path = out_dir / fname
    np.savez_compressed(path, **data)
    print(f"Saved: {path}")
    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=250)
    parser.add_argument('--T', type=int, default=104)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--world', type=str, required=True, choices=['independent','correlated','structural','mixed'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--theta_mu', type=float, default=-0.223)
    parser.add_argument('--theta_sigma', type=float, default=0.5)
    parser.add_argument('--out_dir', type=str, default='./data_lognormal_theta')
    args = parser.parse_args()
    
    data = generate_dgp(args.N, args.T, args.K, args.world, args.seed, args.theta_mu, args.theta_sigma)
    save_dgp(data, args.out_dir)

if __name__ == "__main__":
    main()
