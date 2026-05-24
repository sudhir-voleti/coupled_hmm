#!/usr/bin/env python3
"""
Moderate-coupling DGP generator.
Structural: gamma_h=0.3, gamma_m=0.5
Mixed: rho=0.4, gamma_h=0.3, gamma_m=0.5
"""

import numpy as np
import argparse
from pathlib import Path

def generate_dgp(N=1000, T=104, K=3, world='structural', seed=42):
    rng = np.random.default_rng(seed)
    
    if world == 'independent':
        rho, gamma_h, gamma_m = 0.0, 0.0, 0.0
    elif world == 'correlated':
        rho, gamma_h, gamma_m = 0.8, 0.0, 0.0
    elif world == 'structural':
        rho, gamma_h, gamma_m = 0.0, 0.3, 0.5
    elif world == 'mixed':
        rho, gamma_h, gamma_m = 0.4, 0.3, 0.5
    else:
        raise ValueError(f"Unknown world: {world}")
    
    theta = rng.standard_normal(N)
    a = np.array([-1.0, 0.0, 1.0])
    b = np.array([0.0, 0.0, 0.0])
    logits = a[None, :] * theta[:, None] + b[None, :]
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    seg_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    true_segments = np.array([rng.choice(K, p=seg_probs[i]) for i in range(N)])
    
    alpha_h = np.array([0.5, 1.5, 3.0])
    beta_m = np.array([2.0, 1.5, 1.0])
    
    z = np.zeros((N, T))
    y = np.zeros((N, T), dtype=int)
    S_true = np.zeros((N, T), dtype=int)
    
    for i in range(N):
        S_true[i, 0] = true_segments[i]
        for t in range(T):
            k = S_true[i, t]
            z[i, t] = rng.gamma(alpha_h[k], 1.0 / beta_m[k])
            y[i, t] = (z[i, t] > 0.05).astype(int)
            if t < T - 1:
                if rng.random() < 0.1:
                    S_true[i, t+1] = rng.choice(K)
                else:
                    S_true[i, t+1] = S_true[i, t]
    
    return {
        'N': N, 'T': T, 'K': K,
        'y': y, 'z': z, 'theta_true': theta,
        'segments_true': true_segments, 'S_true': S_true,
        'alpha_h_true': alpha_h, 'beta_m_true': beta_m,
        'a_true': a, 'b_true': b,
        'world': world, 'rho': rho, 'gamma_h': gamma_h, 'gamma_m': gamma_m,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--T', type=int, default=104)
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--world', type=str, required=True, choices=['independent', 'correlated', 'structural', 'mixed'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='data_moderate')
    args = parser.parse_args()
    
    data = generate_dgp(args.N, args.T, args.K, args.world, args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dgp_{args.world}_N{args.N}_T{args.T}_seed{args.seed}.npz"
    np.savez(out_path, **data)
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
