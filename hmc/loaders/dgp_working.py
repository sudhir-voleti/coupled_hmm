#!/usr/bin/env python3
"""
FILE: dgp_rfree_27may_v2.py
CREATED: 2026-05-27
CHAT: jmr_27mayv1
STATUS: DGP with T_full = T + 20 for OOS ground truth
PYMC: 5.26.1, ARVIZ: 0.22.0
PURPOSE: Generate T_full periods, models train on T, OOS evaluates last 20
NEXT: Regenerate all 4 worlds
"""
import numpy as np
import pickle
import os
import argparse


def generate_dgp(world, N=100, T=52, seed=42):
    rng = np.random.RandomState(seed)
    T_full = T + 20  # extra 20 periods for OOS

    world_params = {
        'independent': {
            'Gamma': np.array([[0.90, 0.05, 0.05], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]]),
            'rho': 0.0, 'gamma_m_true': 0.0
        },
        'correlated': {
            'Gamma': np.array([[0.70, 0.20, 0.10], [0.15, 0.70, 0.15], [0.10, 0.20, 0.70]]),
            'rho': 0.4, 'gamma_m_true': 0.5
        },
        'mixed': {
            'Gamma': np.array([[0.80, 0.15, 0.05], [0.10, 0.80, 0.10], [0.05, 0.15, 0.80]]),
            'rho': 0.2, 'gamma_m_true': 0.3
        },
        'structural': {
            'Gamma': np.array([[0.85, 0.10, 0.05], [0.15, 0.70, 0.15], [0.05, 0.20, 0.75]]),
            'rho': 0.0, 'gamma_m_true': 0.5
        }
    }

    wp = world_params.get(world, world_params['structural'])
    Gamma = wp['Gamma']
    rho = wp['rho']
    gamma_m_true = wp['gamma_m_true']

    alpha_h = np.array([0.0, 1.0, 2.0])
    beta_m = np.array([0.0, 1.5, 3.0])
    shape_spend = 2.0
    gamma_h_true = 1.0
    theta = rng.normal(0, 1, size=N)

    r_nb = np.array([1.0, 2.0, 3.0])
    sigma_t = 0.3
    sigma_s = 0.5
    Sigma = [[sigma_t**2, rho*sigma_t*sigma_s], [rho*sigma_t*sigma_s, sigma_s**2]]
    errors = rng.multivariate_normal([0, 0], Sigma, size=(N, T_full))

    Z = np.zeros((N, T_full), dtype=int)
    for i in range(N):
        Z[i, 0] = rng.choice(3, p=[0.4, 0.4, 0.2])
        for t in range(1, T_full):
            Z[i, t] = rng.choice(3, p=Gamma[Z[i, t-1]])

    Y_timing = np.zeros((N, T_full), dtype=int)
    Y_spend = np.zeros((N, T_full))

    for i in range(N):
        for t in range(T_full):
            k = Z[i, t]
            log_lam = alpha_h[k] + gamma_h_true * theta[i] + errors[i, t, 0]
            lam = np.exp(log_lam)
            p_nb = r_nb[k] / (r_nb[k] + lam)
            Y_timing[i, t] = rng.negative_binomial(r_nb[k], p_nb)

            log_mu = beta_m[k] + gamma_m_true * theta[i] + errors[i, t, 1]
            mu_spend = np.exp(log_mu)
            scale = mu_spend / shape_spend
            Y_spend[i, t] = rng.gamma(shape=shape_spend, scale=scale)

    return {
        'N': N, 'T': T, 'T_full': T_full, 'seed': seed, 'world': world,
        'alpha_h_true': alpha_h, 'beta_m_true': beta_m,
        'shape_spend_true': shape_spend, 'r_nb_true': r_nb,
        'gamma_h_true': gamma_h_true, 'gamma_m_true': gamma_m_true,
        'theta_true': theta, 'rho_true': rho,
        'sigma_t_true': sigma_t, 'sigma_s_true': sigma_s,
        'Gamma_true': Gamma, 'Z_true': Z,
        'Y_timing': Y_timing, 'Y_spend': Y_spend
    }


def print_dgp_stats(dgp):
    Y_timing = dgp['Y_timing']
    Y_spend = dgp['Y_spend']
    Z_true = dgp['Z_true']
    N, T_full = Y_timing.shape
    T = dgp['T']
    r_nb = dgp['r_nb_true']

    print('\n=== DGP DESCRIPTIVE STATISTICS ===')
    print('World:  ' + dgp['world'])
    print('N:      ' + str(N))
    print('T:      ' + str(T) + ' (training)')
    print('T_full: ' + str(T_full) + ' (including ' + str(T_full - T) + ' OOS periods)')
    print('r_nb:   ' + str(r_nb.tolist()) + ' (per state)')

    # Training stats
    Y_timing_train = Y_timing[:, :T]
    Y_spend_train = Y_spend[:, :T]
    overall_sparsity = np.mean(Y_timing_train == 0)
    print('\n--- TRAINING PERIODS (1-' + str(T) + ') ---')
    print('Sparsity (timing=0):     ' + str(round(overall_sparsity * 100, 1)) + '%')
    print('Mean timing (all):       ' + str(round(np.mean(Y_timing_train), 2)))
    print('Mean timing (if >0):     ' + str(round(np.mean(Y_timing_train[Y_timing_train > 0]), 2)))
    print('Mean spend (all):        ' + str(round(np.mean(Y_spend_train), 2)))
    print('Mean spend (if >0):      ' + str(round(np.mean(Y_spend_train[Y_spend_train > 0]), 2)))

    # OOS stats
    Y_timing_oos = Y_timing[:, T:]
    Y_spend_oos = Y_spend[:, T:]
    oos_sparsity = np.mean(Y_timing_oos == 0)
    print('\n--- OOS PERIODS (' + str(T+1) + '-' + str(T_full) + ') ---')
    print('Sparsity (timing=0):     ' + str(round(oos_sparsity * 100, 1)) + '%')
    print('Mean timing (all):       ' + str(round(np.mean(Y_timing_oos), 2)))
    print('Mean spend (all):        ' + str(round(np.mean(Y_spend_oos), 2)))

    print('\n--- STATE-SPECIFIC (training) ---')
    for k in range(3):
        mask = Z_true[:, :T] == k
        if mask.sum() == 0:
            continue
        y_t_k = Y_timing_train[mask]
        y_s_k = Y_spend_train[mask]
        sparsity_k = np.mean(y_t_k == 0)
        mean_t_k = np.mean(y_t_k)
        mean_s_k = np.mean(y_s_k)
        var_t_k = np.var(y_t_k)
        var_s_k = np.var(y_s_k)
        vmr_t = var_t_k / mean_t_k if mean_t_k > 0 else np.nan
        vmr_s = var_s_k / mean_s_k if mean_s_k > 0 else np.nan
        print('State ' + str(k) + ' (r=' + str(r_nb[k]) + '):')
        print('  Sparsity:    ' + str(round(sparsity_k * 100, 1)) + '%')
        print('  Mean timing: ' + str(round(mean_t_k, 2)) + ' (var/mean=' + str(round(vmr_t, 2)) + ')')
        print('  Mean spend:  ' + str(round(mean_s_k, 2)) + ' (var/mean=' + str(round(vmr_s, 2)) + ')')

    print('\n--- THETA DISTRIBUTION ---')
    theta = dgp['theta_true']
    print('Mean:  ' + str(round(np.mean(theta), 3)))
    print('Std:   ' + str(round(np.std(theta), 3)))
    print('Range: ' + str(round(np.min(theta), 3)) + ' to ' + str(round(np.max(theta), 3)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', type=str, default='structural')
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--T', type=int, default=52)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='outputs_27may')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dgp = generate_dgp(args.world, N=args.N, T=args.T, seed=args.seed)
    print_dgp_stats(dgp)

    pkl_path = os.path.join(args.output_dir, 'dgp_rfree_' + args.world + '_N' + str(args.N) + '_T' + str(args.T) + '.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(dgp, f)
    print('\nSaved DGP to ' + pkl_path)


if __name__ == '__main__':
    main()
