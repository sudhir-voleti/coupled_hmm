#!/usr/bin/env python3
"""
Minimal BEMMAOR HMM with Log-Normal spend emissions.
Dual observations: y (Poisson count, timing) and z (continuous, spend | y>0).
"""

import os
os.environ['PYTENSOR_FLAGS'] = 'floatX=float32,optimizer=fast_run,openmp=True'

import numpy as np
import pytensor.tensor as pt
from pytensor import scan
import pymc as pm
import arviz as az
import time
import pickle
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
RANDOM_SEED = 42


def forward_algorithm_scan(log_emission, log_Gamma, pi0):
    """Batched forward algorithm."""
    N, T, K = log_emission.shape

    log_alpha_init = pt.log(pi0)[None, :] + log_emission[:, 0, :]
    log_Z_init = pt.logsumexp(log_alpha_init, axis=1, keepdims=True)
    log_alpha_norm_init = log_alpha_init - log_Z_init

    def forward_step(log_emit_t, log_alpha_prev, log_Z_prev, log_Gamma):
        transition = log_alpha_prev[:, :, None] + log_Gamma[None, :, :]
        log_alpha_new = log_emit_t + pt.logsumexp(transition, axis=1)
        log_Z_t = pt.logsumexp(log_alpha_new, axis=1, keepdims=True)
        log_alpha_norm = log_alpha_new - log_Z_t
        return log_alpha_norm, log_Z_t

    log_emit_seq = log_emission[:, 1:, :].swapaxes(0, 1)

    (log_alpha_norm_seq, log_Z_seq), _ = scan(
        fn=forward_step,
        sequences=[log_emit_seq],
        outputs_info=[log_alpha_norm_init, log_Z_init],
        non_sequences=[log_Gamma],
        strict=True
    )

    log_alpha_norm_full = pt.concatenate([
        log_alpha_norm_init[None, :, :],
        log_alpha_norm_seq
    ], axis=0)

    log_marginal = log_Z_init.squeeze() + pt.sum(log_Z_seq.squeeze(), axis=0)

    return log_marginal, log_alpha_norm_full


def make_bemmaor_ln(data, K, pilot=False, prior_log_sigma_s_mu=0.0, prior_log_sigma_s_sigma=1.0):
    """
    BEMMAOR with dual observations:
      y: Poisson count (timing, zero-inflated via NBD)
      z: Continuous spend (log-normal, observed only where y>0)
    Full order constraints: all state parameters sorted ascending.
    """
    y = data['y']
    z = data['z']
    mask = data['mask']
    N, T = data['N'], data['T']

    if pilot:
        print("[PILOT] BEMMAOR-LN dual: N={}, T={}, K={}".format(N, T, K))

    with pm.Model(coords={
        "customer": np.arange(N),
        "time": np.arange(T),
        "state": np.arange(K)
    }) as model:

        # 1. LATENT DYNAMICS
        if K == 1:
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype=np.float32))
            Gamma = pt.as_tensor_variable(np.array([[1.0]], dtype=np.float32))
            log_Gamma = pt.as_tensor_variable(np.array([[0.0]], dtype=np.float32))
        else:
            Gamma = pm.Dirichlet("Gamma", a=np.ones(K) * 1.1, shape=(K, K))
            pi0 = pm.Dirichlet("pi0", a=np.ones(K, dtype=np.float32))
            log_Gamma = pt.log(Gamma)

        # 2. SHARED LATENT FACTOR
        theta = pm.Normal("theta", mu=0, sigma=1, shape=(N, 1))
        gamma_m = pm.HalfNormal("gamma_m", sigma=1.0)
        gamma_h = pm.Normal("gamma_h", mu=0, sigma=1.0)

        # 3. NBD PART (Timing / y) - with order constraints
        if K == 1:
            alpha_h_raw = pm.Normal("alpha_h_raw", 0, 1)
            alpha_h = alpha_h_raw
            log_r_raw = pm.Normal("log_r_raw", 0, 1)
            log_r = log_r_raw
        else:
            # Full order constraints: all state parameters sorted ascending
            alpha_h_raw = pm.Normal("alpha_h_raw", 0, 1, shape=K)
            alpha_h = pm.Deterministic("alpha_h", pt.sort(alpha_h_raw))
            
            log_r_raw = pm.Normal("log_r_raw", 0, 1, shape=K)
            log_r = pm.Deterministic("log_r", pt.sort(log_r_raw))

        r_nbd = pt.exp(log_r)

        if K == 1:
            log_lam = alpha_h + gamma_h * theta
        else:
            log_lam = alpha_h[None, None, :] + gamma_h * theta[:, :, None]

        lam = pt.exp(pt.clip(log_lam, -10, 10))

        if K == 1:
            log_p_zero_nbd = r_nbd * (pt.log(r_nbd) - pt.log(r_nbd + lam.squeeze()))
        else:
            r_exp = r_nbd[None, None, :]
            log_p_zero_nbd = r_exp * (pt.log(r_exp) - pt.log(r_exp + lam))

        # 4. LOG-NORMAL PART (Spend / z, only where y>0)
        if K == 1:
            beta_m_raw = pm.Normal("beta_m_raw", 0, 1)
            beta_m = beta_m_raw
            log_sigma_s_raw = pm.Normal("log_sigma_s_raw", prior_log_sigma_s_mu, prior_log_sigma_s_sigma)
            log_sigma_s = log_sigma_s_raw
        else:
            beta_m_raw = pm.Normal("beta_m_raw", 0, 1, shape=K)
            beta_m = pm.Deterministic("beta_m", pt.sort(beta_m_raw))
            
            log_sigma_s_raw = pm.Normal("log_sigma_s_raw", prior_log_sigma_s_mu, prior_log_sigma_s_sigma, shape=K)
            log_sigma_s = pm.Deterministic("log_sigma_s", pt.sort(log_sigma_s_raw))

        sigma_s = pt.exp(log_sigma_s)

        if K == 1:
            log_mu = beta_m + gamma_m * theta.squeeze()
        else:
            log_mu = beta_m[None, None, :] + gamma_m * theta[:, :, None]

        mu = pt.exp(pt.clip(log_mu, -10, 10))

        # 5. DUAL EMISSION LIKELIHOOD
        if K == 1:
            log_zero = log_p_zero_nbd

            z_clipped = pt.clip(z, 1e-10, 1e10)
            log_z = pt.log(z_clipped)
            log_norm_const = -log_sigma_s - 0.5 * pt.log(2 * np.pi)
            log_quadratic = -0.5 * ((log_z - log_mu) / sigma_s)**2
            log_spend_density = log_norm_const + log_quadratic - log_z

            log_p_pos = pt.log1p(-pt.exp(log_zero) + 1e-10)
            log_pos = log_p_pos + log_spend_density
            log_emission = pt.where(y == 0, log_zero, log_pos)
            log_emission = pt.where(mask, log_emission, 0.0)
            logp_cust = pt.sum(log_emission, axis=1)

        else:
            y_exp = y[..., None]
            z_exp = z[..., None]
            mask_exp = mask[..., None]

            log_zero = log_p_zero_nbd
            log_p_pos = pt.log1p(-pt.exp(log_zero) + 1e-10)

            z_clipped = pt.clip(z_exp, 1e-10, 1e10)
            log_z = pt.log(z_clipped)
            sigma_exp = sigma_s[None, None, :]
            log_norm_const = -pt.log(sigma_exp) - 0.5 * pt.log(2 * np.pi)
            log_quadratic = -0.5 * ((log_z - log_mu) / sigma_exp)**2
            log_spend_density = log_norm_const + log_quadratic - log_z

            log_pos = log_p_pos + log_spend_density
            log_emission = pt.where(pt.eq(y_exp, 0), log_zero, log_pos)
            log_emission = pt.where(mask_exp, log_emission, 0.0)

            logp_cust, log_alpha_norm = forward_algorithm_scan(log_emission, log_Gamma, pi0)

            alpha_filtered = pt.exp(log_alpha_norm.swapaxes(0, 1))
            pm.Deterministic("alpha_filtered", alpha_filtered,
                           dims=("customer", "time", "state"))

        # 6. LIKELIHOOD
        pm.Deterministic("log_likelihood", logp_cust, dims=("customer",))
        pm.Potential("loglike", pt.sum(logp_cust))

        return model

def run_smc(data, K, draws, chains, seed, out_dir, prior_log_sigma_s_mu=0.0, prior_log_sigma_s_sigma=1.0):
    """SMC runner with standardized PKL structure."""
    cores = min(chains, 4)
    t0 = time.time()

    with make_bemmaor_ln(data, K, prior_log_sigma_s_mu=prior_log_sigma_s_mu, prior_log_sigma_s_sigma=prior_log_sigma_s_sigma) as model:
        print("\nBEMMAOR-LN dual: K={}, N={}, T={}".format(K, data['N'], data['T']))
        print("SMC: draws={}, chains={}, cores={}".format(draws, chains, cores))

        idata = pm.sample_smc(
            draws=draws,
            chains=chains,
            cores=cores,
            random_seed=seed,
            return_inferencedata=True
        )

    elapsed = (time.time() - t0) / 60

    # Diagnostics
    try:
        ess = az.ess(idata)
        ess_by_var = {}
        for v in ess.data_vars:
            if hasattr(ess[v].values, 'size'):
                ess_by_var[str(v)] = float(ess[v].values.min())
        ess_min = float(min(ess_by_var.values())) if ess_by_var else np.nan
    except:
        ess_by_var, ess_min = {}, np.nan

    try:
        rhat = az.rhat(idata)
        rhat_max = float(max([rhat[v].values.max() for v in rhat.data_vars if hasattr(rhat[v].values, 'size')]))
    except:
        rhat_max = np.nan

    # Log-evidence: extract last valid value per chain (final tempering stage)
    try:
        lm = idata.sample_stats.log_marginal_likelihood.values
        if lm.dtype == object:
            chain_finals = []
            for chain_vals in lm.flatten():
                if isinstance(chain_vals, (list, tuple, np.ndarray)):
                    valid = [float(v) for v in chain_vals if np.isfinite(v)]
                    if valid:
                        chain_finals.append(valid[-1])
                elif np.isfinite(chain_vals):
                    chain_finals.append(float(chain_vals))
            log_ev = float(np.mean(chain_finals)) if chain_finals else np.nan
        else:
            flat = np.array(lm).flatten()
            valid = flat[np.isfinite(flat)]
            log_ev = float(np.mean(valid)) if len(valid) > 0 else np.nan
    except Exception as e:
        log_ev = np.nan
        print("Log-ev extraction failed:", str(e)[:100])

    # Recovery (ground truth comparison)
    recovery = {}
    if 'true_states' in data and data['true_states'] is not None:
        recovery['S_true'] = data['true_states'].tolist()
    if 'segments_true' in data and data['segments_true'] is not None:
        recovery['segments_true'] = data['segments_true'].tolist()
    if 'theta_true' in data and data['theta_true'] is not None:
        recovery['theta_true'] = data['theta_true'].tolist()

    res = {
        'meta': {
            'model_type': 'BEMMAOR-LN-dual',
            'K': K,
            'N': data['N'],
            'T': data['T'],
            'world': data.get('world', 'unknown'),
            'draws': draws,
            'chains': chains,
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
        },
        'diagnostics': {
            'ess_min': ess_min,
            'ess_by_var': ess_by_var,
            'rhat_max': rhat_max,
            'log_evidence': log_ev,
            'time_min': elapsed,
        },
        'recovery': recovery,
        'predictive': {},
        'bdt': {},
        'data_ref': {
            'dgp_path': str(data.get('source_path', 'unknown')),
            'train_ratio': data.get('train_ratio', 1.0),
        },
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = out_dir / "smc_K{}_BEMMAOR_LN_dual_N{}_T{}_D{}.pkl".format(
        K, data['N'], data['T'], draws)

    with open(pkl_path, 'wb') as f:
        pickle.dump({'idata': idata, 'res': res}, f, protocol=4)

    print("Saved: {}".format(pkl_path))
    return pkl_path, res, idata


def main():
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from pycode.loaders.dgp_loader import load_dgp_npz

    parser = argparse.ArgumentParser(description='BEMMAOR-LN dual obs')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--K', type=int, required=True, choices=[1, 2, 3])
    parser.add_argument('--draws', type=int, default=500)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default='./outputs')
    parser.add_argument('--prior_log_sigma_s_mu', type=float, default=0.0)
    parser.add_argument('--prior_log_sigma_s_sigma', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    data = load_dgp_npz(args.data_path, train_ratio=1.0)

    pkl_path, res, idata = run_smc(
        data=data, K=args.K, draws=args.draws,
        chains=args.chains, seed=args.seed, out_dir=args.out_dir,
        prior_log_sigma_s_mu=args.prior_log_sigma_s_mu,
        prior_log_sigma_s_sigma=args.prior_log_sigma_s_sigma
    )

    print("\nComplete. Log-ev: {:.2f}, ESS min: {:.0f}, Time: {:.1f}min".format(
        res['diagnostics']['log_evidence'], res['diagnostics']['ess_min'], res['diagnostics']['time_min']))
    print("PKL saved to: {}".format(pkl_path))


if __name__ == "__main__":
    main()
