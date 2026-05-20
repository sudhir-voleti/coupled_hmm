#!/usr/bin/env python3
"""
Heckman-style selection model for JMR horse race.
Two-stage: Probit incidence + Gamma spend with Mills ratio correction.
Captures statistical coupling via selection parameter delta.
No individual-level parameters — global correlation only.
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
        fn=forward_step, sequences=[log_emit_seq],
        outputs_info=[log_alpha_norm_init, log_Z_init],
        non_sequences=[log_Gamma], strict=True
    )
    log_alpha_norm_full = pt.concatenate([log_alpha_norm_init[None, :, :], log_alpha_norm_seq], axis=0)
    log_marginal = log_Z_init.squeeze() + pt.sum(log_Z_seq.squeeze(), axis=0)
    return log_marginal, log_alpha_norm_full


def make_heckman_hmm(data, K, prior_log_alpha_gamma_mu=0.0, prior_log_alpha_gamma_sigma=1.0):
    """
    Heckman selection model with HMM states.
    Stage 1: Probit for incidence (NBD approximated by Poisson/Probit)
    Stage 2: Gamma for spend with Mills ratio correction for selection bias.
    """
    y = data['y']
    z = data['z']
    mask = data['mask']
    N, T = data['N'], data['T']

    with pm.Model(coords={"customer": np.arange(N), "time": np.arange(T), "state": np.arange(K)}) as model:
        if K == 1:
            pi0 = pt.as_tensor_variable(np.array([1.0], dtype=np.float32))
            Gamma = pt.as_tensor_variable(np.array([[1.0]], dtype=np.float32))
            log_Gamma = pt.as_tensor_variable(np.array([[0.0]], dtype=np.float32))
        else:
            Gamma = pm.Dirichlet("Gamma", a=np.ones(K) * 1.1, shape=(K, K))
            pi0 = pm.Dirichlet("pi0", a=np.ones(K, dtype=np.float32))
            log_Gamma = pt.log(Gamma)

        # =====================================================================
        # STAGE 1: Probit incidence (approximates NBD zero-inflation)
        # =====================================================================
        if K == 1:
            alpha_h_raw = pm.Normal("alpha_h_raw", 0, 1)
            alpha_h = alpha_h_raw
        else:
            alpha_h_raw = pm.Normal("alpha_h_raw", 0, 1, shape=K)
            alpha_h = pm.Deterministic("alpha_h", pt.sort(alpha_h_raw))

        # Probit probability of purchase
        if K == 1:
            probit_p = pm.math.invprobit(alpha_h)
            log_p_zero = pt.log(1 - probit_p + 1e-10)
            log_p_one = pt.log(probit_p + 1e-10)
        else:
            probit_p = pm.math.invprobit(alpha_h[None, None, :])
            log_p_zero = pt.log(1 - probit_p + 1e-10)
            log_p_one = pt.log(probit_p + 1e-10)

        # =====================================================================
        # STAGE 2: Gamma spend with Mills ratio correction
        # =====================================================================
        if K == 1:
            beta_m_raw = pm.Normal("beta_m_raw", 0, 1)
            beta_m = beta_m_raw
            log_alpha_gamma_raw = pm.Normal("log_alpha_gamma_raw", prior_log_alpha_gamma_mu, prior_log_alpha_gamma_sigma)
            log_alpha_gamma = log_alpha_gamma_raw
        else:
            beta_m_raw = pm.Normal("beta_m_raw", 0, 1, shape=K)
            beta_m = pm.Deterministic("beta_m", pt.sort(beta_m_raw))
            log_alpha_gamma_raw = pm.Normal("log_alpha_gamma_raw", prior_log_alpha_gamma_mu, prior_log_alpha_gamma_sigma, shape=K)
            log_alpha_gamma = pm.Deterministic("log_alpha_gamma", pt.sort(log_alpha_gamma_raw))

        alpha_gamma = pt.exp(log_alpha_gamma)

        # Mills ratio: lambda(a) = phi(a) / Phi(a)
        # For numerical stability, clip Phi away from 0 and 1
        phi_std = (1 / pt.sqrt(2 * np.pi)) * pt.exp(-0.5 * alpha_h**2) if K == 1 else (1 / pt.sqrt(2 * np.pi)) * pt.exp(-0.5 * alpha_h[None, None, :]**2)
        Phi_clip = pt.clip(probit_p, 0.001, 0.999)
        mills = phi_std / Phi_clip

        # Coupling parameter: delta (selection bias)
        delta = pm.Normal("delta", 0, 1)  # GLOBAL correlation parameter

        # Spend mean with Mills correction
        if K == 1:
            log_mu = beta_m + delta * mills
        else:
            log_mu = beta_m[None, None, :] + delta * mills

        mu = pt.exp(pt.clip(log_mu, -5, 5))
        beta_gamma = alpha_gamma / mu

        # =====================================================================
        # EMISSION LIKELIHOOD
        # =====================================================================
        if K == 1:
            z_clipped = pt.clip(z, 1e-10, 1e10)
            log_gamma_density = ((alpha_gamma - 1) * pt.log(z_clipped) - beta_gamma * z_clipped + alpha_gamma * pt.log(beta_gamma) - pt.gammaln(alpha_gamma))
            log_pos = log_p_one + log_gamma_density
            log_emission = pt.where(y == 0, log_p_zero, log_pos)
            log_emission = pt.where(mask, log_emission, 0.0)
            logp_cust = pt.sum(log_emission, axis=1)
        else:
            y_exp = y[..., None]
            z_exp = z[..., None]
            mask_exp = mask[..., None]

            z_clipped = pt.clip(z_exp, 1e-10, 1e10)
            alpha_exp = alpha_gamma[None, None, :]
            beta_exp = beta_gamma

            log_gamma_density = ((alpha_exp - 1) * pt.log(z_clipped) - beta_exp * z_exp + alpha_exp * pt.log(beta_exp) - pt.gammaln(alpha_exp))
            log_pos = log_p_one + log_gamma_density
            log_emission = pt.where(pt.eq(y_exp, 0), log_p_zero, log_pos)
            log_emission = pt.where(mask_exp, log_emission, 0.0)

            logp_cust, log_alpha_norm = forward_algorithm_scan(log_emission, log_Gamma, pi0)
            alpha_filtered = pt.exp(log_alpha_norm.swapaxes(0, 1))
            pm.Deterministic("alpha_filtered", alpha_filtered, dims=("customer", "time", "state"))

        pm.Deterministic("log_likelihood", logp_cust, dims=("customer",))
        pm.Potential("loglike", pt.sum(logp_cust))
        return model


def run_smc(data, K, draws, chains, seed, out_dir, prior_log_alpha_gamma_mu=0.0, prior_log_alpha_gamma_sigma=1.0):
    cores = min(chains, 4)
    t0 = time.time()

    with make_heckman_hmm(data, K, prior_log_alpha_gamma_mu, prior_log_alpha_gamma_sigma) as model:
        print("\nHECKMAN-Gamma: K={}, N={}, T={}".format(K, data['N'], data['T']))
        print("SMC: draws={}, chains={}, cores={}".format(draws, chains, cores))

        idata = pm.sample_smc(draws=draws, chains=chains, cores=cores, random_seed=seed, return_inferencedata=True)
        elapsed = (time.time() - t0) / 60

        try:
            ess = az.ess(idata)
            ess_min = float(min([ess[v].values.min() for v in ess.data_vars if hasattr(ess[v].values, 'size')]))
        except:
            ess_min = np.nan

        try:
            rhat = az.rhat(idata)
            rhat_max = float(max([rhat[v].values.max() for v in rhat.data_vars if hasattr(rhat[v].values, 'size')]))
        except:
            rhat_max = np.nan

        try:
            lm = idata.sample_stats.log_marginal_likelihood.values
            if lm.dtype == object:
                chain_finals = []
                for chain_vals in lm.flatten():
                    if isinstance(chain_vals, (list, tuple, np.ndarray)):
                        valid = [float(v) for v in chain_vals if np.isfinite(v)]
                        if valid: chain_finals.append(valid[-1])
                    elif np.isfinite(chain_vals):
                        chain_finals.append(float(chain_vals))
                log_ev = float(np.mean(chain_finals)) if chain_finals else np.nan
            else:
                flat = np.array(lm).flatten()
                valid = flat[np.isfinite(flat)]
                log_ev = float(np.mean(valid)) if len(valid) > 0 else np.nan
        except:
            log_ev = np.nan

        recovery = {}
        if 'true_states' in data and data['true_states'] is not None:
            recovery['S_true'] = data['true_states'].tolist()

        res = {
            'meta': {'model_type': 'HECKMAN-Gamma', 'K': K, 'N': data['N'], 'T': data['T'], 'world': data.get('world', 'unknown'), 'draws': draws, 'chains': chains, 'seed': seed, 'timestamp': datetime.now().isoformat()},
            'diagnostics': {'ess_min': ess_min, 'rhat_max': rhat_max, 'log_evidence': log_ev, 'time_min': elapsed},
            'recovery': recovery, 'predictive': {}, 'bdt': {},
            'data_ref': {'dgp_path': str(data.get('source_path', 'unknown')), 'train_ratio': data.get('train_ratio', 1.0)},
        }

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = out_dir / "smc_K{}_HECKMAN_Gamma_N{}_T{}_D{}.pkl".format(K, data['N'], data['T'], draws)

        with open(pkl_path, 'wb') as f:
            pickle.dump({'idata': idata, 'res': res}, f, protocol=4)

        print("Saved: {}".format(pkl_path))
        return pkl_path, res, idata


def main():
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from pycode.loaders.dgp_loader import load_dgp_npz

    parser = argparse.ArgumentParser(description='HECKMAN-Gamma for JMR')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--K', type=int, required=True, choices=[1, 2, 3])
    parser.add_argument('--draws', type=int, default=500)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default='./outputs')
    parser.add_argument('--prior_log_alpha_gamma_mu', type=float, default=0.0)
    parser.add_argument('--prior_log_alpha_gamma_sigma', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    data = load_dgp_npz(args.data_path, train_ratio=1.0)
    pkl_path, res, idata = run_smc(data=data, K=args.K, draws=args.draws, chains=args.chains, seed=args.seed, out_dir=args.out_dir, prior_log_alpha_gamma_mu=args.prior_log_alpha_gamma_mu, prior_log_alpha_gamma_sigma=args.prior_log_alpha_gamma_sigma)
    print("\nComplete. Log-ev: {:.2f}, ESS min: {:.0f}, Time: {:.1f}min".format(res['diagnostics']['log_evidence'], res['diagnostics']['ess_min'], res['diagnostics']['time_min']))
    print("PKL: {}".format(pkl_path))


if __name__ == "__main__":
    main()
EOF

echo "Saved: pycode/models/smc_heckman_gamma.py"
