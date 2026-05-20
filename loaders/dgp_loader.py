"""
DGP Loader: Converts DGP output (.npz or .csv) to model-ready dict.
Handles RFM computation, train/test splits, and standardization.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def compute_rfm_features(y, mask):
    """Compute lagged RFM features from panel data."""
    N, T = y.shape
    R = np.zeros((N, T), dtype=np.float32)
    F = np.zeros((N, T), dtype=np.float32)
    M = np.zeros((N, T), dtype=np.float32)

    for i in range(N):
        last_purchase = -1
        cum_freq = 0
        cum_spend = 0.0

        for t in range(T):
            if mask[i, t]:
                if y[i, t] > 0:
                    last_purchase = t
                    cum_freq += 1
                    cum_spend += y[i, t]

                if last_purchase >= 0:
                    R[i, t] = t - last_purchase
                    F[i, t] = cum_freq
                    M[i, t] = cum_spend / cum_freq if cum_freq > 0 else 0.0
                else:
                    R[i, t] = t + 1
                    F[i, t] = 0
                    M[i, t] = 0.0

    return R, F, M


def standardize_rfm(R_train, F_train, M_train, mask_train, R_test=None, F_test=None, M_test=None, mask_test=None):
    """Standardize RFM using training stats. Optionally standardize test arrays with same stats."""
    M_log_train = np.log1p(M_train)

    R_valid = R_train[mask_train]
    F_valid = F_train[mask_train]
    M_valid = M_log_train[mask_train]

    R_mean = R_valid.mean() if len(R_valid) > 0 else 0.0
    R_std = R_valid.std() + 1e-6
    F_mean = F_valid.mean() if len(F_valid) > 0 else 0.0
    F_std = F_valid.std() + 1e-6
    M_mean = M_valid.mean() if len(M_valid) > 0 else 0.0
    M_std = M_valid.std() + 1e-6

    R_scaled = (R_train - R_mean) / R_std
    F_scaled = (F_train - F_mean) / F_std
    M_scaled = (M_log_train - M_mean) / M_std

    out = {
        'R': R_scaled.astype(np.float32),
        'F': F_scaled.astype(np.float32),
        'M': M_scaled.astype(np.float32),
        'R_mean': float(R_mean), 'R_std': float(R_std),
        'F_mean': float(F_mean), 'F_std': float(F_std),
        'M_mean': float(M_mean), 'M_std': float(M_std),
    }

    if R_test is not None and F_test is not None and M_test is not None:
        M_log_test = np.log1p(M_test)
        R_test_scaled = (R_test - R_mean) / R_std
        F_test_scaled = (F_test - F_mean) / F_std
        M_test_scaled = (M_log_test - M_mean) / M_std
        out.update({
            'R_test': R_test_scaled.astype(np.float32),
            'F_test': F_test_scaled.astype(np.float32),
            'M_test': M_test_scaled.astype(np.float32),
        })

    return out


def load_dgp_npz(npz_path, train_ratio=1.0, seed=42):
    """
    Load DGP .npz file and convert to model-ready dict.

    Parameters
    ----------
    npz_path : str or Path
        Path to .npz file from dgp_gen.py
    train_ratio : float
        Fraction of time periods for training (1.0 = no split)
    seed : int
        Random seed (unused for DGP, kept for API consistency)

    Returns
    -------
    data : dict with keys N, T, y, mask, R, F, M, true_states,
           plus test keys if train_ratio < 1.0
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError("NPZ not found: {}".format(npz_path))

    d = np.load(npz_path, allow_pickle=True)

    y = d['y'].astype(np.float32)
    z = d['z'].astype(np.float32)
    S_true = d['S_true'].astype(np.int32)
    theta = d['theta'].astype(np.float32)
    segments = d['segments'].astype(np.int32)

    N, T = int(d['N']), int(d['T'])
    world = str(d.get('world', 'unknown'))
    seed_dgp = int(d.get('seed', 0))

    if y.shape != (N, T):
        raise ValueError("Shape mismatch: y={}, expected ({}, {})".format(y.shape, N, T))

    mask = np.ones((N, T), dtype=bool)

    if train_ratio < 1.0:
        T_train = int(T * train_ratio)
        T_test = T - T_train

        y_train = y[:, :T_train].copy()
        y_test = y[:, T_train:].copy()
        S_train = S_true[:, :T_train].copy()
        S_test = S_true[:, T_train:].copy()

        mask_train = mask[:, :T_train].copy()
        mask_test = mask[:, T_train:].copy()

        R_train, F_train, M_train = compute_rfm_features(y_train, mask_train)
        R_test, F_test, M_test = compute_rfm_features(y_test, mask_test)

        rfm = standardize_rfm(R_train, F_train, M_train, mask_train, R_test, F_test, M_test, mask_test)

        data = {
            'N': N,
            'T': T_train,
            'T_total': T,
            'T_test': T_test,
            'train_ratio': train_ratio,
            'y': y_train,
            'z': z[:, :T_train].copy(),
            'mask': mask_train,
            'R': rfm['R'],
            'F': rfm['F'],
            'M': rfm['M'],
            'true_states': S_train,
            'theta_true': theta,
            'segments_true': segments,
            'world': world,
            'seed': seed_dgp,
            'y_test': y_test,
            'z_test': z[:, T_train:].copy(),
            'mask_test': mask_test,
            'R_test': rfm.get('R_test'),
            'F_test': rfm.get('F_test'),
            'M_test': rfm.get('M_test'),
            'true_states_test': S_test,
            'rfm_stats': {k: v for k, v in rfm.items() if k not in ['R', 'F', 'M', 'R_test', 'F_test', 'M_test']},
        }

    else:
        R, F, M = compute_rfm_features(y, mask)
        rfm = standardize_rfm(R, F, M, mask)

        data = {
            'N': N,
            'T': T,
            'y': y,
            'z': z,
            'mask': mask,
            'R': rfm['R'],
            'F': rfm['F'],
            'M': rfm['M'],
            'true_states': S_true,
            'theta_true': theta,
            'segments_true': segments,
            'world': world,
            'seed': seed_dgp,
            'rfm_stats': {k: v for k, v in rfm.items() if k not in ['R', 'F', 'M']},
        }

    print("DGP loaded: {}, N={}, T={}, world={}, sparsity={:.1%}".format(
        npz_path.name, N, data['T'], world, (data['y'] > 0).mean()))

    return data


def load_dgp_csv(csv_path, train_ratio=1.0, seed=42):
    """
    Load DGP .csv file (long format) and convert to model-ready dict.
    CSV must have columns: customer_id, t, y, z, true_state, segment, theta
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError("CSV not found: {}".format(csv_path))

    df = pd.read_csv(csv_path)

    required = ['customer_id', 't', 'y', 'true_state']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("CSV missing columns: {}. Found: {}".format(missing, list(df.columns)))

    N = df['customer_id'].nunique()
    T = df['t'].nunique()

    y = df.pivot(index='customer_id', columns='t', values='y').values.astype(np.float32)
    S_true = df.pivot(index='customer_id', columns='t', values='true_state').values.astype(np.int32)

    theta = None
    if 'theta' in df.columns:
        theta = df.groupby('customer_id')['theta'].first().values.astype(np.float32)

    segments = None
    if 'segment' in df.columns:
        segments = df.groupby('customer_id')['segment'].first().values.astype(np.int32)

    mask = np.ones((N, T), dtype=bool)

    if train_ratio < 1.0:
        T_train = int(T * train_ratio)
        y_train = y[:, :T_train].copy()
        S_train = S_true[:, :T_train].copy()
        mask_train = mask[:, :T_train].copy()

        R_train, F_train, M_train = compute_rfm_features(y_train, mask_train)
        rfm = standardize_rfm(R_train, F_train, M_train, mask_train)

        data = {
            'N': N, 'T': T_train, 'T_total': T,
            'y': y_train, 'mask': mask_train,
            'R': rfm['R'], 'F': rfm['F'], 'M': rfm['M'],
            'true_states': S_train,
            'theta_true': theta,
            'segments_true': segments,
            'world': 'csv_loaded',
        }
    else:
        R, F, M = compute_rfm_features(y, mask)
        rfm = standardize_rfm(R, F, M, mask)

        data = {
            'N': N, 'T': T,
            'y': y, 'mask': mask,
            'R': rfm['R'], 'F': rfm['F'], 'M': rfm['M'],
            'true_states': S_true,
            'theta_true': theta,
            'segments_true': segments,
            'world': 'csv_loaded',
        }

    print("CSV loaded: {}, N={}, T={}, sparsity={:.1%}".format(
        csv_path.name, N, data['T'], (data['y'] > 0).mean()))

    return data


def test_loader():
    """Quick smoke test on a DGP file."""
    import tempfile
    import os

    N, T = 50, 20
    y = np.random.poisson(1, (N, T)).astype(np.float32)
    S_true = np.random.randint(0, 3, (N, T)).astype(np.int32)
    theta = np.random.normal(0, 1, N).astype(np.float32)
    segments = np.random.randint(0, 3, N).astype(np.int32)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_dgp.npz")
        np.savez(path, y=y, z=np.zeros_like(y), S_true=S_true,
                 theta=theta, segments=segments, N=N, T=T,
                 world="test", seed=42)

        data = load_dgp_npz(path, train_ratio=1.0)
        assert data['N'] == N
        assert data['T'] == T
        assert data['y'].shape == (N, T)
        assert data['R'].shape == (N, T)
        print("Loader test PASSED")

        data_split = load_dgp_npz(path, train_ratio=0.8)
        assert data_split['T'] == 16
        assert 'y_test' in data_split
        print("Train/test split test PASSED")


if __name__ == "__main__":
    test_loader()
