
"""
State Recovery: Viterbi or MAP decode + ARI computation.
Model-agnostic: works with any HMM idata that has alpha_filtered.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent to path for viterbi_decoder import
sys.path.insert(0, str(Path(__file__).parent))
from viterbi_decoder import viterbi_decode, decode_states_map

from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment


def compute_state_recovery(idata, data_dict, decoder='viterbi'):
    """
    Compute state recovery metrics.
    
    Parameters
    ----------
    idata : InferenceData
        Must contain 'alpha_filtered' (chains, draws, N, T, K)
    data_dict : dict
        Must contain 'S_true' (N, T) and optionally 'K'
    decoder : str
        'viterbi' (default) or 'map'
    
    Returns
    -------
    results : dict
        {'ari': float, 'pred_states': ndarray, 'state_accuracy': float, ...}
    """
    S_true = data_dict['S_true']
    N, T = S_true.shape
    K = data_dict.get('K', len(np.unique(S_true)))
    
    # Extract alpha_filtered
    if 'alpha_filtered' not in idata.posterior:
        return {'ari': np.nan, 'error': 'alpha_filtered not found'}
    
    alpha = idata.posterior['alpha_filtered'].mean(dim=['chain', 'draw']).values  # (N, T, K)
    
    # Extract Gamma (for Viterbi)
    Gamma_post = idata.posterior['Gamma'].values
    Gamma = Gamma_post.mean(axis=(0, 1))  # (K, K)
    
    # Decode
    if decoder == 'viterbi':
        pred_states = viterbi_decode(alpha, Gamma)
    else:
        pred_states = decode_states_map(alpha)
    
    # ARI
    ari = compute_ari(S_true, pred_states)
    
    # State-wise accuracy (Hungarian optimal matching)
    confusion = np.zeros((K, K))
    for true_s in range(K):
        for pred_s in range(K):
            confusion[true_s, pred_s] = np.sum((S_true == true_s) & (pred_states == pred_s))
    
    row_ind, col_ind = linear_sum_assignment(-confusion)
    optimal_acc = confusion[row_ind, col_ind].sum() / (N * T)
    
    return {
        'ari': float(ari),
        'pred_states': pred_states,
        'state_accuracy': float(optimal_acc),
        'confusion_matrix': confusion.tolist(),
        'decoder': decoder,
    }


def compute_ari(true_states, pred_states):
    """Compute ARI, handling missing values."""
    if true_states is None or pred_states is None:
        return np.nan
    
    true_flat = true_states.flatten()
    pred_flat = pred_states.flatten()
    
    mask = (true_flat >= 0) & (pred_flat >= 0)
    if mask.sum() < 10:
        return np.nan
    
    return adjusted_rand_score(true_flat[mask], pred_flat[mask])


def test_state_recovery():
    """Deterministic test with perfect state separation."""
    N, T, K = 30, 10, 3
    
    # Deterministic: 10 customers per state, all periods same state
    S_true = np.zeros((N, T), dtype=int)
    for i in range(N):
        S_true[i, :] = i // 10
    
    # Perfect alpha_filtered: 1.0 on true state, 0.0 on others
    alpha = np.zeros((1, 1, N, T, K))
    for i in range(N):
        for t in range(T):
            alpha[0, 0, i, t, S_true[i, t]] = 1.0
    
    # Fake idata with xarray-like posterior
    class FakeVar:
        def __init__(self, values):
            self.values = values
        def mean(self, dim):
            arr = self.values
            for d in dim:
                if d == 'chain':
                    arr = arr.mean(axis=0)
                elif d == 'draw':
                    arr = arr.mean(axis=0)
            return FakeVar(arr)
    
    class FakePosterior:
        def __init__(self):
            self._data = {
                'alpha_filtered': FakeVar(alpha),
                'Gamma': FakeVar(np.eye(K)[None, None, ...] * 0.9 + 0.05)
            }
        def __contains__(self, key):
            return key in self._data
        def __getitem__(self, key):
            return self._data[key]
    
    class FakeIdata:
        def __init__(self):
            self.posterior = FakePosterior()
    
    fake_idata = FakeIdata()
    data_dict = {'S_true': S_true, 'K': K}
    
    # Test both decoders
    results_viterbi = compute_state_recovery(fake_idata, data_dict, decoder='viterbi')
    results_map = compute_state_recovery(fake_idata, data_dict, decoder='map')
    
    print("Viterbi ARI:", results_viterbi['ari'])
    print("MAP ARI:", results_map['ari'])
    
    assert results_viterbi['ari'] > 0.99, "Viterbi failed on perfect data"
    assert results_map['ari'] > 0.99, "MAP failed on perfect data"
    print("State recovery test PASSED")


if __name__ == "__main__":
    test_state_recovery()
