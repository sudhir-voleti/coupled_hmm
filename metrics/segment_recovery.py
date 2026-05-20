"""
Segment Recovery: k-means on theta posteriors vs true segments.
Model-agnostic: works with any model that has individual theta_i.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def compute_segment_recovery(idata, data_dict, n_segments=3, seed=42):
    """
    Cluster posterior theta_i means and compare to true segments.
    
    Parameters
    ----------
    idata : InferenceData
        Must contain 'theta' (chains, draws, N, 1)
    data_dict : dict
        Must contain 'segments_true' (N,)
    n_segments : int
        Number of segments to cluster
    seed : int
        Random seed for k-means
    
    Returns
    -------
    results : dict
        {'ari': float, 'pred_segments': ndarray, 'theta_means': ndarray, ...}
    """
    segments_true = data_dict.get('segments_true')
    if segments_true is None:
        return {'ari': np.nan, 'error': 'segments_true not found'}
    
    # Extract theta posterior means
    if 'theta' not in idata.posterior:
        return {'ari': np.nan, 'error': 'theta not found in posterior'}
    
    theta_post = idata.posterior['theta'].values  # (chains, draws, N, 1)
    theta_means = theta_post.mean(axis=(0, 1)).squeeze()  # (N,)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_segments, random_state=seed, n_init=10)
    pred_segments = kmeans.fit_predict(theta_means.reshape(-1, 1))
    
    # ARI
    ari = adjusted_rand_score(segments_true, pred_segments)
    
    # Segment means (for interpretation)
    seg_means = {}
    for s in range(n_segments):
        mask = (pred_segments == s)
        seg_means[int(s)] = float(theta_means[mask].mean()) if mask.sum() > 0 else np.nan
    
    return {
        'ari': float(ari),
        'pred_segments': pred_segments,
        'theta_means': theta_means,
        'segment_means_est': seg_means,
        'n_segments': n_segments,
    }


def test_segment_recovery():
    """Test with synthetic theta and segments."""
    N = 100
    n_seg = 3
    
    # True segments and theta
    segments_true = np.array([0]*35 + [1]*35 + [2]*30)
    theta_true = np.array(
        [np.random.normal(-1.0, 0.3) for _ in range(35)] +
        [np.random.normal(0.5, 0.3) for _ in range(35)] +
        [np.random.normal(2.0, 0.3) for _ in range(30)]
    )
    
    # Fake posterior (add noise)
    theta_post = theta_true[None, None, :, None] + np.random.normal(0, 0.1, (2, 500, N, 1))
    
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
            self._data = {'theta': FakeVar(theta_post)}
        def __contains__(self, key):
            return key in self._data
        def __getitem__(self, key):
            return self._data[key]
    
    class FakeIdata:
        def __init__(self):
            self.posterior = FakePosterior()
    
    fake_idata = FakeIdata()
    data_dict = {'segments_true': segments_true}
    
    results = compute_segment_recovery(fake_idata, data_dict, n_segments=n_seg)
    
    print("Test ARI:", results['ari'])
    print("Segment means:", results['segment_means_est'])
    
    assert results['ari'] > 0.7, "Segment recovery too low for well-separated data"
    print("Segment recovery test PASSED")


if __name__ == "__main__":
    test_segment_recovery()
