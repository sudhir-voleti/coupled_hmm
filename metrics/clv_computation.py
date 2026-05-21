"""
CLV Computation: First-principles perpetuity formula.
Model-agnostic: uses posterior emission parameters and state occupancy.
"""

import numpy as np


def compute_clv(idata, data_dict, discount_rate=0.001, margin=0.3, horizon=104):
    """
    Compute CLV from first principles.
    
    Formula: CLV = sum_t [margin * E[spend_t | state_t] * P(purchase_t | state_t) / (1+d)^t]
    
    Parameters
    ----------
    idata : InferenceData
    data_dict : dict
    discount_rate : float
        Weekly discount rate (default 0.1% = ~5% annual)
    margin : float
        Contribution margin (default 30%)
    horizon : int
        Projection horizon in weeks (default 104 = 2 years)
    
    Returns
    -------
    results : dict
    """
    K = data_dict.get('K', 3)
    N = data_dict.get('N', 100)
    
    # --- Extract posterior means ---
    
    # Spend parameters (log-normal)
    if 'beta_m' in idata.posterior and 'log_sigma_s' in idata.posterior:
        beta_m = idata.posterior['beta_m'].values.mean(axis=(0, 1))  # (K,)
        sigma_s = np.exp(idata.posterior['log_sigma_s'].values.mean(axis=(0, 1)))  # (K,)
        expected_spend = np.exp(beta_m + 0.5 * sigma_s**2)  # (K,)
    else:
        return {'error': 'No log-normal spend parameters'}
    
    # Timing parameters (NBD)
    if 'alpha_h' in idata.posterior and 'log_r' in idata.posterior:
        alpha_h = idata.posterior['alpha_h'].values.mean(axis=(0, 1))  # (K,)
        log_r = idata.posterior['log_r'].values.mean(axis=(0, 1))  # (K,)
        r_nbd = np.exp(log_r)
        lam = np.exp(alpha_h)
        
        # NBD zero probability: P(Y=0) = (r / (r + lambda))^r
        p_zero = (r_nbd / (r_nbd + lam)) ** r_nbd
        purchase_prob = 1.0 - p_zero
    else:
        return {'error': 'No NBD timing parameters'}
    
    # --- State-level CLV ---
    
    clv_by_state = []
    for k in range(K):
        # Per-period expected contribution
        period_value = margin * expected_spend[k] * purchase_prob[k]
        
        # Geometric series: sum_{t=0}^{H-1} v / (1+d)^t = v * (1 - (1+d)^{-H}) / (1 - 1/(1+d))
        if discount_rate > 1e-6:
            clv_k = period_value * (1 - (1 + discount_rate)**(-horizon)) / discount_rate
        else:
            clv_k = period_value * horizon
        
        clv_by_state.append(float(clv_k))
    
    clv_by_state = np.array(clv_by_state)
    clv_ratio = float(clv_by_state.max() / (clv_by_state.min() + 1e-6))
    
    # --- Customer-level CLV (state-occupancy weighted) ---
    
    if 'alpha_filtered' in idata.posterior:
        alpha = idata.posterior['alpha_filtered'].values.mean(axis=(0, 1))  # (N, T, K)
        state_probs = alpha.mean(axis=1)  # (N, K) - average over time
        clv_customer = state_probs @ clv_by_state  # (N,)
    else:
        # Uniform if no state probs
        clv_customer = np.ones(N) * clv_by_state.mean()
    
    return {
        'clv_by_state': clv_by_state.tolist(),
        'clv_ratio': clv_ratio,
        'clv_customer': clv_customer.tolist(),
        'clv_customer_mean': float(clv_customer.mean()),
        'clv_customer_std': float(clv_customer.std()),
        'expected_spend_by_state': expected_spend.tolist(),
        'purchase_prob_by_state': purchase_prob.tolist(),
        'discount_rate': discount_rate,
        'margin': margin,
        'horizon': horizon,
    }


def test_clv():
    """Test with known parameters."""
    K = 3
    
    # Known parameters
    beta_m = np.array([1.0, 2.0, 3.0])
    log_sigma_s = np.array([-0.5, 0.0, 0.5])
    alpha_h = np.array([0.0, 0.5, 1.0])
    log_r = np.array([0.0, 0.0, 0.0])
    
    class FakeVar:
        def __init__(self, values):
            self.values = values
        def mean(self, dim):
            return FakeVar(self.values)
    
    class FakePosterior:
        def __init__(self):
            self._data = {
                'beta_m': FakeVar(beta_m[None, None, :]),
                'log_sigma_s': FakeVar(log_sigma_s[None, None, :]),
                'alpha_h': FakeVar(alpha_h[None, None, :]),
                'log_r': FakeVar(log_r[None, None, :]),
            }
        def __contains__(self, key):
            return key in self._data
        def __getitem__(self, key):
            return self._data[key]
    
    class FakeIdata:
        def __init__(self):
            self.posterior = FakePosterior()
    
    fake_idata = FakeIdata()
    data_dict = {'K': K, 'N': 100}
    
    # Test with 0.1% weekly discount, 30% margin, 2-year horizon
    results = compute_clv(fake_idata, data_dict, discount_rate=0.001, margin=0.3, horizon=104)
    
    print("CLV by state:", [round(x, 1) for x in results['clv_by_state']])
    print("CLV ratio:", round(results['clv_ratio'], 1))
    print("Customer CLV mean:", round(results['clv_customer_mean'], 1))
    
    # State 2 should have highest CLV
    assert results['clv_by_state'][2] > results['clv_by_state'][0]
    assert results['clv_ratio'] > 2.0
    print("CLV test PASSED")


if __name__ == "__main__":
    test_clv()
