"""
Unit tests for synchronous coupling and control variate variance reduction.
Verifies Lemma 3 exact identity and theoretical bounds from the paper.
"""

import pytest
import numpy as np

from src.config import SimulationConfig, BachelierConfig, VasicekConfig
from src.payoffs import EuropeanBasketCall, OutOfSubspaceSpread
from src.spectral import SpectralEngine
from src.simulators import BachelierSimulator, VasicekSimulator
from src.control_variates import ControlVariateEngine


def test_lemma3_exact_identity():
    r"""
    Verify Lemma 3 (Paper Section 4.3):
    In centered constant-coefficient Bachelier model,
    E[||S_T - \hat{S}_T||^2] == T * sum_{j > d'} lambda_j
    """
    dim = 5
    d_prime = 2
    # Create non-trivial covariance matrix
    rng = np.random.default_rng(999)
    M = rng.standard_normal((dim, dim))
    sigma = M @ np.diag([3.0, 2.0, 1.0, 0.5, 0.2])
    
    # Initial state in range of P, zero drift
    cfg = BachelierConfig(
        dim=dim,
        s0=np.zeros(dim),
        drift=np.zeros(dim),
        sigma=sigma
    )
    sim_cfg = SimulationConfig(n_paths=250_000, T=1.5, seed=101)

    A = SpectralEngine.compute_covariance_bachelier(cfg)
    decomp = SpectralEngine.decompose(A, d_prime=d_prime)

    sim = BachelierSimulator(cfg)
    S_T, S_hat_T = sim.simulate_coupled(decomp.projector, sim_cfg)

    # Empirical mean squared gap
    diff = S_T - S_hat_T
    empirical_gap = np.mean(np.sum(diff ** 2, axis=1))

    # Exact theoretical identity: T * sum_{j > d'} lambda_j
    theoretical_gap = sim_cfg.T * decomp.discarded_tail

    # Monte Carlo standard error of the mean squared gap
    gap_samples = np.sum(diff ** 2, axis=1)
    se_gap = np.std(gap_samples) / np.sqrt(sim_cfg.n_paths)

    # Must match within 3.5 SE
    np.testing.assert_allclose(empirical_gap, theoretical_gap, atol=3.5 * se_gap)


def test_control_variate_variance_reduction():
    """
    Verify Theorem B (Paper Section 3):
    Optimal beta* minimizes variance and achieves VRF = 1 / (1 - rho^2) > 1.
    """
    dim = 4
    d_prime = 2
    sigma = np.array([
        [2.0, 0.8, 0.5, 0.2],
        [0.8, 1.8, 0.4, 0.3],
        [0.5, 0.4, 1.5, 0.2],
        [0.2, 0.3, 0.2, 1.2]
    ])
    cfg = BachelierConfig(
        dim=dim,
        s0=np.ones(dim) * 100.0,
        drift=np.ones(dim) * 2.0,
        sigma=sigma
    )
    sim_cfg = SimulationConfig(n_paths=100_000, T=1.0, r=0.03, seed=555)

    A = SpectralEngine.compute_covariance_bachelier(cfg)
    decomp = SpectralEngine.decompose(A, d_prime=d_prime)

    # Payoff aligned with top eigenmode
    weights = decomp.eigenvectors[:, 0]
    weights = np.abs(weights) / np.sum(np.abs(weights))
    payoff = EuropeanBasketCall(strike=100.0, weights=weights)

    sim = BachelierSimulator(cfg)
    S_T, S_hat_T = sim.simulate_coupled(decomp.projector, sim_cfg)

    # Discount
    discount = np.exp(-sim_cfg.r * sim_cfg.T)
    var_pi = float(np.var(discount * payoff(S_T), ddof=1))
    bound = SpectralEngine.theoretical_upper_bound_one_minus_rho2(
        decomp,
        L_f=payoff.lipschitz_constant,
        var_pi=var_pi,
        r=sim_cfg.r,
        T=sim_cfg.T,
        model_type="bachelier"
    )

    result = ControlVariateEngine.evaluate(
        payoff=payoff,
        S_T=S_T,
        S_hat_T=S_hat_T,
        discount=discount,
        spectral_bound_one_minus_rho2=bound,
        d_prime=d_prime
    )

    # Standard error of CV estimator must be strictly less than plain MC
    assert result.std_err_cv < result.std_err_mc
    # Empirical VRF must be strictly greater than 1
    assert result.vrf_empirical > 1.0
    # Correlation must be positive and substantial
    assert result.rho > 0.85
    # Empirical 1 - rho^2 must be within the theoretical upper bound
    assert result.empirical_one_minus_rho2 <= result.upper_bound_spectral + 1e-6


def test_vrf_monotonicity_in_dimension():
    r"""
    As retained dimension d' increases (retaining more eigenvalues),
    correlation rho should increase, and VRF should strictly increase.
    Follows Section 4's centered setup (s0 in range(P), zero drift).
    """
    dim = 4
    sigma = np.diag([3.0, 1.5, 0.8, 0.4])
    cfg = BachelierConfig(
        dim=dim,
        s0=np.zeros(dim),
        drift=np.zeros(dim),
        sigma=sigma
    )
    sim_cfg = SimulationConfig(n_paths=60_000, T=1.0, r=0.0, seed=777)
    A = SpectralEngine.compute_covariance_bachelier(cfg)

    weights = np.ones(dim) / dim
    payoff = EuropeanBasketCall(strike=0.0, weights=weights)
    sim = BachelierSimulator(cfg)

    vrfs = []
    for d_p in [1, 2, 3]:
        decomp = SpectralEngine.decompose(A, d_prime=d_p)
        S_T, S_hat_T = sim.simulate_coupled(decomp.projector, sim_cfg)
        res = ControlVariateEngine.evaluate(
            payoff=payoff,
            S_T=S_T,
            S_hat_T=S_hat_T,
            discount=1.0,
            spectral_bound_one_minus_rho2=1.0,
            d_prime=d_p
        )
        vrfs.append(res.vrf_empirical)

    # VRF(d'=1) < VRF(d'=2) < VRF(d'=3)
    assert vrfs[0] < vrfs[1] < vrfs[2]


if __name__ == "__main__":
    pytest.main([__file__])
