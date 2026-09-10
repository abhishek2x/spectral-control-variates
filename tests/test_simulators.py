"""
Unit tests for SDE simulation engines: Bachelier, Vasicek, and Numba Heston.
"""

import pytest
import numpy as np

from src.config import SimulationConfig, BachelierConfig, VasicekConfig, HestonConfig
from src.spectral import SpectralEngine
from src.simulators import BachelierSimulator, VasicekSimulator, HestonSimulator


def test_bachelier_moments():
    dim = 3
    s0 = np.array([100.0, 105.0, 95.0])
    drift = np.array([1.0, -0.5, 0.5])
    sigma = np.array([
        [2.0, 0.5, 0.1],
        [0.5, 1.8, 0.3],
        [0.1, 0.3, 1.5]
    ])
    cfg = BachelierConfig(dim=dim, s0=s0, drift=drift, sigma=sigma)
    sim_cfg = SimulationConfig(n_paths=200_000, T=1.0, seed=42)

    A = SpectralEngine.compute_covariance_bachelier(cfg)
    decomp = SpectralEngine.decompose(A, d_prime=2)

    sim = BachelierSimulator(cfg)
    S_T, S_hat_T = sim.simulate_coupled(decomp.projector, sim_cfg)

    # Expected mean: s0 + drift * T
    expected_mean = s0 + drift * sim_cfg.T
    sample_mean = np.mean(S_T, axis=0)
    std_err = np.std(S_T, axis=0) / np.sqrt(sim_cfg.n_paths)
    np.testing.assert_allclose(sample_mean, expected_mean, atol=3.5 * np.max(std_err))

    # Expected covariance: T * sigma @ sigma^T
    expected_cov = sim_cfg.T * (sigma @ sigma.T)
    sample_cov = np.cov(S_T, rowvar=False)
    np.testing.assert_allclose(sample_cov, expected_cov, atol=0.08)

    # Check that S_hat_T is exactly P @ S_T
    expected_S_hat = S_T @ decomp.projector.T
    np.testing.assert_allclose(S_hat_T, expected_S_hat, atol=1e-12)


def test_vasicek_simulation():
    dim = 2
    cfg = VasicekConfig(
        dim=dim,
        s0=np.array([100.0, 100.0]),
        theta=np.array([100.0, 100.0]),
        gamma=np.array([2.0, 1.5]),
        sigma=np.array([[2.0, 0.5], [0.5, 1.5]])
    )
    sim_cfg = SimulationConfig(n_paths=10_000, n_steps=50, T=0.5, seed=123)
    A = SpectralEngine.compute_covariance_vasicek(cfg)
    decomp = SpectralEngine.decompose(A, d_prime=1)

    sim = VasicekSimulator(cfg)
    S_T, S_hat_T = sim.simulate_coupled(decomp.projector, sim_cfg)

    assert S_T.shape == (sim_cfg.n_paths, dim)
    assert S_hat_T.shape == (sim_cfg.n_paths, dim)
    assert not np.any(np.isnan(S_T))
    assert not np.any(np.isnan(S_hat_T))


def test_heston_numba_martingality():
    dim = 2
    s0 = np.array([100.0, 100.0])
    cfg = HestonConfig(
        dim=dim,
        s0=s0,
        v0=np.array([0.04, 0.04]),
        kappa=np.array([2.0, 2.0]),
        theta=np.array([0.04, 0.04]),
        xi=np.array([0.3, 0.3]),
        rho=np.array([-0.7, -0.7]),
        asset_corr=np.array([[1.0, 0.5], [0.5, 1.0]])
    )
    sim_cfg = SimulationConfig(n_paths=50_000, n_steps=60, T=0.5, r=0.03, seed=789)
    A_bar = SpectralEngine.compute_covariance_heston(cfg, T=sim_cfg.T)
    decomp = SpectralEngine.decompose(A_bar, d_prime=1)

    sim = HestonSimulator(cfg)
    S_T, S_hat_T = sim.simulate_coupled(decomp.projector, sim_cfg)

    # Martingality test: E[exp(-rT) S_T] should be close to s0
    discount = np.exp(-sim_cfg.r * sim_cfg.T)
    discounted_S_T = discount * S_T
    mean_discounted = np.mean(discounted_S_T, axis=0)
    std_err = np.std(discounted_S_T, axis=0) / np.sqrt(sim_cfg.n_paths)

    # Check within 3 standard errors
    np.testing.assert_allclose(mean_discounted, s0, atol=3.0 * np.max(std_err))
    assert np.all(S_T > 0), "Stock prices under Heston should remain strictly positive"


def test_numba_numpy_statistical_parity():
    """Verify statistical parity between pure NumPy and Numba parallel implementations."""
    dim = 2
    cfg_b = BachelierConfig(
        dim=dim,
        s0=np.array([100.0, 100.0]),
        drift=np.array([1.0, -1.0]),
        sigma=np.array([[2.0, 0.5], [0.5, 1.5]])
    )
    sim_cfg = SimulationConfig(n_paths=50_000, n_steps=40, T=0.5, seed=42)
    P = np.array([[1.0, 0.0], [0.0, 0.0]])

    # Bachelier parity
    sim_b_np = BachelierSimulator(cfg_b, use_numba=False)
    sim_b_nb = BachelierSimulator(cfg_b, use_numba=True)
    S_np, _ = sim_b_np.simulate_coupled(P, sim_cfg)
    S_nb, _ = sim_b_nb.simulate_coupled(P, sim_cfg)
    np.testing.assert_allclose(np.mean(S_np, axis=0), np.mean(S_nb, axis=0), atol=0.08)

    # Vasicek parity
    cfg_v = VasicekConfig(
        dim=dim,
        s0=np.array([100.0, 100.0]),
        theta=np.array([100.0, 100.0]),
        gamma=np.array([2.0, 1.5]),
        sigma=np.array([[2.0, 0.5], [0.5, 1.5]])
    )
    sim_v_np = VasicekSimulator(cfg_v, use_numba=False)
    sim_v_nb = VasicekSimulator(cfg_v, use_numba=True)
    Sv_np, _ = sim_v_np.simulate_coupled(P, sim_cfg)
    Sv_nb, _ = sim_v_nb.simulate_coupled(P, sim_cfg)
    np.testing.assert_allclose(np.mean(Sv_np, axis=0), np.mean(Sv_nb, axis=0), atol=0.1)

    # Heston parity
    cfg_h = HestonConfig(
        dim=dim,
        s0=np.array([100.0, 100.0]),
        v0=np.array([0.04, 0.04]),
        kappa=np.array([2.0, 2.0]),
        theta=np.array([0.04, 0.04]),
        xi=np.array([0.3, 0.3]),
        rho=np.array([-0.7, -0.7]),
        asset_corr=np.array([[1.0, 0.5], [0.5, 1.0]])
    )
    sim_h_np = HestonSimulator(cfg_h, use_numba=False)
    sim_h_nb = HestonSimulator(cfg_h, use_numba=True)
    Sh_np, _ = sim_h_np.simulate_coupled(P, sim_cfg)
    Sh_nb, _ = sim_h_nb.simulate_coupled(P, sim_cfg)
    np.testing.assert_allclose(np.mean(Sh_np, axis=0), np.mean(Sh_nb, axis=0), atol=0.25)


if __name__ == "__main__":
    pytest.main([__file__])
