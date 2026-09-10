"""
Unit tests for the SpectralEngine, Ky Fan optimality, and projector properties.
"""

import pytest
import numpy as np
import scipy.linalg

from src.config import BachelierConfig, VasicekConfig, HestonConfig
from src.spectral import SpectralEngine, SpectralDecomposition


def test_projector_properties():
    dim = 5
    d_prime = 2
    # Create random symmetric positive definite matrix A
    rng = np.random.default_rng(123)
    M = rng.standard_normal((dim, dim))
    A = M @ M.T + np.eye(dim)

    decomp = SpectralEngine.decompose(A, d_prime=d_prime)
    P = decomp.projector
    P_perp = decomp.projector_perp

    # 1. Symmetry: P = P^T
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    # 2. Idempotence: P^2 = P
    np.testing.assert_allclose(P @ P, P, atol=1e-12)
    # 3. Complement: P + P_perp = I
    np.testing.assert_allclose(P + P_perp, np.eye(dim), atol=1e-12)
    # 4. Orthogonality: P @ P_perp = 0
    np.testing.assert_allclose(P @ P_perp, np.zeros((dim, dim)), atol=1e-12)
    # 5. Rank: rank(P) = d'
    rank_P = np.linalg.matrix_rank(P)
    assert rank_P == d_prime


def test_ky_fan_optimality():
    """
    Verify Section 7.3: The top-d' eigenspace of A minimizes
    tr((I - Q) A) over all rank-d' orthogonal projectors Q.
    """
    dim = 6
    d_prime = 3
    rng = np.random.default_rng(456)
    M = rng.standard_normal((dim, dim))
    A = M @ M.T + 0.1 * np.eye(dim)

    decomp = SpectralEngine.decompose(A, d_prime=d_prime)
    P_opt = decomp.projector
    optimal_discarded = np.trace((np.eye(dim) - P_opt) @ A)

    # Must equal exact eigenvalue tail
    np.testing.assert_allclose(optimal_discarded, decomp.discarded_tail, atol=1e-10)

    # Test 50 random rank-d' orthogonal projectors
    for _ in range(50):
        V = rng.standard_normal((dim, d_prime))
        Q_sub, _ = np.linalg.qr(V)
        Q_proj = Q_sub @ Q_sub.T
        random_discarded = np.trace((np.eye(dim) - Q_proj) @ A)
        
        # Random projector must discard AT LEAST as much energy as optimal projector
        assert random_discarded >= optimal_discarded - 1e-10


def test_heston_covariance_symmetry():
    dim = 4
    cfg = HestonConfig(
        dim=dim,
        s0=np.array([100.0, 105.0, 95.0, 102.0]),
        v0=np.array([0.04, 0.05, 0.03, 0.04]),
        kappa=np.array([1.5, 2.0, 1.2, 1.8]),
        theta=np.array([0.04, 0.04, 0.04, 0.04]),
        xi=np.array([0.3, 0.35, 0.25, 0.3]),
        rho=np.array([-0.7, -0.6, -0.75, -0.65]),
        asset_corr=np.array([
            [1.0, 0.5, 0.3, 0.4],
            [0.5, 1.0, 0.4, 0.35],
            [0.3, 0.4, 1.0, 0.5],
            [0.4, 0.35, 0.5, 1.0]
        ])
    )
    A_bar = SpectralEngine.compute_covariance_heston(cfg, T=1.5)
    np.testing.assert_allclose(A_bar, A_bar.T, atol=1e-14)
    eigvals = scipy.linalg.eigvalsh(A_bar)
    assert np.all(eigvals >= -1e-12), "A_bar must be positive semi-definite"


def test_vasicek_zero_reversion_limit():
    """Section 13.5: Vasicek tail recovers Section 4 as gamma -> 0."""
    dim = 3
    d_prime = 1
    sigma = np.diag([0.2, 0.15, 0.1])
    cfg = VasicekConfig(
        dim=dim,
        s0=np.ones(dim) * 100.0,
        theta=np.ones(dim) * 100.0,
        gamma=np.ones(dim) * 1e-8,  # Near zero gamma
        sigma=sigma
    )
    A = SpectralEngine.compute_covariance_vasicek(cfg)
    T = 2.0
    decomp = SpectralEngine.decompose(A, d_prime=d_prime, gamma=cfg.gamma, T=T)

    # In limit gamma -> 0, tail_vasicek -> T * discarded_tail
    np.testing.assert_allclose(decomp.discarded_tail_vasicek, decomp.discarded_tail * T, rtol=1e-4)


def test_vasicek_general_matrix_quadrature():
    """
    Verify that compute_exact_path_gap_vasicek_general matches the analytical
    mode-by-mode formula when Theta and Sigma Sigma^T share the eigenbasis.
    """
    dim = 4
    d_prime = 2
    gamma = np.array([0.8, 1.5, 2.0, 3.0])
    Theta = np.diag(gamma)
    sigma = np.diag([2.0, 1.5, 0.8, 0.4])
    A = sigma @ sigma.T
    T = 1.2

    decomp = SpectralEngine.decompose(A, d_prime=d_prime, gamma=gamma, T=T)
    analytical_tail = decomp.discarded_tail_vasicek

    quad_tail = SpectralEngine.compute_exact_path_gap_vasicek_general(
        Theta=Theta,
        sigma=sigma,
        projector=decomp.projector,
        T=T,
        num_quad_steps=150
    )

    np.testing.assert_allclose(quad_tail, analytical_tail, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
