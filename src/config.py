"""
Configuration and parameter dataclasses for the spectral reduced-variance simulation testbed.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Simulation run parameters."""
    n_paths: int = 100_000
    n_steps: int = 100
    T: float = 1.0
    r: float = 0.03
    seed: int = 42


@dataclass(frozen=True)
class BachelierConfig:
    """
    Multi-asset Bachelier / Constant-Coefficient Gaussian model:
    dS_t = b dt + Sigma dW_t
    """
    dim: int
    s0: np.ndarray       # Initial asset values, shape (dim,)
    drift: np.ndarray    # Constant drift b, shape (dim,)
    sigma: np.ndarray    # Constant diffusion matrix Sigma, shape (dim, dim)

    def __post_init__(self):
        assert self.s0.shape == (self.dim,), f"s0 shape {self.s0.shape} != ({self.dim},)"
        assert self.drift.shape == (self.dim,), f"drift shape {self.drift.shape} != ({self.dim},)"
        assert self.sigma.shape == (self.dim, self.dim), f"sigma shape {self.sigma.shape} != ({self.dim}, {self.dim})"


@dataclass(frozen=True)
class VasicekConfig:
    """
    Multi-factor Mean-Reverting Gaussian (Vasicek / Ornstein-Uhlenbeck) model:
    dS_t = -Theta (S_t - theta) dt + Sigma dW_t
    """
    dim: int
    s0: np.ndarray       # Initial values, shape (dim,)
    theta: np.ndarray    # Long-term mean, shape (dim,)
    gamma: np.ndarray    # Mean-reversion speeds (diagonal of Theta), shape (dim,)
    sigma: np.ndarray    # Constant diffusion matrix Sigma, shape (dim, dim)

    def __post_init__(self):
        assert self.s0.shape == (self.dim,)
        assert self.theta.shape == (self.dim,)
        assert self.gamma.shape == (self.dim,)
        assert self.sigma.shape == (self.dim, self.dim)
        assert np.all(self.gamma > 0), "All mean-reversion rates gamma must be positive"


@dataclass(frozen=True)
class HestonConfig:
    """
    Multi-factor Heston Stochastic Volatility Model:
    dS_t^i = r S_t^i dt + sqrt(v_t^i) S_t^i dW_t^{S, i}
    dv_t^i = kappa_i (theta_i - v_t^i) dt + xi_i sqrt(v_t^i) dW_t^{v, i}
    with corr(dW^{S, i}, dW^{v, i}) = rho_i, and cross-asset correlation matrix asset_corr.
    """
    dim: int
    s0: np.ndarray          # Initial asset prices, shape (dim,)
    v0: np.ndarray          # Initial variance, shape (dim,)
    kappa: np.ndarray       # CIR mean-reversion rate, shape (dim,)
    theta: np.ndarray       # CIR long-term variance, shape (dim,)
    xi: np.ndarray          # CIR vol-of-vol, shape (dim,)
    rho: np.ndarray         # Leverage correlation corr(W_S, W_v), shape (dim,)
    asset_corr: np.ndarray  # Cross-asset correlation matrix, shape (dim, dim)

    def __post_init__(self):
        assert self.s0.shape == (self.dim,)
        assert self.v0.shape == (self.dim,)
        assert self.kappa.shape == (self.dim,)
        assert self.theta.shape == (self.dim,)
        assert self.xi.shape == (self.dim,)
        assert self.rho.shape == (self.dim,)
        assert self.asset_corr.shape == (self.dim, self.dim)
        assert np.all(self.v0 > 0), "v0 must be strictly positive"
        assert np.all(self.kappa > 0), "kappa must be strictly positive"
        assert np.all(self.theta > 0), "theta must be strictly positive"
        assert np.all(self.xi > 0), "xi must be strictly positive"
