"""
Stochastic Differential Equation (SDE) simulation engines.
Includes vectorized NumPy implementations and Numba JIT-compiled kernels
for synchronous Brownian coupling between full and reduced state dynamics.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import numba
from numba import njit, prange

from .config import SimulationConfig, BachelierConfig, VasicekConfig, HestonConfig


class BaseSDESolver(ABC):
    """Abstract base class for coupled full/reduced SDE simulators."""

    @abstractmethod
    def simulate_coupled(
        self,
        projector: np.ndarray,
        sim_cfg: SimulationConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Simulate both full state S_T and reduced state \hat{S}_T synchronously
        driven by the identical Brownian paths.
        
        Parameters
        ----------
        projector : np.ndarray
            Orthogonal projector P onto top-d' subspace, shape (dim, dim).
        sim_cfg : SimulationConfig
            Simulation hyperparameters (n_paths, n_steps, T, r, seed).
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (S_T, S_hat_T), each of shape (n_paths, dim).
        """
        pass


@njit(parallel=True, fastmath=True)
def _simulate_bachelier_coupled_numba(
    s0: np.ndarray,
    drift: np.ndarray,
    sigma: np.ndarray,
    projector: np.ndarray,
    T: float,
    n_paths: int,
    dim: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Numba parallelized simulation of coupled Bachelier paths."""
    np.random.seed(seed)
    sqrt_T = np.sqrt(T)
    S_T = np.zeros((n_paths, dim), dtype=np.float64)
    S_hat_T = np.zeros((n_paths, dim), dtype=np.float64)

    for p in prange(n_paths):
        z = np.random.standard_normal(dim)
        for i in range(dim):
            diff = 0.0
            for j in range(dim):
                diff += sigma[i, j] * z[j]
            s_val = s0[i] + drift[i] * T + sqrt_T * diff
            S_T[p, i] = s_val

        for i in range(dim):
            proj_val = 0.0
            for j in range(dim):
                proj_val += projector[i, j] * S_T[p, j]
            S_hat_T[p, i] = proj_val

    return S_T, S_hat_T


@njit(parallel=True, fastmath=True)
def _simulate_vasicek_coupled_numba(
    s0: np.ndarray,
    theta: np.ndarray,
    gamma: np.ndarray,
    sigma: np.ndarray,
    projector: np.ndarray,
    T: float,
    n_paths: int,
    n_steps: int,
    dim: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Numba parallelized simulation of coupled Vasicek OU paths."""
    np.random.seed(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    # Precompute P @ sigma
    P_sigma = np.zeros((dim, dim), dtype=np.float64)
    for i in range(dim):
        for j in range(dim):
            val = 0.0
            for k in range(dim):
                val += projector[i, k] * sigma[k, j]
            P_sigma[i, j] = val

    S_T = np.zeros((n_paths, dim), dtype=np.float64)
    S_hat_T = np.zeros((n_paths, dim), dtype=np.float64)

    for p in prange(n_paths):
        S = s0.copy()
        S_hat = np.zeros(dim, dtype=np.float64)
        for i in range(dim):
            val = 0.0
            for j in range(dim):
                val += projector[i, j] * s0[j]
            S_hat[i] = val

        for step in range(n_steps):
            z = np.random.standard_normal(dim)

            # Full state update
            for i in range(dim):
                diff_s = 0.0
                for j in range(dim):
                    diff_s += sigma[i, j] * z[j]
                drift_s = -gamma[i] * (S[i] - theta[i]) * dt
                S[i] += drift_s + sqrt_dt * diff_s

            # Reduced state unprojected drift
            unproj_drift = np.zeros(dim, dtype=np.float64)
            for i in range(dim):
                unproj_drift[i] = -gamma[i] * (S_hat[i] - theta[i]) * dt

            # Projected drift and diffusion
            for i in range(dim):
                proj_drift = 0.0
                diff_hat = 0.0
                for j in range(dim):
                    proj_drift += projector[i, j] * unproj_drift[j]
                    diff_hat += P_sigma[i, j] * z[j]
                S_hat[i] += proj_drift + sqrt_dt * diff_hat

        for i in range(dim):
            S_T[p, i] = S[i]
            S_hat_T[p, i] = S_hat[i]

    return S_T, S_hat_T


class BachelierSimulator(BaseSDESolver):
    r"""
    Multi-Asset Bachelier / Constant-Coefficient Gaussian Model:
    dS_t = b dt + Sigma dW_t
    """

    def __init__(self, config: BachelierConfig, use_numba: bool = True):
        self.config = config
        self.use_numba = use_numba

    def simulate_coupled(
        self,
        projector: np.ndarray,
        sim_cfg: SimulationConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_numba:
            return _simulate_bachelier_coupled_numba(
                s0=self.config.s0,
                drift=self.config.drift,
                sigma=self.config.sigma,
                projector=projector,
                T=sim_cfg.T,
                n_paths=sim_cfg.n_paths,
                dim=self.config.dim,
                seed=sim_cfg.seed
            )

        rng = np.random.default_rng(sim_cfg.seed)
        dim = self.config.dim
        n_paths = sim_cfg.n_paths
        T = sim_cfg.T

        Z = rng.standard_normal(size=(n_paths, dim))
        sqrt_T = np.sqrt(T)

        diffusion_term = sqrt_T * (Z @ self.config.sigma.T)
        deterministic_term = self.config.s0 + self.config.drift * T

        S_T = deterministic_term + diffusion_term
        S_hat_T = S_T @ projector.T

        return S_T, S_hat_T


class VasicekSimulator(BaseSDESolver):
    r"""
    Multi-Factor Mean-Reverting Gaussian (Vasicek / OU) Model (Section 13):
    dS_t = -Theta (S_t - theta) dt + Sigma dW_t
    d\hat{S}_t = -P Theta (\hat{S}_t - theta) dt + P Sigma dW_t
    """

    def __init__(self, config: VasicekConfig, use_numba: bool = True):
        self.config = config
        self.use_numba = use_numba

    def simulate_coupled(
        self,
        projector: np.ndarray,
        sim_cfg: SimulationConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_numba:
            return _simulate_vasicek_coupled_numba(
                s0=self.config.s0,
                theta=self.config.theta,
                gamma=self.config.gamma,
                sigma=self.config.sigma,
                projector=projector,
                T=sim_cfg.T,
                n_paths=sim_cfg.n_paths,
                n_steps=sim_cfg.n_steps,
                dim=self.config.dim,
                seed=sim_cfg.seed
            )

        rng = np.random.default_rng(sim_cfg.seed)
        dim = self.config.dim
        n_paths = sim_cfg.n_paths
        n_steps = sim_cfg.n_steps
        T = sim_cfg.T
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        S = np.tile(self.config.s0, (n_paths, 1)).astype(np.float64)
        S_hat = (S @ projector.T).astype(np.float64)

        theta = self.config.theta
        gamma = self.config.gamma
        sigma = self.config.sigma

        P_sigma_T = (projector @ sigma).T
        sigma_T = sigma.T

        for _ in range(n_steps):
            Z = rng.standard_normal(size=(n_paths, dim))
            drift_S = -gamma * (S - theta) * dt
            diff_S = sqrt_dt * (Z @ sigma_T)
            S += drift_S + diff_S

            drift_unproj = -gamma * (S_hat - theta)
            drift_hat = (drift_unproj @ projector.T) * dt
            diff_hat = sqrt_dt * (Z @ P_sigma_T)
            S_hat += drift_hat + diff_hat

        return S, S_hat


# ---------------------------------------------------------------------------
# Numba JIT-accelerated multi-factor Heston kernel
# ---------------------------------------------------------------------------

@njit(parallel=True, fastmath=True)
def _simulate_heston_coupled_numba(
    s0: np.ndarray,
    v0: np.ndarray,
    kappa: np.ndarray,
    theta: np.ndarray,
    xi: np.ndarray,
    rho: np.ndarray,
    L_S: np.ndarray,
    projector: np.ndarray,
    r: float,
    T: float,
    n_paths: int,
    n_steps: int,
    dim: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba parallelized Euler-Maruyama simulation of coupled Heston SDEs
    with full truncation scheme for CIR variances (Lord et al., 2010).
    """
    np.random.seed(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    S_T = np.zeros((n_paths, dim), dtype=np.float64)
    S_hat_T = np.zeros((n_paths, dim), dtype=np.float64)

    # Precalculate sqrt(1 - rho^2)
    sqrt_one_minus_rho2 = np.sqrt(np.maximum(1.0 - rho * rho, 0.0))

    for p in prange(n_paths):
        # Local state arrays for this path
        S = s0.copy()
        # Initial reduced state is P @ s0
        S_hat = np.zeros(dim, dtype=np.float64)
        for i in range(dim):
            val = 0.0
            for j in range(dim):
                val += projector[i, j] * s0[j]
            S_hat[i] = val

        v = v0.copy()

        # Step through time
        for step in range(n_steps):
            # Generate independent standard normals
            z_s_raw = np.random.standard_normal(dim)
            z_orth = np.random.standard_normal(dim)

            # Correlated variance shocks: z_v[i] = rho[i]*z_s_raw[i] + sqrt(1 - rho[i]^2)*z_orth[i]
            z_v = np.zeros(dim, dtype=np.float64)
            for i in range(dim):
                z_v[i] = rho[i] * z_s_raw[i] + sqrt_one_minus_rho2[i] * z_orth[i]

            # Correlated asset shocks: z_s = L_S @ z_s_raw
            z_s = np.zeros(dim, dtype=np.float64)
            for i in range(dim):
                acc = 0.0
                for j in range(dim):
                    acc += L_S[i, j] * z_s_raw[j]
                z_s[i] = acc

            # Full truncation CIR variance update
            v_pos = np.zeros(dim, dtype=np.float64)
            for i in range(dim):
                v_pos[i] = max(v[i], 0.0)
                sqrt_v = np.sqrt(v_pos[i])
                v[i] += kappa[i] * (theta[i] - v_pos[i]) * dt + xi[i] * sqrt_v * sqrt_dt * z_v[i]

            # Full asset price update: dS = r S dt + sqrt(v_pos) * S * z_s * sqrt_dt
            diff_S = np.zeros(dim, dtype=np.float64)
            for i in range(dim):
                diff_S[i] = np.sqrt(v_pos[i]) * S[i] * z_s[i] * sqrt_dt
                S[i] += r * S[i] * dt + diff_S[i]

            # Reduced asset price update:
            # dS_hat = r * (P @ S_hat) dt + P @ (sqrt(v_pos) * S_hat * z_s * sqrt_dt)
            unproj_diff = np.zeros(dim, dtype=np.float64)
            unproj_drift = np.zeros(dim, dtype=np.float64)
            for i in range(dim):
                unproj_diff[i] = np.sqrt(v_pos[i]) * S_hat[i] * z_s[i] * sqrt_dt
                unproj_drift[i] = r * S_hat[i] * dt

            proj_diff = np.zeros(dim, dtype=np.float64)
            proj_drift = np.zeros(dim, dtype=np.float64)
            for i in range(dim):
                for j in range(dim):
                    proj_diff[i] += projector[i, j] * unproj_diff[j]
                    proj_drift[i] += projector[i, j] * unproj_drift[j]

            for i in range(dim):
                S_hat[i] += proj_drift[i] + proj_diff[i]

        for i in range(dim):
            S_T[p, i] = S[i]
            S_hat_T[p, i] = S_hat[i]

    return S_T, S_hat_T


def _simulate_heston_coupled_numpy(
    s0: np.ndarray,
    v0: np.ndarray,
    kappa: np.ndarray,
    theta: np.ndarray,
    xi: np.ndarray,
    rho: np.ndarray,
    L_S: np.ndarray,
    projector: np.ndarray,
    r: float,
    T: float,
    n_paths: int,
    n_steps: int,
    dim: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure vectorized NumPy Euler-Maruyama simulation of coupled Heston SDEs."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)

    S = np.tile(s0, (n_paths, 1)).astype(np.float64)
    S_hat = (S @ projector.T).astype(np.float64)
    v = np.tile(v0, (n_paths, 1)).astype(np.float64)
    sqrt_one_minus_rho2 = np.sqrt(np.maximum(1.0 - rho * rho, 0.0))

    for _ in range(n_steps):
        Z_S_raw = rng.standard_normal((n_paths, dim))
        Z_orth = rng.standard_normal((n_paths, dim))
        Z_v = rho * Z_S_raw + sqrt_one_minus_rho2 * Z_orth
        Z_S = Z_S_raw @ L_S.T

        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)

        # Full truncation CIR
        v += kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * Z_v

        # Full asset price
        diff_S = sqrt_v * S * Z_S * sqrt_dt
        S += r * S * dt + diff_S

        # Reduced asset price
        unproj_diff = sqrt_v * S_hat * Z_S * sqrt_dt
        unproj_drift = r * S_hat * dt
        S_hat += (unproj_drift @ projector.T) + (unproj_diff @ projector.T)

    return S, S_hat


class HestonSimulator(BaseSDESolver):
    """
    Multi-Factor Heston Stochastic Volatility Simulator.
    Supports both Numba JIT multi-threaded compilation and pure vectorized NumPy.
    """

    def __init__(self, config: HestonConfig, use_numba: bool = True):
        self.config = config
        self.use_numba = use_numba
        # Cholesky factor of asset correlation matrix
        self.L_S = np.linalg.cholesky(config.asset_corr)

    def simulate_coupled(
        self,
        projector: np.ndarray,
        sim_cfg: SimulationConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_numba:
            return _simulate_heston_coupled_numba(
                s0=self.config.s0,
                v0=self.config.v0,
                kappa=self.config.kappa,
                theta=self.config.theta,
                xi=self.config.xi,
                rho=self.config.rho,
                L_S=self.L_S,
                projector=projector,
                r=sim_cfg.r,
                T=sim_cfg.T,
                n_paths=sim_cfg.n_paths,
                n_steps=sim_cfg.n_steps,
                dim=self.config.dim,
                seed=sim_cfg.seed
            )

        return _simulate_heston_coupled_numpy(
            s0=self.config.s0,
            v0=self.config.v0,
            kappa=self.config.kappa,
            theta=self.config.theta,
            xi=self.config.xi,
            rho=self.config.rho,
            L_S=self.L_S,
            projector=projector,
            r=sim_cfg.r,
            T=sim_cfg.T,
            n_paths=sim_cfg.n_paths,
            n_steps=sim_cfg.n_steps,
            dim=self.config.dim,
            seed=sim_cfg.seed
        )
