"""
Payoff classes for option pricing with explicit Lipschitz constants
and gradient directions for active subspace evaluation.
"""

from abc import ABC, abstractmethod
import numpy as np


class BasePayoff(ABC):
    """Abstract base class for multi-asset option payoffs."""

    @abstractmethod
    def __call__(self, S_T: np.ndarray) -> np.ndarray:
        """
        Evaluate discounted or undiscounted payoff on terminal state batch.
        
        Parameters
        ----------
        S_T : np.ndarray
            Terminal asset states, shape (n_paths, dim).
            
        Returns
        -------
        np.ndarray
            Payoff per path, shape (n_paths,).
        """
        pass

    @property
    @abstractmethod
    def lipschitz_constant(self) -> float:
        """
        Upper bound on the Lipschitz constant L_f in Euclidean norm:
        |f(x) - f(y)| <= L_f ||x - y||_2.
        """
        pass

    @property
    @abstractmethod
    def primary_subspace_weights(self) -> np.ndarray:
        """
        Primary sensitivity direction or gradient expectation vector
        used for active-subspace alignment (Section 16).
        """
        pass


class EuropeanBasketCall(BasePayoff):
    """
    Standard European Basket Call option:
    f(S_T) = max(w^T S_T - K, 0)
    """

    def __init__(self, strike: float, weights: np.ndarray):
        self.strike = float(strike)
        self.weights = np.asarray(weights, dtype=np.float64)
        assert self.weights.ndim == 1, "Weights must be a 1D vector"
        self._norm = float(np.linalg.norm(self.weights))

    def __call__(self, S_T: np.ndarray) -> np.ndarray:
        basket_val = S_T @ self.weights
        return np.maximum(basket_val - self.strike, 0.0)

    @property
    def lipschitz_constant(self) -> float:
        # Since |max(w^T x - K, 0) - max(w^T y - K, 0)| <= |w^T (x - y)| <= ||w||_2 * ||x - y||_2
        return self._norm

    @property
    def primary_subspace_weights(self) -> np.ndarray:
        return self.weights / self._norm if self._norm > 0 else self.weights


class EuropeanBasketPut(BasePayoff):
    """
    Standard European Basket Put option:
    f(S_T) = max(K - w^T S_T, 0)
    """

    def __init__(self, strike: float, weights: np.ndarray):
        self.strike = float(strike)
        self.weights = np.asarray(weights, dtype=np.float64)
        assert self.weights.ndim == 1, "Weights must be a 1D vector"
        self._norm = float(np.linalg.norm(self.weights))

    def __call__(self, S_T: np.ndarray) -> np.ndarray:
        basket_val = S_T @ self.weights
        return np.maximum(self.strike - basket_val, 0.0)

    @property
    def lipschitz_constant(self) -> float:
        return self._norm

    @property
    def primary_subspace_weights(self) -> np.ndarray:
        return self.weights / self._norm if self._norm > 0 else self.weights


class OutOfSubspaceSpread(BasePayoff):
    """
    Spread option whose sensitivity vector lies in the discarded subspace P_perp:
    f(S_T) = max(u^T S_T - K, 0)
    
    Used to validate Section 16 (Active Subspace vs plain PCA) where plain PCA fails
    because payoff sensitivity is concentrated in discarded eigenmodes.
    """

    def __init__(self, strike: float, direction_vector: np.ndarray):
        self.strike = float(strike)
        self.direction = np.asarray(direction_vector, dtype=np.float64)
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction / norm
        self._norm = float(np.linalg.norm(self.direction))

    def __call__(self, S_T: np.ndarray) -> np.ndarray:
        proj = S_T @ self.direction
        return np.maximum(proj - self.strike, 0.0)

    @property
    def lipschitz_constant(self) -> float:
        return self._norm

    @property
    def primary_subspace_weights(self) -> np.ndarray:
        return self.direction


class RainbowWorstOf(BasePayoff):
    """
    Worst-of (Rainbow) Call option on basket:
    f(S_T) = max(min(S_T^1, ..., S_T^d) - K, 0)
    
    Has strong nonlinear cross-asset sensitivity that tests the two-sided criterion
    and conditional-variance lower bound in Section 15.
    """

    def __init__(self, strike: float, dim: int):
        self.strike = float(strike)
        self.dim = dim

    def __call__(self, S_T: np.ndarray) -> np.ndarray:
        worst_asset = np.min(S_T, axis=1)
        return np.maximum(worst_asset - self.strike, 0.0)

    @property
    def lipschitz_constant(self) -> float:
        # |min(x) - min(y)| <= max_i |x_i - y_i| <= ||x - y||_2
        return 1.0

    @property
    def primary_subspace_weights(self) -> np.ndarray:
        # Uniform weighting as proxy
        w = np.ones(self.dim, dtype=np.float64) / np.sqrt(self.dim)
        return w
