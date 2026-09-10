"""
Spectral reduction engine: analytical covariance operator computation,
eigendecomposition, orthogonal projectors, and theoretical variance-reduction bounds.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import scipy.linalg

from .config import BachelierConfig, VasicekConfig, HestonConfig
from .payoffs import BasePayoff


@dataclass
class SpectralDecomposition:
    """Spectral decomposition results and orthogonal projectors."""
    eigenvalues: np.ndarray         # Ordered lambda_1 >= ... >= lambda_d >= 0
    eigenvectors: np.ndarray        # Orthonormal columns u_j, shape (d, d)
    retained_dim: int               # d'
    projector: np.ndarray           # P = sum_{j<=d'} u_j u_j^T, shape (d, d)
    projector_perp: np.ndarray      # P_perp = I - P, shape (d, d)
    discarded_tail: float           # sum_{j > d'} lambda_j
    discarded_tail_vasicek: float   # sum_{j > d'} lambda_j * (1 - exp(-2*gamma_j*T)) / (2*gamma_j)
    retained_energy_ratio: float    # sum_{j<=d'} lambda_j / sum_j lambda_j


class SpectralEngine:
    """Analytical computation and spectral reduction of diffusion operators."""

    @staticmethod
    def compute_covariance_bachelier(config: BachelierConfig) -> np.ndarray:
        """
        Analytical covariance matrix A for constant-coefficient Bachelier model:
        A = Sigma @ Sigma^T
        """
        return config.sigma @ config.sigma.T

    @staticmethod
    def compute_covariance_vasicek(config: VasicekConfig) -> np.ndarray:
        """
        Analytical covariance matrix A for constant-diffusion Vasicek model:
        A = Sigma @ Sigma^T
        """
        return config.sigma @ config.sigma.T

    @staticmethod
    def compute_covariance_heston(config: HestonConfig, T: float) -> np.ndarray:
        r"""
        Analytical time-and-path-averaged covariance operator \bar{A} for Heston model
        via the CIR first-moment cascade (Paper Section 7.2 & Section 11.1, 11.4):
        
        \bar{v}_i = \frac{1}{T} \int_0^T E[v_t^i] dt = theta_i + (v_{0, i} - theta_i) \frac{1 - e^{-\kappa_i T}}{\kappa_i T}
        \bar{A}_{ij} = \sqrt{\bar{v}_i \bar{v}_j} \cdot \Gamma_{ij}
        where \Gamma is the cross-asset correlation matrix.
        """
        # CIR first moment time-average
        kappa_T = config.kappa * T
        # Avoid division by zero if kappa_T is tiny
        ratio = np.where(kappa_T > 1e-7, (1.0 - np.exp(-kappa_T)) / kappa_T, 1.0 - 0.5 * kappa_T)
        v_bar = config.theta + (config.v0 - config.theta) * ratio  # shape (dim,)
        
        # Construct path/time-averaged covariance operator in asset price coordinates:
        # Sigma(S_t) = diag(S_t) diag(sqrt(v_t)) L_S => E[Sigma Sigma^T] ~ diag(s0 * sqrt(v_bar)) @ asset_corr @ diag(s0 * sqrt(v_bar))
        sqrt_v = np.sqrt(np.maximum(v_bar, 0.0))
        vol_scale = config.s0 * sqrt_v
        A_bar = np.outer(vol_scale, vol_scale) * config.asset_corr
        
        # Ensure exact symmetry
        return 0.5 * (A_bar + A_bar.T)

    @staticmethod
    def decompose(
        A: np.ndarray,
        d_prime: int,
        gamma: Optional[np.ndarray] = None,
        T: float = 1.0
    ) -> SpectralDecomposition:
        """
        Compute exact eigendecomposition of symmetric positive semi-definite matrix A,
        order eigenvalues in descending order, and construct the Ky Fan optimal rank-d' projector.
        
        Parameters
        ----------
        A : np.ndarray
            Symmetric positive semi-definite covariance matrix, shape (d, d).
        d_prime : int
            Number of retained eigenmodes (d' <= d).
        gamma : Optional[np.ndarray]
            Mean reversion rates (for Vasicek), shape (d,).
        T : float
            Maturity horizon.
            
        Returns
        -------
        SpectralDecomposition
        """
        d = A.shape[0]
        assert 1 <= d_prime <= d, f"d_prime={d_prime} must be between 1 and {d}"

        # scipy.linalg.eigh returns eigenvalues in ascending order
        eigvals, eigvecs = scipy.linalg.eigh(A)
        
        # Clean small numerical negative eigenvalues
        eigvals = np.maximum(eigvals, 0.0)
        
        # Sort in descending order: lambda_1 >= lambda_2 >= ... >= lambda_d
        sort_idx = np.argsort(eigvals)[::-1]
        ordered_eigvals = eigvals[sort_idx]
        ordered_eigvecs = eigvecs[:, sort_idx]

        # Top d' eigenvectors
        U_retained = ordered_eigvecs[:, :d_prime]  # shape (d, d')
        
        # Orthogonal projector P = U_retained @ U_retained^T
        P = U_retained @ U_retained.T
        I = np.eye(d)
        P_perp = I - P

        # Discarded spectral energy
        total_energy = float(np.sum(ordered_eigvals))
        discarded_tail = float(np.sum(ordered_eigvals[d_prime:])) if d_prime < d else 0.0
        retained_energy_ratio = 1.0 - (discarded_tail / total_energy) if total_energy > 0 else 1.0

        # Vasicek specific tail calculation
        if gamma is not None and d_prime < d:
            # Paper Section 13.4: each mode's effective mean-reversion rate is u_j^T Theta u_j
            Theta_mat = np.diag(gamma) if gamma.ndim == 1 else gamma
            # Mode-by-mode effective reversion rate: u_j^T Theta u_j
            mode_gammas = np.sum(ordered_eigvecs * (Theta_mat @ ordered_eigvecs), axis=0)
            discarded_gammas = mode_gammas[d_prime:]
            two_g_T = 2.0 * discarded_gammas * T
            rate_factor = np.where(two_g_T > 1e-7, (1.0 - np.exp(-two_g_T)) / (2.0 * discarded_gammas), T)
            discarded_tail_vasicek = float(np.sum(ordered_eigvals[d_prime:] * rate_factor))
        else:
            discarded_tail_vasicek = discarded_tail * T

        return SpectralDecomposition(
            eigenvalues=ordered_eigvals,
            eigenvectors=ordered_eigvecs,
            retained_dim=d_prime,
            projector=P,
            projector_perp=P_perp,
            discarded_tail=discarded_tail,
            discarded_tail_vasicek=discarded_tail_vasicek,
            retained_energy_ratio=retained_energy_ratio
        )

    @staticmethod
    def compute_exact_path_gap_vasicek_general(
        Theta: np.ndarray,
        sigma: np.ndarray,
        projector: np.ndarray,
        T: float,
        num_quad_steps: int = 150
    ) -> float:
        r"""
        Exact Ito-isometry path gap identity for general (potentially non-commuting)
        matrix drift Theta and diffusion sigma (Paper Section 13.2, lines 1803-1807):
        
        E[||e_T||^2] = \int_0^T || e^{-\Theta v} \Sigma - e^{-\hat{\Theta} v} \mathcal{P} \Sigma ||_{HS}^2 dv
        where \hat{\Theta} = \mathcal{P} \Theta \mathcal{P}.
        Evaluated via high-order Gauss-Legendre quadrature with matrix exponentials.
        """
        P_Theta_P = projector @ Theta @ projector
        P_sigma = projector @ sigma

        nodes, weights = np.polynomial.legendre.leggauss(num_quad_steps)
        # Transform [-1, 1] to [0, T]
        v_points = 0.5 * T * (nodes + 1.0)
        w_points = 0.5 * T * weights

        integral = 0.0
        for v, w in zip(v_points, w_points):
            exp_full = scipy.linalg.expm(-Theta * v) @ sigma
            exp_proj = scipy.linalg.expm(-P_Theta_P * v) @ P_sigma
            diff = exp_full - exp_proj
            # Hilbert-Schmidt norm squared = trace(diff.T @ diff)
            hs_norm_sq = np.sum(diff ** 2)
            integral += w * hs_norm_sq

        return float(integral)

    @staticmethod
    def compute_active_subspace_projector(
        payoff: BasePayoff,
        A: np.ndarray,
        d_prime: int
    ) -> np.ndarray:
        """
        Construct payoff-aware active-subspace projector (Section 16).
        Prioritizes the payoff sensitivity direction w over plain PCA.
        """
        d = A.shape[0]
        w = payoff.primary_subspace_weights
        w_norm = np.linalg.norm(w)
        if w_norm == 0:
            # Fallback to plain PCA
            return SpectralEngine.decompose(A, d_prime).projector
        
        w_unit = (w / w_norm).reshape(-1, 1)  # shape (d, 1)
        
        if d_prime == 1:
            return w_unit @ w_unit.T
        
        # For d' > 1: Gram-Schmidt starting from w_unit, then top eigenvectors of (I - w_unit w_unit^T) A (I - w_unit w_unit^T)
        P_w = w_unit @ w_unit.T
        P_perp_w = np.eye(d) - P_w
        A_proj = P_perp_w @ A @ P_perp_w
        
        vals, vecs = scipy.linalg.eigh(A_proj)
        sort_idx = np.argsort(vals)[::-1]
        additional_vecs = vecs[:, sort_idx[:d_prime - 1]]
        
        basis = np.hstack([w_unit, additional_vecs])
        # Re-orthogonalize via QR decomposition
        Q, _ = np.linalg.qr(basis)
        P_active = Q[:, :d_prime] @ Q[:, :d_prime].T
        return P_active

    @staticmethod
    def theoretical_upper_bound_one_minus_rho2(
        spec: SpectralDecomposition,
        L_f: float,
        var_pi: float,
        r: float,
        T: float,
        model_type: str = "bachelier"
    ) -> float:
        r"""
        Compute the theoretical upper bound on (1 - rho^2).
        
        - Constant-Coefficient / Bachelier (Section 4):
          1 - rho^2 <= (L_f^2 e^{-2rT} T / Var(Pi)) * sum_{j > d'} lambda_j
          
        - Vasicek (Section 13):
          1 - rho^2 <= (L_f^2 e^{-2rT} / Var(Pi)) * sum_{j > d'} lambda_j * (1 - e^{-2*gamma_j*T}) / (2*gamma_j)
          
        - Heston (Section 7.5 / 11):
          1 - rho^2 <= (L_f^2 e^{-2rT} T / Var(Pi)) * sum_{j > d'} \bar{\lambda}_j
        """
        if var_pi <= 0.0:
            return 1.0

        discount_sq = np.exp(-2.0 * r * T)
        prefactor = (L_f ** 2) * discount_sq / var_pi

        if model_type == "vasicek":
            bound = prefactor * spec.discarded_tail_vasicek
        else:
            bound = prefactor * T * spec.discarded_tail

        return float(bound)

    @staticmethod
    def theoretical_vrf_floor(bound_one_minus_rho2: float) -> float:
        """
        Theoretical floor on the Variance Reduction Factor:
        VRF = 1 / (1 - rho^2) >= 1 / bound
        """
        if bound_one_minus_rho2 <= 0.0:
            return float("inf")
        # Since 1 - rho^2 <= min(1.0, bound), VRF >= 1 / bound
        return 1.0 / bound_one_minus_rho2
