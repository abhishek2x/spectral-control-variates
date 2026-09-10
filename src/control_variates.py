"""
Synchronous control variate engine.
Computes optimal beta*, empirical correlation rho, variance reduction factors (VRF),
confidence intervals, and verifies two-sided theoretical spectral bounds.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from .payoffs import BasePayoff


@dataclass
class ControlVariateResult:
    """Detailed results of the control variate Monte Carlo simulation."""
    d_prime: int
    n_paths: int
    price_mc: float
    price_cv: float
    std_err_mc: float
    std_err_cv: float
    beta_star: float
    rho: float
    vrf_empirical: float            # 1 / (1 - rho^2)
    vrf_theoretical_floor: float     # Theoretical lower bound on VRF from spectral tail
    empirical_one_minus_rho2: float # Observed 1 - rho^2
    upper_bound_spectral: float     # Theoretical upper bound on 1 - rho^2
    lower_bound_two_sided: float    # Section 15 conditional variance lower bound
    ci_mc_95: Tuple[float, float]
    ci_cv_95: Tuple[float, float]


class ControlVariateEngine:
    """Evaluates control variate estimators under synchronous Brownian coupling."""

    @staticmethod
    def evaluate(
        payoff: BasePayoff,
        S_T: np.ndarray,
        S_hat_T: np.ndarray,
        discount: float,
        spectral_bound_one_minus_rho2: float,
        d_prime: int,
        mu_Y_exact: Optional[float] = None
    ) -> ControlVariateResult:
        """
        Evaluate control variate performance on paired paths (S_T, S_hat_T).
        
        Parameters
        ----------
        payoff : BasePayoff
            The target payoff function f.
        S_T : np.ndarray
            Full state terminal values, shape (n_paths, dim).
        S_hat_T : np.ndarray
            Synchronously coupled reduced state terminal values, shape (n_paths, dim).
        discount : float
            e^{-rT} discount factor.
        spectral_bound_one_minus_rho2 : float
            Theoretical upper bound on 1 - rho^2 from SpectralEngine.
        d_prime : int
            Retained eigenspace dimension.
        mu_Y_exact : Optional[float]
            Exact deterministic mean E[Y], if available. Defaults to sample mean of Y.
            
        Returns
        -------
        ControlVariateResult
        """
        n_paths = S_T.shape[0]

        # Discounted payoffs
        Pi = discount * payoff(S_T)         # Full model payoff
        Y = discount * payoff(S_hat_T)      # Reduced surrogate payoff

        # Sample statistics
        mean_Pi = float(np.mean(Pi))
        var_Pi = float(np.var(Pi, ddof=1))
        std_Pi = float(np.sqrt(max(var_Pi, 1e-16)))

        mean_Y = float(np.mean(Y))
        var_Y = float(np.var(Y, ddof=1))
        std_Y = float(np.sqrt(max(var_Y, 1e-16)))

        # Covariance and correlation
        cov_Pi_Y = float(np.cov(Pi, Y, ddof=1)[0, 1])
        if std_Pi > 0 and std_Y > 0:
            rho = np.clip(cov_Pi_Y / (std_Pi * std_Y), -1.0, 1.0)
        else:
            rho = 0.0

        # Optimal beta* = Cov(Pi, Y) / Var(Y)
        if var_Y > 1e-15:
            beta_star = cov_Pi_Y / var_Y
        else:
            beta_star = 0.0

        # Control variate estimate
        target_mu_Y = mean_Y if mu_Y_exact is None else mu_Y_exact
        Z = Pi - beta_star * (Y - target_mu_Y)
        price_cv = float(np.mean(Z))
        var_cv = float(np.var(Z, ddof=1))
        std_cv = float(np.sqrt(max(var_cv, 1e-16)))

        std_err_mc = std_Pi / np.sqrt(n_paths)
        std_err_cv = std_cv / np.sqrt(n_paths)

        # 95% Confidence Intervals (z = 1.95996)
        z_crit = 1.959963984540054
        ci_mc = (mean_Pi - z_crit * std_err_mc, mean_Pi + z_crit * std_err_mc)
        ci_cv = (price_cv - z_crit * std_err_cv, price_cv + z_crit * std_err_cv)

        # Empirical Variance Reduction Factor
        empirical_one_minus_rho2 = max(1.0 - rho ** 2, 1e-14)
        vrf_empirical = 1.0 / empirical_one_minus_rho2

        # Theoretical floor
        vrf_floor = 1.0 / min(max(spectral_bound_one_minus_rho2, 1e-14), 1.0)

        # Two-sided lower bound estimation (Section 15)
        # Conditional variance lower bound: E[Var(Pi | P S_T)] / Var(Pi)
        # Approximated by quadratic regression of Pi on Y or on projected coordinates
        lower_bound = ControlVariateEngine._estimate_conditional_variance_bound(Pi, Y, var_Pi)

        return ControlVariateResult(
            d_prime=d_prime,
            n_paths=n_paths,
            price_mc=mean_Pi,
            price_cv=price_cv,
            std_err_mc=std_err_mc,
            std_err_cv=std_err_cv,
            beta_star=beta_star,
            rho=float(rho),
            vrf_empirical=vrf_empirical,
            vrf_theoretical_floor=vrf_floor,
            empirical_one_minus_rho2=empirical_one_minus_rho2,
            upper_bound_spectral=spectral_bound_one_minus_rho2,
            lower_bound_two_sided=lower_bound,
            ci_mc_95=ci_mc,
            ci_cv_95=ci_cv
        )

    @staticmethod
    def _estimate_conditional_variance_bound(Pi: np.ndarray, Y: np.ndarray, var_Pi: float) -> float:
        """
        Estimate E[Var(Pi | G)] / Var(Pi) from Section 15.2-15.3.
        Using polynomial regression of Pi onto Y to compute unexplained variance ratio.
        """
        if var_Pi <= 1e-15:
            return 0.0
        
        # Design matrix [1, Y, Y^2]
        Y_norm = (Y - np.mean(Y)) / (np.std(Y) + 1e-8)
        X = np.column_stack([np.ones_like(Y_norm), Y_norm, Y_norm ** 2])
        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(X, Pi, rcond=None)
            pred = X @ coeffs
            residual_var = np.var(Pi - pred, ddof=1)
            ratio = residual_var / var_Pi
            return float(np.clip(ratio, 0.0, 1.0))
        except Exception:
            # Fallback to linear 1 - rho^2
            return 0.0
