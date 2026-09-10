"""
Benchmarking suite for spectral reduced-variance option pricing.
Validates theoretical bounds against Monte Carlo simulations across:
1. Retained dimension d' in [1, ..., d-1]
2. Path convergence N in [10^4, 10^6]
3. In-subspace European basket vs out-of-subspace active payoffs (Section 16)
"""

import os
import json
import time
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from src.config import SimulationConfig, BachelierConfig, VasicekConfig, HestonConfig
from src.payoffs import EuropeanBasketCall, EuropeanBasketPut, OutOfSubspaceSpread, RainbowWorstOf
from src.spectral import SpectralEngine
from src.simulators import BachelierSimulator, VasicekSimulator, HestonSimulator
from src.control_variates import ControlVariateEngine, ControlVariateResult


def run_dimension_sweep_bachelier(
    dim: int = 5,
    n_paths: int = 150_000,
    T: float = 1.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Sweep retained dimension d' from 1 to dim-1 on a multi-asset Bachelier model.
    Compare empirical VRF against the theoretical spectral bound floor.
    """
    print(f"\n--- Running Dimension Sweep (Bachelier, dim={dim}, N={n_paths}) ---")
    rng = np.random.default_rng(seed)
    
    # Generate structured eigenvalues: steep decay (e.g. market factor + sector + idiosyncratic)
    variances = np.array([4.0, 1.8, 0.8, 0.3, 0.1])[:dim]
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    A_target = Q @ np.diag(variances) @ Q.T
    # Cholesky factor as sigma
    sigma = np.linalg.cholesky(A_target)

    cfg = BachelierConfig(
        dim=dim,
        s0=np.zeros(dim),
        drift=np.zeros(dim),
        sigma=sigma
    )
    sim_cfg = SimulationConfig(n_paths=n_paths, T=T, r=0.03, seed=seed)
    sim = BachelierSimulator(cfg)
    A = SpectralEngine.compute_covariance_bachelier(cfg)

    # Basket weights uniform
    weights = np.ones(dim) / np.sqrt(dim)
    payoff = EuropeanBasketCall(strike=0.0, weights=weights)

    # Pre-simulate full paths
    discount = np.exp(-sim_cfg.r * sim_cfg.T)
    # Estimate Var(Pi)
    full_S_T, _ = sim.simulate_coupled(np.eye(dim), sim_cfg)
    var_pi = float(np.var(discount * payoff(full_S_T), ddof=1))

    results = []
    for d_p in range(1, dim):
        decomp = SpectralEngine.decompose(A, d_prime=d_p, T=T)
        S_T, S_hat_T = sim.simulate_coupled(decomp.projector, sim_cfg)

        bound_one_minus_rho2 = SpectralEngine.theoretical_upper_bound_one_minus_rho2(
            decomp,
            L_f=payoff.lipschitz_constant,
            var_pi=var_pi,
            r=sim_cfg.r,
            T=sim_cfg.T,
            model_type="bachelier"
        )
        res = ControlVariateEngine.evaluate(
            payoff=payoff,
            S_T=S_T,
            S_hat_T=S_hat_T,
            discount=discount,
            spectral_bound_one_minus_rho2=bound_one_minus_rho2,
            d_prime=d_p
        )
        
        results.append({
            "model": "Bachelier",
            "d_prime": d_p,
            "retained_energy_pct": decomp.retained_energy_ratio * 100.0,
            "discarded_tail": decomp.discarded_tail,
            "rho": res.rho,
            "one_minus_rho2_empirical": res.empirical_one_minus_rho2,
            "one_minus_rho2_bound": res.upper_bound_spectral,
            "vrf_empirical": res.vrf_empirical,
            "vrf_theoretical_floor": res.vrf_theoretical_floor,
            "lower_bound_two_sided": res.lower_bound_two_sided,
            "std_err_mc": res.std_err_mc,
            "std_err_cv": res.std_err_cv,
            "bound_satisfied": res.empirical_one_minus_rho2 <= res.upper_bound_spectral + 1e-7
        })
        print(f"  d'={d_p} | Tail={decomp.discarded_tail:.3f} | rho={res.rho:.4f} | "
              f"VRF={res.vrf_empirical:.2f}x | Floor={res.vrf_theoretical_floor:.2f}x | "
              f"Bound Valid: {results[-1]['bound_satisfied']}")

    return pd.DataFrame(results)


def run_dimension_sweep_vasicek(
    dim: int = 5,
    n_paths: int = 100_000,
    T: float = 1.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Sweep retained dimension d' from 1 to dim-1 on a multi-factor Vasicek / OU model (Section 13).
    """
    print(f"\n--- Running Dimension Sweep (Vasicek, dim={dim}, N={n_paths}) ---")
    rng = np.random.default_rng(seed)
    
    variances = np.array([3.5, 1.5, 0.7, 0.3, 0.1])[:dim]
    gamma = np.array([0.5, 1.0, 1.5, 2.0, 2.5])[:dim]
    sigma = np.diag(np.sqrt(variances))

    cfg = VasicekConfig(
        dim=dim,
        s0=np.zeros(dim),
        theta=np.zeros(dim),
        gamma=gamma,
        sigma=sigma
    )
    sim_cfg = SimulationConfig(n_paths=n_paths, n_steps=60, T=T, r=0.03, seed=seed)
    sim = VasicekSimulator(cfg)
    A = SpectralEngine.compute_covariance_vasicek(cfg)

    weights = np.ones(dim) / np.sqrt(dim)
    payoff = EuropeanBasketCall(strike=0.0, weights=weights)

    discount = np.exp(-sim_cfg.r * sim_cfg.T)
    full_S_T, _ = sim.simulate_coupled(np.eye(dim), sim_cfg)
    var_pi = float(np.var(discount * payoff(full_S_T), ddof=1))

    results = []
    for d_p in range(1, dim):
        decomp = SpectralEngine.decompose(A, d_prime=d_p, gamma=cfg.gamma, T=T)
        S_T, S_hat_T = sim.simulate_coupled(decomp.projector, sim_cfg)

        bound_one_minus_rho2 = SpectralEngine.theoretical_upper_bound_one_minus_rho2(
            decomp,
            L_f=payoff.lipschitz_constant,
            var_pi=var_pi,
            r=sim_cfg.r,
            T=sim_cfg.T,
            model_type="vasicek"
        )
        res = ControlVariateEngine.evaluate(
            payoff=payoff,
            S_T=S_T,
            S_hat_T=S_hat_T,
            discount=discount,
            spectral_bound_one_minus_rho2=bound_one_minus_rho2,
            d_prime=d_p
        )
        
        results.append({
            "model": "Vasicek",
            "d_prime": d_p,
            "retained_energy_pct": decomp.retained_energy_ratio * 100.0,
            "discarded_tail": decomp.discarded_tail_vasicek,
            "rho": res.rho,
            "one_minus_rho2_empirical": res.empirical_one_minus_rho2,
            "one_minus_rho2_bound": res.upper_bound_spectral,
            "vrf_empirical": res.vrf_empirical,
            "vrf_theoretical_floor": res.vrf_theoretical_floor,
            "lower_bound_two_sided": res.lower_bound_two_sided,
            "std_err_mc": res.std_err_mc,
            "std_err_cv": res.std_err_cv,
            "bound_satisfied": res.empirical_one_minus_rho2 <= res.upper_bound_spectral + 1e-7
        })
        print(f"  d'={d_p} | Tail_OU={decomp.discarded_tail_vasicek:.3f} | rho={res.rho:.4f} | "
              f"VRF={res.vrf_empirical:.2f}x | Floor={res.vrf_theoretical_floor:.2f}x | "
              f"Bound Valid: {results[-1]['bound_satisfied']}")

    return pd.DataFrame(results)


def run_dimension_sweep_heston(
    dim: int = 4,
    n_paths: int = 100_000,
    T: float = 1.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Sweep retained dimension d' from 1 to dim-1 on Multi-Factor Heston Stochastic Volatility Model.
    """
    print(f"\n--- Running Dimension Sweep (Heston, dim={dim}, N={n_paths}) ---")
    
    # Cross-asset correlation with dominant first factor
    corr = np.array([
        [1.00, 0.70, 0.60, 0.50],
        [0.70, 1.00, 0.65, 0.55],
        [0.60, 0.65, 1.00, 0.50],
        [0.50, 0.55, 0.50, 1.00]
    ])[:dim, :dim]

    cfg = HestonConfig(
        dim=dim,
        s0=np.ones(dim) * 100.0,
        v0=np.array([0.04, 0.05, 0.035, 0.045])[:dim],
        kappa=np.array([1.5, 2.0, 1.8, 1.2])[:dim],
        theta=np.array([0.04, 0.04, 0.04, 0.04])[:dim],
        xi=np.array([0.3, 0.35, 0.25, 0.3])[:dim],
        rho=np.array([-0.7, -0.65, -0.7, -0.6])[:dim],
        asset_corr=corr
    )
    sim_cfg = SimulationConfig(n_paths=n_paths, n_steps=60, T=T, r=0.03, seed=seed)
    sim = HestonSimulator(cfg)
    A_bar = SpectralEngine.compute_covariance_heston(cfg, T=T)

    weights = np.ones(dim) / dim
    payoff = EuropeanBasketCall(strike=100.0, weights=weights)

    discount = np.exp(-sim_cfg.r * sim_cfg.T)
    # Pilot run for var_pi
    full_S_T, _ = sim.simulate_coupled(np.eye(dim), sim_cfg)
    var_pi = float(np.var(discount * payoff(full_S_T), ddof=1))

    results = []
    for d_p in range(1, dim):
        decomp = SpectralEngine.decompose(A_bar, d_prime=d_p, T=T)
        S_T, S_hat_T = sim.simulate_coupled(decomp.projector, sim_cfg)

        bound_one_minus_rho2 = SpectralEngine.theoretical_upper_bound_one_minus_rho2(
            decomp,
            L_f=payoff.lipschitz_constant,
            var_pi=var_pi,
            r=sim_cfg.r,
            T=sim_cfg.T,
            model_type="heston"
        )
        res = ControlVariateEngine.evaluate(
            payoff=payoff,
            S_T=S_T,
            S_hat_T=S_hat_T,
            discount=discount,
            spectral_bound_one_minus_rho2=bound_one_minus_rho2,
            d_prime=d_p
        )
        
        results.append({
            "model": "Heston",
            "d_prime": d_p,
            "retained_energy_pct": decomp.retained_energy_ratio * 100.0,
            "discarded_tail": decomp.discarded_tail,
            "rho": res.rho,
            "one_minus_rho2_empirical": res.empirical_one_minus_rho2,
            "one_minus_rho2_bound": res.upper_bound_spectral,
            "vrf_empirical": res.vrf_empirical,
            "vrf_theoretical_floor": res.vrf_theoretical_floor,
            "lower_bound_two_sided": res.lower_bound_two_sided,
            "std_err_mc": res.std_err_mc,
            "std_err_cv": res.std_err_cv,
            "bound_satisfied": res.empirical_one_minus_rho2 <= res.upper_bound_spectral + 1e-7
        })
        print(f"  d'={d_p} | Tail_bar={decomp.discarded_tail:.4f} | rho={res.rho:.4f} | "
              f"VRF={res.vrf_empirical:.2f}x | Floor={res.vrf_theoretical_floor:.2f}x")

    return pd.DataFrame(results)


def run_convergence_benchmark(
    path_counts: List[int] = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000],
    dim: int = 5,
    d_prime: int = 2,
    seed: int = 101
) -> pd.DataFrame:
    """
    Convergence validation across paths N in [10^4, 10^6].
    Evaluates empirical standard error decay O(1 / sqrt(N)) and stability of beta*.
    """
    print(f"\n--- Running Path Convergence Benchmark (N in {path_counts[0]} to {path_counts[-1]}) ---")
    
    variances = np.array([4.0, 1.8, 0.8, 0.3, 0.1])
    sigma = np.diag(np.sqrt(variances))
    cfg = BachelierConfig(dim=dim, s0=np.zeros(dim), drift=np.zeros(dim), sigma=sigma)
    sim = BachelierSimulator(cfg)
    A = SpectralEngine.compute_covariance_bachelier(cfg)
    decomp = SpectralEngine.decompose(A, d_prime=d_prime, T=1.0)

    weights = np.ones(dim) / np.sqrt(dim)
    payoff = EuropeanBasketCall(strike=0.0, weights=weights)

    results = []
    for N in path_counts:
        sim_cfg = SimulationConfig(n_paths=N, T=1.0, r=0.03, seed=seed)
        t0 = time.perf_counter()
        S_T, S_hat_T = sim.simulate_coupled(decomp.projector, sim_cfg)
        sim_time = time.perf_counter() - t0

        discount = np.exp(-sim_cfg.r * sim_cfg.T)
        res = ControlVariateEngine.evaluate(
            payoff=payoff,
            S_T=S_T,
            S_hat_T=S_hat_T,
            discount=discount,
            spectral_bound_one_minus_rho2=1.0,
            d_prime=d_prime
        )

        results.append({
            "n_paths": N,
            "price_mc": res.price_mc,
            "price_cv": res.price_cv,
            "std_err_mc": res.std_err_mc,
            "std_err_cv": res.std_err_cv,
            "error_reduction_ratio": res.std_err_mc / res.std_err_cv,
            "beta_star": res.beta_star,
            "rho": res.rho,
            "vrf": res.vrf_empirical,
            "sim_time_sec": sim_time
        })
        print(f"  N={N:10,d} | SE_MC={res.std_err_mc:.5f} | SE_CV={res.std_err_cv:.5f} | "
              f"Ratio={res.std_err_mc/res.std_err_cv:.2f}x | beta*={res.beta_star:.4f}")

    return pd.DataFrame(results)


def run_active_subspace_benchmark(
    dim: int = 5,
    n_paths: int = 150_000,
    seed: int = 888
) -> pd.DataFrame:
    """
    Test cases comparing:
    1. Standard European basket payoff (gradient strongly in dominant eigenspace)
    2. Out-of-subspace spread payoff (gradient aligned with discarded tail mode u_d)
    3. Worst-of Rainbow payoff (nonlinear cross-asset sensitivity)
    
    Demonstrates Section 16 (Active Subspaces): Plain PCA fails when payoff sensitivity
    is orthogonal to dominant eigenspace, whereas active subspace recovers high correlation!
    """
    print(f"\n--- Running Payoff Alignment & Active Subspace Benchmark (dim={dim}) ---")
    
    variances = np.array([5.0, 2.0, 1.0, 0.4, 0.1])
    sigma = np.diag(np.sqrt(variances))
    cfg = BachelierConfig(dim=dim, s0=np.zeros(dim), drift=np.zeros(dim), sigma=sigma)
    sim = BachelierSimulator(cfg)
    A = SpectralEngine.compute_covariance_bachelier(cfg)
    decomp_pca = SpectralEngine.decompose(A, d_prime=1, T=1.0)
    sim_cfg = SimulationConfig(n_paths=n_paths, T=1.0, r=0.03, seed=seed)
    discount = np.exp(-sim_cfg.r * sim_cfg.T)

    # Payoffs:
    # 1. Basket Call (aligned with top mode u_1 = [1, 0, 0, 0, 0])
    payoff_basket = EuropeanBasketCall(strike=0.0, weights=np.array([1.0, 0.5, 0.2, 0.1, 0.05]))
    
    # 2. Out-of-subspace Spread (strictly aligned with the last mode u_5 = [0, 0, 0, 0, 1])
    payoff_spread = OutOfSubspaceSpread(strike=0.0, direction_vector=np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
    
    # 3. Rainbow Worst-Of
    payoff_rainbow = RainbowWorstOf(strike=0.0, dim=dim)

    # Active projector for spread payoff (Section 16)
    P_active_spread = SpectralEngine.compute_active_subspace_projector(payoff_spread, A, d_prime=1)

    payoffs_tests = [
        ("Basket Call", payoff_basket, decomp_pca.projector, "Plain PCA (d'=1)"),
        ("Out-of-Subspace Spread", payoff_spread, decomp_pca.projector, "Plain PCA (d'=1)"),
        ("Out-of-Subspace Spread", payoff_spread, P_active_spread, "Active Subspace (d'=1)"),
        ("Rainbow Worst-Of", payoff_rainbow, decomp_pca.projector, "Plain PCA (d'=1)")
    ]

    results = []
    for name, p_obj, proj, method in payoffs_tests:
        S_T, S_hat_T = sim.simulate_coupled(proj, sim_cfg)
        res = ControlVariateEngine.evaluate(
            payoff=p_obj,
            S_T=S_T,
            S_hat_T=S_hat_T,
            discount=discount,
            spectral_bound_one_minus_rho2=1.0,
            d_prime=1
        )
        results.append({
            "payoff_name": name,
            "method": method,
            "rho": res.rho,
            "vrf": res.vrf_empirical,
            "std_err_mc": res.std_err_mc,
            "std_err_cv": res.std_err_cv,
            "lower_bound_two_sided": res.lower_bound_two_sided
        })
        print(f"  [{name:22s} | {method:24s}] -> rho={res.rho:.4f} | VRF={res.vrf_empirical:.2f}x | "
              f"SE_MC={res.std_err_mc:.5f} | SE_CV={res.std_err_cv:.5f}")

    return pd.DataFrame(results)


def run_all_benchmarks(output_dir: str = "benchmark_results") -> Dict[str, pd.DataFrame]:
    """Execute all benchmark suites and save results to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    print("======================================================================")
    print("STARTING FULL SPECTRAL REDUCED-VARIANCE BENCHMARK SUITE")
    print("======================================================================")

    df_bachelier = run_dimension_sweep_bachelier()
    df_bachelier.to_csv(os.path.join(output_dir, "bachelier_dimension_sweep.csv"), index=False)

    df_vasicek = run_dimension_sweep_vasicek()
    df_vasicek.to_csv(os.path.join(output_dir, "vasicek_dimension_sweep.csv"), index=False)

    df_heston = run_dimension_sweep_heston()
    df_heston.to_csv(os.path.join(output_dir, "heston_dimension_sweep.csv"), index=False)

    df_convergence = run_convergence_benchmark()
    df_convergence.to_csv(os.path.join(output_dir, "convergence_benchmark.csv"), index=False)

    df_active = run_active_subspace_benchmark()
    df_active.to_csv(os.path.join(output_dir, "active_subspace_benchmark.csv"), index=False)

    print("\nAll benchmark runs completed successfully! CSV data saved in:", output_dir)
    return {
        "bachelier": df_bachelier,
        "vasicek": df_vasicek,
        "heston": df_heston,
        "convergence": df_convergence,
        "active_subspace": df_active
    }


if __name__ == "__main__":
    run_all_benchmarks()
