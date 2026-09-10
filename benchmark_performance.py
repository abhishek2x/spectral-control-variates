"""
Performance benchmark: Pure Vectorized NumPy vs @njit Numba Multi-Threaded Loops.
Evaluates simulation runtime, throughput (paths/sec), and speedup for N up to 10^6 paths.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from src.config import SimulationConfig, BachelierConfig, VasicekConfig, HestonConfig
from src.spectral import SpectralEngine
from src.simulators import BachelierSimulator, VasicekSimulator, HestonSimulator


def benchmark_model_performance(
    model_name: str,
    sim_numpy,
    sim_numba,
    projector: np.ndarray,
    path_counts: list,
    n_steps: int = 100,
    T: float = 1.0,
    r: float = 0.03,
    n_warmup: int = 1000
) -> list:
    """Benchmark NumPy vs Numba for a given simulator across path counts."""
    print(f"\n==================================================================")
    print(f"BENCHMARKING {model_name.upper()}: NumPy Vectorized vs @njit Numba")
    print(f"==================================================================")

    # Warmup Numba JIT to exclude compilation latency from benchmark measurements
    warmup_cfg = SimulationConfig(n_paths=n_warmup, n_steps=n_steps, T=T, r=r, seed=1)
    _ = sim_numba.simulate_coupled(projector, warmup_cfg)
    _ = sim_numpy.simulate_coupled(projector, warmup_cfg)

    results = []
    for N in path_counts:
        sim_cfg = SimulationConfig(n_paths=N, n_steps=n_steps, T=T, r=r, seed=42)

        # 1. Pure NumPy Vectorized
        t0 = time.perf_counter()
        S_np, S_hat_np = sim_numpy.simulate_coupled(projector, sim_cfg)
        time_np = time.perf_counter() - t0

        # 2. Numba @njit Multi-threaded
        t0 = time.perf_counter()
        S_nb, S_hat_nb = sim_numba.simulate_coupled(projector, sim_cfg)
        time_nb = time.perf_counter() - t0

        speedup = time_np / max(time_nb, 1e-9)
        throughput_np = N / time_np
        throughput_nb = N / time_nb

        # Parity check
        mean_diff = float(np.max(np.abs(np.mean(S_np, axis=0) - np.mean(S_nb, axis=0))))

        print(f"  N = {N:10,d} | NumPy: {time_np:7.3f}s ({throughput_np/1e6:5.2f}M paths/s) | "
              f"Numba: {time_nb:7.3f}s ({throughput_nb/1e6:5.2f}M paths/s) | Speedup: {speedup:6.1f}x")

        results.append({
            "model": model_name,
            "n_paths": N,
            "n_steps": n_steps,
            "time_numpy_sec": time_np,
            "time_numba_sec": time_nb,
            "throughput_numpy_mpaths_per_sec": throughput_np / 1e6,
            "throughput_numba_mpaths_per_sec": throughput_nb / 1e6,
            "speedup": speedup,
            "max_mean_diff": mean_diff
        })

    return results


def run_full_performance_suite():
    """Execute complete performance benchmark for Bachelier, Vasicek, and Heston."""
    path_counts = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
    all_records = []

    # 1. Bachelier Model (dim=5)
    dim_b = 5
    cfg_b = BachelierConfig(
        dim=dim_b,
        s0=np.zeros(dim_b),
        drift=np.zeros(dim_b),
        sigma=np.diag([2.0, 1.5, 1.0, 0.5, 0.2])
    )
    A_b = SpectralEngine.compute_covariance_bachelier(cfg_b)
    decomp_b = SpectralEngine.decompose(A_b, d_prime=2)
    sim_b_np = BachelierSimulator(cfg_b, use_numba=False)
    sim_b_nb = BachelierSimulator(cfg_b, use_numba=True)
    all_records.extend(benchmark_model_performance(
        "Bachelier", sim_b_np, sim_b_nb, decomp_b.projector, path_counts, n_steps=1
    ))

    # 2. Vasicek OU Model (dim=5, n_steps=100)
    dim_v = 5
    cfg_v = VasicekConfig(
        dim=dim_v,
        s0=np.zeros(dim_v),
        theta=np.zeros(dim_v),
        gamma=np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
        sigma=np.diag([2.0, 1.5, 1.0, 0.5, 0.2])
    )
    A_v = SpectralEngine.compute_covariance_vasicek(cfg_v)
    decomp_v = SpectralEngine.decompose(A_v, d_prime=2, gamma=cfg_v.gamma)
    sim_v_np = VasicekSimulator(cfg_v, use_numba=False)
    sim_v_nb = VasicekSimulator(cfg_v, use_numba=True)
    all_records.extend(benchmark_model_performance(
        "Vasicek", sim_v_np, sim_v_nb, decomp_v.projector, path_counts, n_steps=100
    ))

    # 3. Heston Model (dim=4, n_steps=100)
    dim_h = 4
    corr_h = np.array([
        [1.00, 0.70, 0.60, 0.50],
        [0.70, 1.00, 0.65, 0.55],
        [0.60, 0.65, 1.00, 0.50],
        [0.50, 0.55, 0.50, 1.00]
    ])
    cfg_h = HestonConfig(
        dim=dim_h,
        s0=np.ones(dim_h) * 100.0,
        v0=np.array([0.04, 0.05, 0.035, 0.045]),
        kappa=np.array([1.5, 2.0, 1.8, 1.2]),
        theta=np.array([0.04, 0.04, 0.04, 0.04]),
        xi=np.array([0.3, 0.35, 0.25, 0.3]),
        rho=np.array([-0.7, -0.65, -0.7, -0.6]),
        asset_corr=corr_h
    )
    A_h = SpectralEngine.compute_covariance_heston(cfg_h, T=1.0)
    decomp_h = SpectralEngine.decompose(A_h, d_prime=2)
    sim_h_np = HestonSimulator(cfg_h, use_numba=False)
    sim_h_nb = HestonSimulator(cfg_h, use_numba=True)
    all_records.extend(benchmark_model_performance(
        "Heston", sim_h_np, sim_h_nb, decomp_h.projector, path_counts, n_steps=100
    ))

    df = pd.DataFrame(all_records)
    os.makedirs("benchmark_results", exist_ok=True)
    df.to_csv("benchmark_results/performance_benchmark.csv", index=False)
    print("\nSaved benchmark results to benchmark_results/performance_benchmark.csv")
    return df


def plot_performance_benchmark(df: pd.DataFrame, save_path: str = "figures/fig5_numba_vs_numpy_performance.png"):
    """Generate high-resolution publication plot comparing NumPy vs Numba performance."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Palatino", "Georgia"],
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Filter by model
    df_vas = df[df["model"] == "Vasicek"]
    df_hes = df[df["model"] == "Heston"]
    N = df_vas["n_paths"].values

    # Subplot 1: Wall-Clock Execution Time (Log-Log) for 100-step SDEs
    ax1 = axes[0]
    ax1.loglog(N, df_vas["time_numpy_sec"], "o--", color="#d62728", linewidth=2.2, label="Vasicek (NumPy)")
    ax1.loglog(N, df_vas["time_numba_sec"], "o-", color="#2ca02c", linewidth=2.5, label="Vasicek (Numba)")
    ax1.loglog(N, df_hes["time_numpy_sec"], "s--", color="#ff7f0e", linewidth=2.2, label="Heston (NumPy)")
    ax1.loglog(N, df_hes["time_numba_sec"], "s-", color="#1f77b4", linewidth=2.5, label="Heston (Numba)")

    ax1.set_title("Simulation Wall-Clock Time (100 Steps)", fontweight="bold", pad=12)
    ax1.set_xlabel("Number of Simulated Paths $N$")
    ax1.set_ylabel("Execution Time (Seconds, Log Scale)")
    ax1.legend(loc="upper left", frameon=True)

    # Subplot 2: Throughput at N = 10^6 paths
    ax2 = axes[1]
    models = ["Bachelier\n(1 step)", "Vasicek\n(100 steps)", "Heston\n(100 steps)"]
    df_1m = df[df["n_paths"] == 1_000_000]
    tp_np = df_1m["throughput_numpy_mpaths_per_sec"].values
    tp_nb = df_1m["throughput_numba_mpaths_per_sec"].values

    x = np.arange(len(models))
    width = 0.35
    b1 = ax2.bar(x - width/2, tp_np, width, label="NumPy Vectorized", color="#e74c3c", edgecolor="black")
    b2 = ax2.bar(x + width/2, tp_nb, width, label="Numba @njit Multi-Threaded", color="#2ecc71", edgecolor="black")

    ax2.set_ylabel("Throughput (Million Paths / Second)")
    ax2.set_title(r"Simulation Throughput at $N = 10^6$ Paths", fontweight="bold", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend(loc="upper right")

    for bar in b1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.05, f"{h:.2f}M", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.05, f"{h:.2f}M", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Subplot 3: Numba Speedup vs N
    ax3 = axes[2]
    ax3.plot(N, df_vas["speedup"], "o-", color="#2ca02c", linewidth=2.5, markersize=8, label="Vasicek (100 Steps)")
    ax3.plot(N, df_hes["speedup"], "s-", color="#1f77b4", linewidth=2.5, markersize=8, label="Heston (100 Steps)")
    df_bac = df[df["model"] == "Bachelier"]
    ax3.plot(N, df_bac["speedup"], "^-", color="#9467bd", linewidth=2.0, markersize=7, label="Bachelier (1 Step)")

    ax3.set_xscale("log")
    ax3.set_title("Numba Parallel Speedup vs. Path Count", fontweight="bold", pad=12)
    ax3.set_xlabel("Number of Simulated Paths $N$")
    ax3.set_ylabel(r"Speedup Factor ($\times$ Faster than NumPy)")
    ax3.axhline(1.0, color="gray", linestyle=":", label="Parity (1.0x)")
    ax3.legend(loc="upper left")

    for x_val, y_val in zip(N, df_vas["speedup"]):
        if x_val in [10_000, 100_000, 1_000_000]:
            ax3.annotate(f"{y_val:.1f}x", xy=(x_val, y_val), xytext=(0, 7),
                         textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle(r"Path Generation Optimization: NumPy Vectorization vs. @njit Numba ($N \leq 10^6$ Paths)",
                 fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved performance plot to {save_path}")


if __name__ == "__main__":
    df_results = run_full_performance_suite()
    plot_performance_benchmark(df_results)
