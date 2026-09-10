"""
Publication-grade plotting suite for spectral reduced-variance option pricing.
Generates high-resolution figures validating theoretical bounds against Monte Carlo simulations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


def setup_style():
    """Set publication-grade aesthetics."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Palatino", "Georgia"],
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 15,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })


def plot_dimension_sweeps(
    df_bach: pd.DataFrame,
    df_vasi: pd.DataFrame,
    df_hest: pd.DataFrame,
    save_path: str = "figures/fig1_dimension_sweeps.png"
):
    """
    Figure 1: Empirical VRF vs Theoretical Bound Floor across retained dimension d'.
    Validates that empirical VRF is floored by the theoretical spectral bound.
    """
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    models = [
        ("Bachelier (Constant Coeff)", df_bach, axes[0], "#1f77b4", "#aec7e8"),
        ("Vasicek (Mean-Reverting OU)", df_vasi, axes[1], "#2ca02c", "#98df8a"),
        ("Heston (Stochastic Volatility)", df_hest, axes[2], "#d62728", "#ff9896")
    ]

    for title, df, ax, c_emp, c_floor in models:
        d_primes = df["d_prime"].values
        vrf_emp = df["vrf_empirical"].values
        vrf_floor = df["vrf_theoretical_floor"].values

        ax.plot(d_primes, vrf_emp, "o-", color=c_emp, linewidth=2.5, markersize=8,
                label=r"Empirical VRF $\frac{1}{1 - \rho^2}$", zorder=4)
        ax.plot(d_primes, vrf_floor, "s--", color="#333333", linewidth=2.0, markersize=7,
                label=r"Theoretical Floor $\frac{1}{\text{Bound}}$", zorder=3)
        ax.fill_between(d_primes, vrf_floor, vrf_emp, color=c_floor, alpha=0.35,
                        label="Guaranteed Performance Margin")

        ax.set_title(title, fontweight="bold", pad=12)
        ax.set_xlabel(r"Retained Dimension $d'$ (Eigenmodes)")
        ax.set_ylabel(r"Variance Reduction Factor (VRF)")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        
        # Use log scale if dynamic range is huge (e.g. Bachelier d'=4)
        if np.max(vrf_emp) > 1000:
            ax.set_yscale("log")
            ax.set_ylabel(r"Variance Reduction Factor (VRF, Log Scale)")
            ax.set_ylim(bottom=0.8, top=np.max(vrf_emp) * 10)
        else:
            ax.set_ylim(bottom=0.8)

        ax.legend(loc="upper left", frameon=True, framealpha=0.9)

        # Annotate points
        for x, y_emp in zip(d_primes, vrf_emp):
            label_text = f"{y_emp:.1e}x" if y_emp > 9999 else f"{y_emp:.1f}x"
            ax.annotate(label_text, xy=(x, y_emp), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("Empirical Variance Reduction Factor vs. Theoretical Spectral Bound Floor",
                 fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_convergence(
    df_conv: pd.DataFrame,
    save_path: str = "figures/fig2_convergence_scaling.png"
):
    """
    Figure 2: Monte Carlo vs Control Variate Convergence Scaling across N in [10^4, 10^6].
    Shows O(1 / sqrt(N)) scaling and parameter stability.
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    N = df_conv["n_paths"].values
    se_mc = df_conv["std_err_mc"].values
    se_cv = df_conv["std_err_cv"].values
    beta = df_conv["beta_star"].values
    rho = df_conv["rho"].values

    # Subplot 1: Standard Error Decay (Log-Log)
    ax1.loglog(N, se_mc, "o-", color="#d62728", linewidth=2.2, markersize=8, label="Standard Monte Carlo")
    ax1.loglog(N, se_cv, "s-", color="#1f77b4", linewidth=2.2, markersize=8, label=r"Spectral CV Estimator ($d'=2$)")
    
    # Reference 1/sqrt(N) slopes
    ref_mc = se_mc[0] * np.sqrt(N[0] / N)
    ref_cv = se_cv[0] * np.sqrt(N[0] / N)
    ax1.loglog(N, ref_mc, ":", color="#888888", label=r"$\mathcal{O}(1/\sqrt{N})$ theoretical slope")
    ax1.loglog(N, ref_cv, ":", color="#888888")

    ax1.set_title("Standard Error Scaling", fontweight="bold", pad=12)
    ax1.set_xlabel("Number of Simulated Paths $N$")
    ax1.set_ylabel("Standard Error (SE)")
    ax1.legend(loc="upper right", frameon=True)

    # Annotate speedup
    speedup = (se_mc / se_cv) ** 2
    avg_speedup = np.mean(speedup)
    ax1.text(0.05, 0.15, f"Effective Sample Size Speedup: ~{avg_speedup:.1f}x",
             transform=ax1.transAxes, fontsize=10.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f4f8", edgecolor="#1f77b4"))

    # Subplot 2: Parameter Stability (beta* and rho)
    color_beta = "#2ca02c"
    color_rho = "#9467bd"

    ax2_twin = ax2.twinx()
    l1 = ax2.plot(N, beta, "o-", color=color_beta, linewidth=2.2, markersize=7, label=r"Optimal Coefficient $\beta^*$")
    l2 = ax2_twin.plot(N, rho, "s-", color=color_rho, linewidth=2.2, markersize=7, label=r"Correlation $\rho$")

    ax2.set_xscale("log")
    ax2.set_title(r"Parameter Stability Across Path Counts $N$", fontweight="bold", pad=12)
    ax2.set_xlabel("Number of Simulated Paths $N$")
    ax2.set_ylabel(r"Optimal $\beta^*$", color=color_beta)
    ax2_twin.set_ylabel(r"Empirical Correlation $\rho$", color=color_rho)
    ax2.tick_params(axis="y", labelcolor=color_beta)
    ax2_twin.tick_params(axis="y", labelcolor=color_rho)
    ax2_twin.set_ylim(bottom=0.8, top=1.0)

    # Combined legend
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="lower right", frameon=True)

    plt.suptitle("Convergence Validation & Scalability Suite", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_active_subspace_benchmark(
    df_act: pd.DataFrame,
    save_path: str = "figures/fig3_active_subspace_comparison.png"
):
    """
    Figure 3: Active Subspaces vs Plain PCA (Section 16).
    Demonstrates failure of PCA for out-of-subspace payoffs and resolution via Active Subspaces.
    """
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    labels = [
        "Basket Call\n(PCA d'=1)",
        "Out-of-Subspace Spread\n(PCA d'=1)",
        "Out-of-Subspace Spread\n(Active Subspace d'=1)",
        "Worst-Of Rainbow\n(PCA d'=1)"
    ]
    rhos = df_act["rho"].values
    vrfs = df_act["vrf"].values

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    # Subplot 1: Correlation rho
    bars1 = ax1.bar(labels, rhos, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel(r"Correlation $\rho = \mathrm{corr}(\Pi, Y)$")
    ax1.set_title(r"Empirical Correlation $\rho$ with Surrogate", fontweight="bold", pad=12)
    ax1.set_ylim(0.0, 1.1)
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.7)

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., h + 0.02, f"{h:.3f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Subplot 2: Variance Reduction Factor VRF (Log Scale)
    bars2 = ax2.bar(labels, vrfs, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
    ax2.set_yscale("log")
    ax2.set_ylabel(r"Variance Reduction Factor (VRF, Log Scale)")
    ax2.set_title(r"Variance Reduction Factor $\frac{1}{1 - \rho^2}$", fontweight="bold", pad=12)
    ax2.axhline(1.0, color="red", linestyle=":", label="No Reduction (1.0x)")

    for bar in bars2:
        h = bar.get_height()
        label_text = f"{h:.1e}x" if h > 9999 else f"{h:.1f}x"
        ax2.text(bar.get_x() + bar.get_width() / 2., h * 1.25, label_text,
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.legend(loc="upper right")

    plt.suptitle("Active Subspace Refinement: Plain PCA Failure vs. Gradient Alignment (Section 16)",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def plot_two_sided_sandwich(
    df_bach: pd.DataFrame,
    save_path: str = "figures/fig4_two_sided_sandwich.png"
):
    """
    Figure 4: Two-Sided Sandwich Bound (Paper Section 15).
    Demonstrates: Lower Bound <= 1 - rho^2 <= Upper Bound.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    d_primes = df_bach["d_prime"].values
    lower = df_bach["lower_bound_two_sided"].values
    empirical = df_bach["one_minus_rho2_empirical"].values
    upper = df_bach["one_minus_rho2_bound"].values

    ax.plot(d_primes, upper, "s--", color="#d62728", linewidth=2.2, markersize=8,
            label=r"Spectral Upper Bound $\frac{L_f^2 e^{-2rT} T}{\sigma_\Pi^2} \sum_{j > d'} \lambda_j$")
    ax.plot(d_primes, empirical, "o-", color="#1f77b4", linewidth=2.5, markersize=9,
            label=r"Empirical $1 - \rho^2$")
    ax.plot(d_primes, lower, "^-.", color="#2ca02c", linewidth=2.2, markersize=8,
            label=r"Conditional Variance Lower Bound $\frac{\mathbb{E}[\mathrm{Var}(\Pi \mid \mathcal{P}S_T)]}{\mathrm{Var}(\Pi)}$")

    ax.fill_between(d_primes, lower, upper, color="#e6f2ff", alpha=0.5, label="Theoretical Sandwich Interval")

    ax.set_title("Two-Sided Spectral Criterion Sandwich (Section 15)", fontweight="bold", pad=12)
    ax.set_xlabel(r"Retained Dimension $d'$")
    ax.set_ylabel(r"$1 - \rho^2$ (Fractional Unexplained Variance)")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(loc="lower left", frameon=True, framealpha=0.95)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def generate_all_plots(benchmark_dir: str = "benchmark_results", figures_dir: str = "figures"):
    """Load benchmark CSVs and render all figures."""
    os.makedirs(figures_dir, exist_ok=True)

    df_bach = pd.read_csv(os.path.join(benchmark_dir, "bachelier_dimension_sweep.csv"))
    df_vasi = pd.read_csv(os.path.join(benchmark_dir, "vasicek_dimension_sweep.csv"))
    df_hest = pd.read_csv(os.path.join(benchmark_dir, "heston_dimension_sweep.csv"))
    df_conv = pd.read_csv(os.path.join(benchmark_dir, "convergence_benchmark.csv"))
    df_act = pd.read_csv(os.path.join(benchmark_dir, "active_subspace_benchmark.csv"))

    plot_dimension_sweeps(
        df_bach, df_vasi, df_hest,
        save_path=os.path.join(figures_dir, "fig1_dimension_sweeps.png")
    )
    plot_convergence(
        df_conv,
        save_path=os.path.join(figures_dir, "fig2_convergence_scaling.png")
    )
    plot_active_subspace_benchmark(
        df_act,
        save_path=os.path.join(figures_dir, "fig3_active_subspace_comparison.png")
    )
    plot_two_sided_sandwich(
        df_bach,
        save_path=os.path.join(figures_dir, "fig4_two_sided_sandwich.png")
    )

    perf_csv = os.path.join(benchmark_dir, "performance_benchmark.csv")
    if os.path.exists(perf_csv):
        from benchmark_performance import plot_performance_benchmark
        df_perf = pd.read_csv(perf_csv)
        plot_performance_benchmark(
            df_perf,
            save_path=os.path.join(figures_dir, "fig5_numba_vs_numpy_performance.png")
        )

    print("\nAll publication-grade figures successfully generated in:", figures_dir)


if __name__ == "__main__":
    generate_all_plots()
