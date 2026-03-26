"""
Analysis script for scaling law experiments.
Loads results, fits power-law curves L = A * C^(-alpha), and plots scaling curves.

Usage:
  python analyze.py [--results_dir checkpoints] [--output scaling_law.png]
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def power_law(C, A, alpha):
    """L = A * C^(-alpha)"""
    return A * np.power(C, -alpha)


def load_results(results_dir: str) -> dict:
    """Load all result files and organize by variant."""
    results = {"baseline": [], "full_attnres": [], "block_attnres": []}
    for f in sorted(os.listdir(results_dir)):
        if f.endswith("_results.pt"):
            data = torch.load(os.path.join(results_dir, f), map_location="cpu", weights_only=False)
            variant = data["variant"]
            if variant in results:
                results[variant].append(data)
    return results


def fit_and_plot(results: dict, output_path: str):
    """Fit power-law curves and generate the scaling plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = {"baseline": "#1f77b4", "full_attnres": "#ff7f0e", "block_attnres": "#2ca02c"}
    labels = {"baseline": "Baseline (PreNorm)", "full_attnres": "Full AttnRes", "block_attnres": "Block AttnRes"}
    markers = {"baseline": "o", "full_attnres": "s", "block_attnres": "^"}

    fitted_params = {}

    for variant, data_points in results.items():
        if len(data_points) < 2:
            print(f"Skipping {variant}: only {len(data_points)} data points")
            continue

        # Sort by compute
        data_points.sort(key=lambda x: x["pflops_days"])
        C = np.array([d["pflops_days"] for d in data_points])
        L = np.array([d["val_loss"] for d in data_points])

        # Scatter plot
        ax.scatter(C, L, color=colors[variant], marker=markers[variant],
                   s=80, zorder=5, label=None)

        # Fit power law
        try:
            popt, pcov = curve_fit(power_law, C, L, p0=[2.0, 0.05], maxfev=10000)
            A, alpha = popt
            fitted_params[variant] = (A, alpha)

            # Plot fitted curve
            C_fit = np.linspace(C.min() * 0.8, C.max() * 1.2, 200)
            L_fit = power_law(C_fit, A, alpha)
            label_str = f"{labels[variant]}: {A:.3f} x C^(-{alpha:.3f})"
            ax.plot(C_fit, L_fit, color=colors[variant], linewidth=2, label=label_str)
            print(f"{variant}: L = {A:.4f} * C^(-{alpha:.4f})")
        except Exception as e:
            print(f"Could not fit {variant}: {e}")
            ax.plot(C, L, color=colors[variant], linewidth=2,
                    label=f"{labels[variant]} (no fit)", marker=markers[variant])

    ax.set_xlabel("PFLOP/s-days", fontsize=13)
    ax.set_ylabel("Validation Loss", fontsize=13)
    ax.set_title("Scaling Law: Attention Residuals", fontsize=14)
    ax.set_xscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")

    # Print summary table
    print("\n" + "="*70)
    print(f"{'Variant':<20} {'A':>8} {'alpha':>8} {'Data Points':>12}")
    print("-"*70)
    for variant, data_points in results.items():
        if variant in fitted_params:
            A, alpha = fitted_params[variant]
            print(f"{variant:<20} {A:8.4f} {alpha:8.4f} {len(data_points):>12}")
    print("="*70)

    # Compute advantage at largest compute
    if "baseline" in fitted_params and "block_attnres" in fitted_params:
        A_b, alpha_b = fitted_params["baseline"]
        A_a, alpha_a = fitted_params["block_attnres"]
        # Find compute where block_attnres reaches baseline's best loss
        all_C = []
        for dp in results["baseline"]:
            all_C.append(dp["pflops_days"])
        C_max = max(all_C)
        L_baseline = power_law(C_max, A_b, alpha_b)
        L_attnres = power_law(C_max, A_a, alpha_a)
        print(f"\nAt {C_max:.4f} PFLOP/s-days:")
        print(f"  Baseline loss:     {L_baseline:.4f}")
        print(f"  Block AttnRes loss: {L_attnres:.4f}")
        if L_attnres < L_baseline:
            # Compute equivalent: find C where baseline = L_attnres
            C_equiv = (L_attnres / A_b) ** (-1.0 / alpha_b)
            advantage = C_equiv / C_max
            print(f"  Compute advantage: {advantage:.2f}x")

    return fitted_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="checkpoints")
    parser.add_argument("--output", type=str, default="scaling_law.png")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    total = sum(len(v) for v in results.values())
    print(f"Loaded {total} result files from {args.results_dir}")

    if total == 0:
        print("No results found. Run training first.")
        return

    fit_and_plot(results, args.output)


if __name__ == "__main__":
    main()
