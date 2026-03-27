"""
Analysis script for scaling law experiments.
Generates two plots:
  1. Grouped bar chart: val loss per model size, comparing all 3 variants
  2. Loss improvement (delta) vs. model size

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


def load_results(results_dir: str) -> dict:
    """Load all result files and organize by (config, variant)."""
    results = {}
    for f in sorted(os.listdir(results_dir)):
        if f.endswith("_results.pt"):
            data = torch.load(os.path.join(results_dir, f), map_location="cpu", weights_only=False)
            key = (data["config"], data["variant"])
            results[key] = data
    return results


def make_plots(results: dict, output_path: str):
    """Generate a two-panel figure: bar chart + delta chart."""

    # Organize data by config
    configs_order = ["124M", "172M", "231M", "313M", "401M"]
    variants_order = ["baseline", "full_attnres", "block_attnres"]
    variant_labels = {
        "baseline": "Baseline (PreNorm)",
        "full_attnres": "Full AttnRes",
        "block_attnres": "Block AttnRes",
    }
    colors = {
        "baseline": "#4878CF",
        "full_attnres": "#EE854A",
        "block_attnres": "#6ACC64",
    }

    # Filter to configs that have all 3 variants
    available_configs = []
    for cfg in configs_order:
        if all((cfg, v) in results for v in variants_order):
            available_configs.append(cfg)

    if not available_configs:
        print("No configs with all 3 variants found.")
        return

    n_configs = len(available_configs)
    n_variants = len(variants_order)

    # Extract losses and tokens
    losses = {v: [] for v in variants_order}
    tokens_info = []
    params_info = []
    for cfg in available_configs:
        for v in variants_order:
            d = results[(cfg, v)]
            losses[v].append(d["val_loss"])
        d0 = results[(cfg, "baseline")]
        tokens_info.append(d0["tokens_seen"])
        params_info.append(d0["num_params"])

    # ---- Figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [3, 2]})

    # -- Panel 1: Grouped bar chart --
    bar_width = 0.22
    x = np.arange(n_configs)

    for i, v in enumerate(variants_order):
        offset = (i - 1) * bar_width
        bars = ax1.bar(
            x + offset, losses[v], bar_width,
            label=variant_labels[v], color=colors[v],
            edgecolor="white", linewidth=0.8,
        )
        # Value labels on bars
        for bar, val in zip(bars, losses[v]):
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            )

    # X-axis labels with token counts
    xlabels = []
    for cfg, tok, par in zip(available_configs, tokens_info, params_info):
        tok_m = tok / 1e6
        par_m = par / 1e6
        xlabels.append(f"{cfg}\n({par_m:.0f}M params, {tok_m:.0f}M tok)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, fontsize=8.5)
    ax1.set_ylabel("Validation Loss", fontsize=12)
    ax1.set_title("Validation Loss by Model Size", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)

    # -- Panel 2: Improvement (delta) chart --
    baseline_losses = np.array(losses["baseline"])
    for v in ["full_attnres", "block_attnres"]:
        deltas = baseline_losses - np.array(losses[v])
        ax2.plot(x, deltas, marker="o", linewidth=2, markersize=8,
                 color=colors[v], label=variant_labels[v])
        for xi, d in zip(x, deltas):
            ax2.annotate(
                f"{d:+.2f}", (xi, d),
                textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8, fontweight="bold",
                color=colors[v],
            )

    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(available_configs, fontsize=9)
    ax2.set_xlabel("Model Size", fontsize=11)
    ax2.set_ylabel("Loss Improvement vs Baseline", fontsize=11)
    ax2.set_title("AttnRes Improvement Over Baseline", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    # Shade positive region
    ax2.axhspan(0, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 1, alpha=0.05, color="green")
    ax2.axhspan(ax2.get_ylim()[0] if ax2.get_ylim()[0] < 0 else -1, 0, alpha=0.05, color="red")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    # ---- Print summary table ----
    print()
    print("=" * 85)
    print(f"{'Config':<8} {'Tokens':>10} {'Baseline':>10} {'Full AttnRes':>14} {'Block AttnRes':>15} {'Best Delta':>12}")
    print("-" * 85)
    for i, cfg in enumerate(available_configs):
        bl = losses["baseline"][i]
        fa = losses["full_attnres"][i]
        ba = losses["block_attnres"][i]
        best = min(fa, ba)
        delta = bl - best
        winner = "Full" if fa < ba else "Block"
        print(f"{cfg:<8} {tokens_info[i]:>10,} {bl:>10.4f} {fa:>14.4f} {ba:>15.4f} {delta:>+11.4f} ({winner})")
    print("=" * 85)

    # Overall stats
    wins = sum(1 for i in range(n_configs)
               if min(losses["full_attnres"][i], losses["block_attnres"][i]) < losses["baseline"][i])
    print(f"\nAttnRes wins: {wins}/{n_configs} model sizes")
    avg_delta_full = np.mean(np.array(losses["baseline"]) - np.array(losses["full_attnres"]))
    avg_delta_block = np.mean(np.array(losses["baseline"]) - np.array(losses["block_attnres"]))
    print(f"Avg improvement — Full AttnRes: {avg_delta_full:+.4f}, Block AttnRes: {avg_delta_block:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="checkpoints")
    parser.add_argument("--output", type=str, default="scaling_law.png")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    total = len(results)
    print(f"Loaded {total} result files from {args.results_dir}")

    if total == 0:
        print("No results found. Run training first.")
        return

    make_plots(results, args.output)


if __name__ == "__main__":
    main()
