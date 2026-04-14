"""
Analysis script for scaling law experiments.
Generates a clean two-panel figure comparing all 3 residual variants.

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
import matplotlib.ticker as ticker


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
    configs_order = ["124M", "172M", "194M", "231M", "241M", "296M", "313M", "401M", "436M", "528M"]
    variants_order = ["baseline", "full_attnres", "block_attnres"]
    variant_labels = {
        "baseline": "Baseline (PreNorm)",
        "full_attnres": "Full AttnRes",
        "block_attnres": "Block AttnRes",
    }
    colors = {
        "baseline": "#5B7FBF",
        "full_attnres": "#E8873D",
        "block_attnres": "#59B95D",
    }

    available_configs = [c for c in configs_order
                         if all((c, v) in results for v in variants_order)]
    if not available_configs:
        print("No configs with all 3 variants found.")
        return

    n = len(available_configs)
    losses = {v: [results[(c, v)]["val_loss"] for c in available_configs] for v in variants_order}
    tokens = [results[(c, "baseline")]["tokens_seen"] for c in available_configs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={"width_ratios": [1.4, 1], "wspace": 0.35})

    # ── Panel 1: grouped bars ──
    bar_w = 0.25
    x = np.arange(n)

    for i, v in enumerate(variants_order):
        offset = (i - 1) * bar_w
        bars = ax1.bar(x + offset, losses[v], bar_w,
                       label=variant_labels[v], color=colors[v],
                       edgecolor="white", linewidth=1.0, zorder=3)
        for bar, val in zip(bars, losses[v]):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.06,
                     f"{val:.2f}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold", color="#333")

    # x labels: model name on first line, token count on second
    tok_labels = []
    for c, t in zip(available_configs, tokens):
        if t >= 1e9:
            tok_labels.append(f"{c}\n({t/1e9:.1f}B tokens)")
        else:
            tok_labels.append(f"{c}\n({t/1e6:.0f}M tokens)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(tok_labels, fontsize=12)
    ax1.set_ylabel("Validation Loss", fontsize=14)
    ax1.set_title("Validation Loss by Model Size", fontsize=16, fontweight="bold", pad=12)
    ax1.legend(fontsize=12, loc="upper right", framealpha=0.9)
    ax1.grid(axis="y", alpha=0.25, zorder=0)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="y", labelsize=11)
    ax1.set_ylim(0, max(max(losses[v]) for v in variants_order) * 1.15)

    # ── Panel 2: delta bars ──
    bl = np.array(losses["baseline"])
    delta_full = bl - np.array(losses["full_attnres"])
    delta_block = bl - np.array(losses["block_attnres"])

    bar_w2 = 0.35
    bars_f = ax2.bar(x - bar_w2 / 2, delta_full, bar_w2,
                     label="Full AttnRes", color=colors["full_attnres"],
                     edgecolor="white", linewidth=1.0, zorder=3)
    bars_b = ax2.bar(x + bar_w2 / 2, delta_block, bar_w2,
                     label="Block AttnRes", color=colors["block_attnres"],
                     edgecolor="white", linewidth=1.0, zorder=3)

    for bar, val in zip(bars_f, delta_full):
        y = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 y + (0.02 if y >= 0 else -0.06),
                 f"{val:+.2f}", ha="center", va="bottom" if y >= 0 else "top",
                 fontsize=10, fontweight="bold", color=colors["full_attnres"])
    for bar, val in zip(bars_b, delta_block):
        y = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 y + (0.02 if y >= 0 else -0.06),
                 f"{val:+.2f}", ha="center", va="bottom" if y >= 0 else "top",
                 fontsize=10, fontweight="bold", color=colors["block_attnres"])

    ax2.axhline(0, color="#666", linewidth=1.2, zorder=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(available_configs, fontsize=12)
    ax2.set_ylabel("Loss Improvement vs Baseline", fontsize=14)
    ax2.set_title("AttnRes Gain Over Baseline", fontsize=16, fontweight="bold", pad=12)
    ax2.legend(fontsize=12, loc="lower right", framealpha=0.9)
    ax2.grid(axis="y", alpha=0.25, zorder=0)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="y", labelsize=11)

    # shade positive / negative regions
    ylim = ax2.get_ylim()
    ax2.axhspan(0, max(ylim[1], 0.1), alpha=0.06, color="green", zorder=0)
    ax2.axhspan(min(ylim[0], -0.1), 0, alpha=0.06, color="red", zorder=0)

    # footnote
    fig.text(0.5, 0.01,
             "Positive = AttnRes better than baseline.  Negative = baseline better.  "
             "All variants share identical hyperparameters per model size.",
             ha="center", fontsize=10, color="#666", style="italic")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.3)
    print(f"Plot saved to {output_path}")

    # ── Summary table ──
    print()
    hdr = f"{'Config':<8} {'Tokens':>10} {'Baseline':>10} {'Full':>10} {'Block':>10} {'Best Delta':>12}"
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for i, cfg in enumerate(available_configs):
        b, f_, bk = losses["baseline"][i], losses["full_attnres"][i], losses["block_attnres"][i]
        best = min(f_, bk)
        delta = b - best
        tag = "Full" if f_ < bk else "Block"
        print(f"{cfg:<8} {tokens[i]:>10,} {b:>10.4f} {f_:>10.4f} {bk:>10.4f} {delta:>+11.3f} ({tag})")
    print("=" * len(hdr))
    wins = sum(1 for i in range(n)
               if min(losses["full_attnres"][i], losses["block_attnres"][i]) < losses["baseline"][i])
    avg_f = np.mean(bl - np.array(losses["full_attnres"]))
    avg_b = np.mean(bl - np.array(losses["block_attnres"]))
    print(f"\nAttnRes wins: {wins}/{n} model sizes")
    print(f"Avg improvement -- Full: {avg_f:+.3f}, Block: {avg_b:+.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="checkpoints")
    parser.add_argument("--output", type=str, default="scaling_law.png")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} result files from {args.results_dir}")
    if not results:
        print("No results found. Run training first.")
        return
    make_plots(results, args.output)


if __name__ == "__main__":
    main()
