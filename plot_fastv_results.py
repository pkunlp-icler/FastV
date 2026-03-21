#!/usr/bin/env python3
"""
Plot FastV experiment results.
Reads JSON outputs from fastv_experiments.py and generates figures.

Usage:
    python plot_fastv_results.py --results_dir ./fastv_results
    python plot_fastv_results.py --results_dir ./fastv_results --output_dir ./figures
"""

import os
import json
import argparse
from pathlib import Path
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Style
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

COLORS = {
    "qwen2vl": "#2196F3",
    "llava":   "#FF5722",
}
MARKERS = {
    "qwen2vl": "o",
    "llava":   "s",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════
# Figure 1: Performance vs FLOPs Reduction  (Paper Figure 1)
# ════════════════════════════════════════════════════════════════

def plot_sweep(results_dir, output_dir):
    """Plot accuracy vs FLOPs reduction for all sweep results found."""
    sweep_files = sorted(glob(str(Path(results_dir) / "sweep_*.json")))
    if not sweep_files:
        print("No sweep results found."); return

    fig, ax = plt.subplots(figsize=(9, 6))

    for fpath in sweep_files:
        data = load_json(fpath)
        model_type = data["model_type"]
        bench = data["benchmark"]
        model_name = data["model"].split("/")[-1]

        configs = data["configs"]
        x = [c["flops_reduction"] * 100 for c in configs]  # percent
        y = [c["accuracy"] for c in configs]
        labels = [c["label"] for c in configs]

        color = COLORS.get(model_type, "#333")
        marker = MARKERS.get(model_type, "^")

        ax.plot(x, y, marker=marker, color=color, linewidth=1.5, markersize=7,
                label=f"{model_name} / {bench.upper()}", zorder=3)

        # Annotate key points
        for xi, yi, lab in zip(x, y, labels):
            if lab in ("Baseline", "K=2,R=50%", "K=2,R=90%"):
                ax.annotate(lab, (xi, yi), textcoords="offset points",
                            xytext=(5, 8), fontsize=7, color=color, alpha=0.8)

    ax.set_xlabel("FLOPs Reduction (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("FastV: Performance vs Efficiency Trade-off")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 85)

    out = Path(output_dir) / "figure1_sweep.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# Figure 1b: Grouped bar chart per K (side-by-side comparison)
# ════════════════════════════════════════════════════════════════

def plot_sweep_bars(results_dir, output_dir):
    """Bar chart: accuracy for each (K, R) config, grouped by K."""
    sweep_files = sorted(glob(str(Path(results_dir) / "sweep_*.json")))
    if not sweep_files:
        return

    for fpath in sweep_files:
        data = load_json(fpath)
        model_name = data["model"].split("/")[-1]
        bench = data["benchmark"]
        configs = data["configs"]

        labels = [c["label"] for c in configs]
        accs   = [c["accuracy"] for c in configs]
        flops  = [c["flops_ratio"] * 100 for c in configs]

        fig, ax1 = plt.subplots(figsize=(12, 5))
        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax1.bar(x - width/2, accs, width, label="Accuracy (%)",
                        color="#2196F3", alpha=0.85)
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, flops, width, label="FLOPs Ratio (%)",
                        color="#FF9800", alpha=0.65)

        ax1.set_ylabel("Accuracy (%)")
        ax2.set_ylabel("FLOPs Ratio (%)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
        ax1.set_title(f"{model_name} – {bench.upper()}: Accuracy & FLOPs by Config")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
        ax1.grid(axis="y", alpha=0.3)

        out = Path(output_dir) / f"bars_{data['model_type']}_{bench}.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"  Saved {out}")
        plt.close(fig)


# ════════════════════════════════════════════════════════════════
# Table 4: Latency comparison
# ════════════════════════════════════════════════════════════════

def plot_latency(results_dir, output_dir):
    """Latency bar chart + table printout."""
    latency_files = sorted(glob(str(Path(results_dir) / "latency_*.json")))
    if not latency_files:
        print("No latency results found."); return

    for fpath in latency_files:
        data = load_json(fpath)
        model_name = data["model"].split("/")[-1]
        configs = data["configs"]

        labels   = [c["label"] for c in configs]
        latency  = [c["avg_latency_s"] for c in configs]
        accuracy = [c["accuracy"] for c in configs]
        memory   = [c["peak_memory_mb"] for c in configs]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Latency
        bars = axes[0].barh(labels, latency, color="#2196F3", alpha=0.85)
        axes[0].set_xlabel("Avg Latency (s/sample)")
        axes[0].set_title("Latency")
        for bar, val in zip(bars, latency):
            axes[0].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                         f"{val:.3f}s", va="center", fontsize=9)

        # Accuracy
        bars = axes[1].barh(labels, accuracy, color="#4CAF50", alpha=0.85)
        axes[1].set_xlabel("Accuracy (%)")
        axes[1].set_title("Accuracy")
        for bar, val in zip(bars, accuracy):
            axes[1].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                         f"{val:.1f}%", va="center", fontsize=9)

        # Memory
        bars = axes[2].barh(labels, memory, color="#FF9800", alpha=0.85)
        axes[2].set_xlabel("Peak Memory (MB)")
        axes[2].set_title("GPU Memory")
        for bar, val in zip(bars, memory):
            axes[2].text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                         f"{val:.0f}", va="center", fontsize=9)

        fig.suptitle(f"{model_name} – Latency Comparison", fontsize=14)
        fig.tight_layout()

        out = Path(output_dir) / f"latency_{data['model_type']}.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"  Saved {out}")
        plt.close(fig)


# ════════════════════════════════════════════════════════════════
# Figure 3: Attention Pattern per Layer  (Paper Figure 3 / §3)
# ════════════════════════════════════════════════════════════════

def plot_attention(results_dir, output_dir):
    """Per-layer attention allocation and efficiency."""
    attn_files = sorted(glob(str(Path(results_dir) / "attention_*.json")))
    if not attn_files:
        print("No attention results found."); return

    cat_colors = {
        "pre_image":  "#4CAF50",
        "image":      "#2196F3",
        "post_image": "#FF5722",
    }
    cat_labels = {
        "pre_image":  "System / Pre-Image",
        "image":      "Image Tokens",
        "post_image": "Instruction / Post-Image",
    }

    for fpath in attn_files:
        data = load_json(fpath)
        model_name = data["model"].split("/")[-1]
        num_layers = data["num_layers"]
        layers = np.arange(num_layers)

        # ── Figure A: Attention Allocation (stacked area) ──
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        alloc = data["attention_allocation"]
        bottom = np.zeros(num_layers)
        for cat in ["pre_image", "image", "post_image"]:
            vals = np.array(alloc[cat])
            ax1.fill_between(layers, bottom, bottom + vals,
                             alpha=0.7, color=cat_colors[cat],
                             label=cat_labels[cat])
            bottom += vals

        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Attention Allocation (last token)")
        ax1.set_title(f"Attention Allocation per Layer")
        ax1.legend(loc="upper right")
        ax1.set_xlim(0, num_layers - 1)
        ax1.set_ylim(0, 1.05)
        ax1.grid(axis="y", alpha=0.3)

        # ── Figure B: Attention Efficiency (line plot, log scale) ──
        eff = data["attention_efficiency"]
        for cat in ["pre_image", "image", "post_image"]:
            vals = np.array(eff[cat])
            vals = np.clip(vals, 1e-8, None)  # avoid log(0)
            ax2.plot(layers, vals, color=cat_colors[cat],
                     label=cat_labels[cat], linewidth=2)

        ax2.set_xlabel("Layer Index")
        ax2.set_ylabel("Attention Efficiency (per token, log scale)")
        ax2.set_title(f"Attention Efficiency per Layer")
        ax2.set_yscale("log")
        ax2.legend()
        ax2.set_xlim(0, num_layers - 1)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"{model_name} – Attention Analysis "
                     f"({data['num_samples']} samples)", fontsize=14)
        fig.tight_layout()

        out = Path(output_dir) / f"attention_{data['model_type']}.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"  Saved {out}")
        plt.close(fig)

        # ── Figure C: Image attention ratio (single line) ──
        fig, ax = plt.subplots(figsize=(8, 4))
        img_alloc = np.array(alloc["image"])
        ax.plot(layers, img_alloc * 100, color="#2196F3", linewidth=2.5,
                marker="o", markersize=3)
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5,
                    label="50% line")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Attention to Image Tokens (%)")
        ax.set_title(f"{model_name} – Image Token Attention by Layer")
        ax.legend()
        ax.set_xlim(0, num_layers - 1)
        ax.grid(True, alpha=0.3)

        out = Path(output_dir) / f"attention_image_only_{data['model_type']}.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"  Saved {out}")
        plt.close(fig)


# ════════════════════════════════════════════════════════════════
# Cross-model comparison (if both models available)
# ════════════════════════════════════════════════════════════════

def plot_cross_model(results_dir, output_dir):
    """Compare LLaVA and Qwen2-VL attention patterns side by side."""
    attn_qwen = Path(results_dir) / "attention_qwen2vl.json"
    attn_llava = Path(results_dir) / "attention_llava.json"

    if not (attn_qwen.exists() and attn_llava.exists()):
        return  # need both

    data_q = load_json(attn_qwen)
    data_l = load_json(attn_llava)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, data, title in [
        (ax1, data_q, data_q["model"].split("/")[-1]),
        (ax2, data_l, data_l["model"].split("/")[-1]),
    ]:
        layers = np.arange(data["num_layers"])
        alloc = data["attention_allocation"]
        bottom = np.zeros(data["num_layers"])
        colors = {"pre_image": "#4CAF50", "image": "#2196F3",
                  "post_image": "#FF5722"}
        names = {"pre_image": "Pre-Image", "image": "Image",
                 "post_image": "Post-Image"}
        for cat in ["pre_image", "image", "post_image"]:
            vals = np.array(alloc[cat])
            ax.fill_between(layers, bottom, bottom + vals,
                            alpha=0.7, color=colors[cat], label=names[cat])
            bottom += vals
        ax.set_xlabel("Layer Index")
        ax.set_title(title)
        ax.set_xlim(0, data["num_layers"] - 1)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    ax1.set_ylabel("Attention Allocation")
    ax1.legend(loc="upper right")
    fig.suptitle("Cross-Model Attention Pattern Comparison", fontsize=14)
    fig.tight_layout()

    out = Path(output_dir) / "attention_cross_model.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# Ablation: Random vs Attention Pruning  (Paper Table 7)
# ════════════════════════════════════════════════════════════════

def plot_ablation(results_dir, output_dir):
    """Grouped bar chart comparing attention-ranked vs random pruning."""
    ablation_files = sorted(glob(str(Path(results_dir) / "ablation_*.json")))
    if not ablation_files:
        print("No ablation results found."); return

    for fpath in ablation_files:
        data = load_json(fpath)
        model_name = data["model"].split("/")[-1]
        bench = data["benchmark"]
        configs = data["configs"]

        # ── Figure A: Paired bar chart (attention vs random) ──
        # Group configs by (K, R_prune) and separate attn vs random.
        # Baseline is standalone; the rest come in pairs.
        baseline = [c for c in configs if c["R_prune"] == 0.0]
        paired = [c for c in configs if c["R_prune"] > 0.0]

        # Build groups: each group is a (K, R_prune) pair
        groups = {}
        for c in paired:
            key = (c["K"], c["R_prune"])
            groups.setdefault(key, {})
            kind = "random" if c["random_pruning"] else "attn"
            groups[key][kind] = c

        group_labels = []
        attn_accs = []
        rand_accs = []
        for (K, R), pair in sorted(groups.items()):
            group_labels.append(f"K={K}, R={int(R*100)}%")
            attn_accs.append(pair.get("attn", {}).get("accuracy", 0))
            rand_accs.append(pair.get("random", {}).get("accuracy", 0))

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(group_labels))
        width = 0.35

        bars_attn = ax.bar(x - width/2, attn_accs, width,
                           label="Attention-ranked", color="#2196F3", alpha=0.85)
        bars_rand = ax.bar(x + width/2, rand_accs, width,
                           label="Random", color="#FF9800", alpha=0.75)

        # Draw baseline as a horizontal line
        if baseline:
            bl_acc = baseline[0]["accuracy"]
            ax.axhline(y=bl_acc, color="#4CAF50", linestyle="--", linewidth=1.5,
                       label=f"Baseline ({bl_acc:.1f}%)", zorder=1)

        # Value labels on bars
        for bar, val in zip(bars_attn, attn_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        for bar, val in zip(bars_rand, rand_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=10)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{model_name} – {bench.upper()}: Attention vs Random Pruning")
        ax.legend(loc="lower left")
        ax.grid(axis="y", alpha=0.3)

        out = Path(output_dir) / f"ablation_{data['model_type']}_{bench}.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"  Saved {out}")
        plt.close(fig)

        # ── Figure B: Delta plot (accuracy drop from baseline) ──
        if baseline:
            bl_acc = baseline[0]["accuracy"]
            fig, ax = plt.subplots(figsize=(10, 5))

            attn_deltas = [a - bl_acc for a in attn_accs]
            rand_deltas = [r - bl_acc for r in rand_accs]

            bars_a = ax.bar(x - width/2, attn_deltas, width,
                            label="Attention-ranked", color="#2196F3", alpha=0.85)
            bars_r = ax.bar(x + width/2, rand_deltas, width,
                            label="Random", color="#FF9800", alpha=0.75)

            ax.axhline(y=0, color="black", linewidth=0.8)

            for bar, val in zip(bars_a, attn_deltas):
                offset = -0.5 if val < 0 else 0.3
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + offset,
                        f"{val:+.1f}", ha="center", va="bottom" if val >= 0 else "top",
                        fontsize=8)
            for bar, val in zip(bars_r, rand_deltas):
                offset = -0.5 if val < 0 else 0.3
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + offset,
                        f"{val:+.1f}", ha="center", va="bottom" if val >= 0 else "top",
                        fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels(group_labels, fontsize=10)
            ax.set_ylabel("Accuracy Change from Baseline (pp)")
            ax.set_title(f"{model_name} – {bench.upper()}: Accuracy Drop by Pruning Strategy")
            ax.legend(loc="lower left")
            ax.grid(axis="y", alpha=0.3)

            out = Path(output_dir) / f"ablation_delta_{data['model_type']}_{bench}.png"
            fig.savefig(out, bbox_inches="tight")
            print(f"  Saved {out}")
            plt.close(fig)


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Plot FastV results")
    parser.add_argument("--results_dir", default="./fastv_results")
    parser.add_argument("--output_dir", default="./fastv_figures")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Reading results from: {args.results_dir}")
    print(f"Saving figures to:    {args.output_dir}\n")

    plot_sweep(args.results_dir, args.output_dir)
    plot_sweep_bars(args.results_dir, args.output_dir)
    plot_latency(args.results_dir, args.output_dir)
    plot_attention(args.results_dir, args.output_dir)
    plot_cross_model(args.results_dir, args.output_dir)
    plot_ablation(args.results_dir, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()