"""
Crossover Line Plots
시퀀스 길이별 throughput 라인 + 교차점 표시.

Usage:
    python visualization/plot_crossover.py
"""
import os
import csv
import json
import argparse
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, "config", "roofline_config.yaml")


def load_config(config_path=None):
    path = config_path or DEFAULT_CONFIG
    with open(path) as f:
        return yaml.safe_load(f)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_throughput_curves(data, phase, x_col, y_col, crossover_data,
                           output_dir, colors, title_suffix=""):
    """Plot throughput vs seq_len for all backends."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Group by (backend, config_id)
    groups = {}
    for row in data:
        key = (row["backend"], row["config_id"])
        if key not in groups:
            groups[key] = {"x": [], "y": []}
        x = float(row[x_col])
        y = float(row[y_col])
        if y > 0:
            groups[key]["x"].append(x)
            groups[key]["y"].append(y)

    backend_color_map = {}
    linestyles = ["-", "--", "-.", ":"]
    markers_list = ["o", "s", "^", "D", "v", "p"]

    idx = 0
    for (backend, config_id), g in sorted(groups.items()):
        if not g["x"]:
            continue

        pairs = sorted(zip(g["x"], g["y"]))
        x_sorted = [p[0] for p in pairs]
        y_sorted = [p[1] for p in pairs]

        backend_key = backend.split("_")[0]
        color = colors.get(backend_key, f"C{idx}")
        ls = linestyles[idx % len(linestyles)]
        marker = markers_list[idx % len(markers_list)]

        label = f"{backend}/{config_id}"
        ax.plot(x_sorted, y_sorted, color=color, linestyle=ls, marker=marker,
                markersize=6, linewidth=1.5, label=label)
        idx += 1

    # Mark crossover points
    phase_crossovers = [c for c in crossover_data if c.get("phase") == phase]
    for c in phase_crossovers:
        cx = float(c["crossover_seq_len"])
        cy = float(c["crossover_throughput"])
        ax.axvline(x=cx, color='gray', linestyle=':', alpha=0.5)
        ax.scatter(cx, cy, color='red', s=100, marker='x', zorder=10, linewidths=2)
        ax.annotate(f'{cx:.0f}', xy=(cx, cy), xytext=(5, 10),
                    textcoords='offset points', fontsize=8, color='red')

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Throughput (tok/s)', fontsize=12)
    ax.set_title(f'{phase.capitalize()} Throughput vs Sequence Length{title_suffix}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(fontsize=9, loc='best')

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"crossover_{phase}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved crossover_{phase}.pdf/png")


def main():
    parser = argparse.ArgumentParser(description="Crossover Visualization")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--raw_dir", type=str,
                        default=os.path.join(PROJECT_DIR, "results", "raw"))
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    colors = cfg.get("colors", {"cpu": "#3498DB", "gpu": "#2ECC71", "npu": "#E74C3C"})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(PROJECT_DIR, "results", f"figures_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    prefill_data = load_csv(os.path.join(args.raw_dir, "prefill_sweep.csv"))
    decode_data = load_csv(os.path.join(args.raw_dir, "decode_sweep.csv"))
    crossover_data = load_csv(os.path.join(args.raw_dir, "crossover_points.csv"))

    print(f"{'='*60}")
    print("Generating Crossover Plots")
    print(f"{'='*60}")

    if prefill_data:
        plot_throughput_curves(prefill_data, "prefill", "seq_len", "prefill_tok_s",
                               crossover_data, output_dir, colors)
    else:
        print("  [SKIP] No prefill data")

    if decode_data:
        plot_throughput_curves(decode_data, "decode", "context_len", "decode_tok_s",
                               crossover_data, output_dir, colors)
    else:
        print("  [SKIP] No decode data")

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
