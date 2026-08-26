"""
Optimal Backend Heatmap
(시퀀스 길이 × 양자화 설정)에서 최적 백엔드를 색상으로 표시.

Usage:
    python visualization/plot_heatmap.py
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
from matplotlib.colors import ListedColormap
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


def plot_optimal_heatmap(data, phase, output_dir, colors):
    """Plot heatmap of optimal backend for a given phase."""
    phase_data = [r for r in data if r["phase"] == phase]
    if not phase_data:
        print(f"  [SKIP] No {phase} data for heatmap")
        return

    # Get unique seq_lens and config_ids
    seq_lens = sorted(set(int(r["seq_len"]) for r in phase_data))
    config_ids = sorted(set(r["config_id"] for r in phase_data))

    if not seq_lens or not config_ids:
        return

    # Build grid
    backend_names = sorted(set(r["best_backend"] for r in phase_data))
    backend_to_idx = {b: i for i, b in enumerate(backend_names)}

    grid = np.full((len(config_ids), len(seq_lens)), -1, dtype=int)
    throughput_grid = np.full((len(config_ids), len(seq_lens)), 0.0)

    for r in phase_data:
        i = config_ids.index(r["config_id"])
        j = seq_lens.index(int(r["seq_len"]))
        grid[i, j] = backend_to_idx.get(r["best_backend"], -1)
        throughput_grid[i, j] = float(r["best_throughput"])

    # Color map
    backend_colors = []
    for b in backend_names:
        key = b.split("_")[0]
        backend_colors.append(colors.get(key, "#CCCCCC"))
    cmap = ListedColormap(backend_colors)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(grid, cmap=cmap, aspect='auto', vmin=0, vmax=len(backend_names) - 1)

    # Annotate with throughput values
    for i in range(len(config_ids)):
        for j in range(len(seq_lens)):
            val = throughput_grid[i, j]
            if val > 0:
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=7, color='white', fontweight='bold')

    ax.set_xticks(range(len(seq_lens)))
    ax.set_xticklabels(seq_lens, fontsize=9)
    ax.set_yticks(range(len(config_ids)))
    ax.set_yticklabels(config_ids, fontsize=9)
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('Quantization Config', fontsize=12)
    ax.set_title(f'Optimal Backend — {phase.capitalize()} (tok/s)', fontsize=14, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=backend_colors[i], label=backend_names[i])
                       for i in range(len(backend_names))]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"heatmap_optimal_{phase}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap_optimal_{phase}.pdf/png")


def main():
    parser = argparse.ArgumentParser(description="Optimal Backend Heatmap")
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

    optimal_data = load_csv(os.path.join(args.raw_dir, "optimal_backend_map.csv"))

    print(f"{'='*60}")
    print("Generating Optimal Backend Heatmaps")
    print(f"{'='*60}")

    if not optimal_data:
        print("  [SKIP] No optimal map data (run analysis first)")
        return

    plot_optimal_heatmap(optimal_data, "prefill", output_dir, colors)
    plot_optimal_heatmap(optimal_data, "decode", output_dir, colors)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
