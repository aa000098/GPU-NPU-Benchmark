"""
Summary Multi-Panel Publication Figure
논문용 6-패널 종합 figure 생성.

Layout (2x3):
  (0,0) Combined Roofline    (0,1) Prefill Crossover    (0,2) Decode Crossover
  (1,0) Heatmap (Prefill)    (1,1) Heatmap (Decode)     (1,2) KV Cache Memory

Usage:
    python visualization/plot_summary.py
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
from matplotlib.patches import Patch
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


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def draw_mini_roofline(ax, peaks, colors):
    """Draw combined roofline in a subplot."""
    labels = {"cpu": "CPU", "gpu": "GPU", "npu": "NPU"}
    for backend, hw in peaks.items():
        pg = hw.get("peak_gflops", 0)
        pb = hw.get("peak_bandwidth_gb_s", 0)
        if pg <= 0 or pb <= 0:
            continue
        x = np.logspace(-2, 4, 300)
        y = np.minimum(pg, pb * x)
        ax.loglog(x, y, color=colors.get(backend, "#333"), linewidth=1.5,
                  label=f'{labels.get(backend, backend)} ({pg:.0f} GF)')
    ax.set_xlabel('FLOP/Byte', fontsize=8)
    ax.set_ylabel('GFLOPS', fontsize=8)
    ax.set_title('Roofline Model', fontsize=9, fontweight='bold')
    ax.set_xlim(0.01, 10000)
    ax.grid(True, which="both", ls="-", alpha=0.15)
    ax.legend(fontsize=7, loc='lower right')
    ax.tick_params(labelsize=7)


def draw_crossover_panel(ax, data, x_col, y_col, crossover_data, phase, colors):
    """Draw throughput curves in a subplot."""
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

    markers = ["o", "s", "^", "D", "v"]
    for idx, (key, g) in enumerate(sorted(groups.items())):
        if not g["x"]:
            continue
        pairs = sorted(zip(g["x"], g["y"]))
        bk = key[0].split("_")[0]
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                color=colors.get(bk, f"C{idx}"),
                marker=markers[idx % len(markers)], markersize=4,
                linewidth=1.2, label=f"{key[0]}/{key[1]}")

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Sequence Length', fontsize=8)
    ax.set_ylabel('tok/s', fontsize=8)
    ax.set_title(f'{phase.capitalize()} Throughput', fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=6, loc='best')
    ax.tick_params(labelsize=7)


def draw_heatmap_panel(ax, data, phase, colors):
    """Draw optimal backend heatmap in a subplot."""
    phase_data = [r for r in data if r["phase"] == phase]
    if not phase_data:
        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', fontsize=10)
        ax.set_title(f'{phase.capitalize()} Heatmap', fontsize=9)
        return

    seq_lens = sorted(set(int(r["seq_len"]) for r in phase_data))
    config_ids = sorted(set(r["config_id"] for r in phase_data))
    backend_names = sorted(set(r["best_backend"] for r in phase_data))
    backend_to_idx = {b: i for i, b in enumerate(backend_names)}

    grid = np.full((len(config_ids), len(seq_lens)), -1, dtype=int)
    for r in phase_data:
        i = config_ids.index(r["config_id"])
        j = seq_lens.index(int(r["seq_len"]))
        grid[i, j] = backend_to_idx.get(r["best_backend"], -1)

    bc = [colors.get(b.split("_")[0], "#CCC") for b in backend_names]
    cmap = ListedColormap(bc)
    ax.imshow(grid, cmap=cmap, aspect='auto', vmin=0, vmax=max(len(backend_names)-1, 0))

    ax.set_xticks(range(len(seq_lens)))
    ax.set_xticklabels(seq_lens, fontsize=6, rotation=45)
    ax.set_yticks(range(len(config_ids)))
    ax.set_yticklabels(config_ids, fontsize=7)
    ax.set_title(f'{phase.capitalize()} Optimal', fontsize=9, fontweight='bold')

    legend_elements = [Patch(facecolor=bc[i], label=backend_names[i]) for i in range(len(backend_names))]
    ax.legend(handles=legend_elements, fontsize=5, loc='upper right')


def draw_kv_panel(ax, data, colors):
    """Draw KV cache memory in a subplot."""
    if not data:
        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', fontsize=10)
        ax.set_title('KV Cache', fontsize=9)
        return

    groups = {}
    for row in data:
        key = f"{row['backend']}/{row['config_id']}"
        if key not in groups:
            groups[key] = {"x": [], "y": [], "backend": row["backend"]}
        groups[key]["x"].append(int(row["seq_len"]))
        groups[key]["y"].append(float(row["measured_memory_mb"]))

    markers = ["o", "s", "^", "D"]
    for idx, (label, g) in enumerate(sorted(groups.items())):
        bk = g["backend"].split("_")[0]
        pairs = sorted(zip(g["x"], g["y"]))
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                color=colors.get(bk, f"C{idx}"),
                marker=markers[idx % len(markers)], markersize=4,
                linewidth=1.2, label=label)

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Seq Length', fontsize=8)
    ax.set_ylabel('Memory (MB)', fontsize=8)
    ax.set_title('KV Cache Memory', fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)


def main():
    parser = argparse.ArgumentParser(description="Summary Publication Figure")
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

    # Load all data
    peaks = load_json(os.path.join(args.raw_dir, "hardware_peaks.json"))
    prefill = load_csv(os.path.join(args.raw_dir, "prefill_sweep.csv"))
    decode = load_csv(os.path.join(args.raw_dir, "decode_sweep.csv"))
    crossover = load_csv(os.path.join(args.raw_dir, "crossover_points.csv"))
    optimal = load_csv(os.path.join(args.raw_dir, "optimal_backend_map.csv"))
    kv = load_csv(os.path.join(args.raw_dir, "kv_cache_memory.csv"))

    if not peaks:
        peaks = {
            "cpu": {"peak_gflops": 25, "peak_bandwidth_gb_s": 15},
            "gpu": {"peak_gflops": 10, "peak_bandwidth_gb_s": 12},
            "npu": {"peak_gflops": 6, "peak_bandwidth_gb_s": 8},
        }

    print(f"{'='*60}")
    print("Generating Summary Publication Figure")
    print(f"{'='*60}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (0,0) Roofline
    draw_mini_roofline(axes[0, 0], peaks, colors)

    # (0,1) Prefill crossover
    if prefill:
        draw_crossover_panel(axes[0, 1], prefill, "seq_len", "prefill_tok_s",
                              crossover, "prefill", colors)
    else:
        axes[0, 1].text(0.5, 0.5, 'No Prefill Data', transform=axes[0, 1].transAxes, ha='center')
        axes[0, 1].set_title('Prefill Throughput', fontsize=9)

    # (0,2) Decode crossover
    if decode:
        draw_crossover_panel(axes[0, 2], decode, "context_len", "decode_tok_s",
                              crossover, "decode", colors)
    else:
        axes[0, 2].text(0.5, 0.5, 'No Decode Data', transform=axes[0, 2].transAxes, ha='center')
        axes[0, 2].set_title('Decode Throughput', fontsize=9)

    # (1,0) Prefill heatmap
    draw_heatmap_panel(axes[1, 0], optimal, "prefill", colors)

    # (1,1) Decode heatmap
    draw_heatmap_panel(axes[1, 1], optimal, "decode", colors)

    # (1,2) KV cache
    draw_kv_panel(axes[1, 2], kv, colors)

    fig.suptitle('3-Backend Roofline Analysis — RK3588 LLM Inference',
                 fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"fig_roofline_summary.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_roofline_summary.pdf/png")
    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
