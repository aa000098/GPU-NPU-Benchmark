"""
KV Cache Memory Visualization
시퀀스 길이별 메모리 사용량 그래프.

Usage:
    python visualization/plot_kv_cache.py
"""
import os
import csv
import argparse
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


def main():
    parser = argparse.ArgumentParser(description="KV Cache Memory Visualization")
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

    kv_data = load_csv(os.path.join(args.raw_dir, "kv_cache_memory.csv"))

    print(f"{'='*60}")
    print("Generating KV Cache Memory Plots")
    print(f"{'='*60}")

    if not kv_data:
        print("  [SKIP] No KV cache data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Measured memory vs seq_len
    ax1 = axes[0]
    groups = {}
    for row in kv_data:
        key = f"{row['backend']}/{row['config_id']}"
        if key not in groups:
            groups[key] = {"x": [], "y": [], "backend": row["backend"]}
        groups[key]["x"].append(int(row["seq_len"]))
        groups[key]["y"].append(float(row["measured_memory_mb"]))

    markers = ["o", "s", "^", "D", "v", "p"]
    linestyles = ["-", "--", "-.", ":"]
    for idx, (label, g) in enumerate(sorted(groups.items())):
        backend_key = g["backend"].split("_")[0]
        color = colors.get(backend_key, f"C{idx}")
        pairs = sorted(zip(g["x"], g["y"]))
        ax1.plot([p[0] for p in pairs], [p[1] for p in pairs],
                 color=color, marker=markers[idx % len(markers)],
                 linestyle=linestyles[idx % len(linestyles)],
                 linewidth=1.5, markersize=5, label=label)

    # DRAM limit line
    dram_gb = cfg.get("system", {}).get("memory_gb", 8)
    ax1.axhline(y=dram_gb * 1024, color='red', linestyle='--', alpha=0.7, label=f'DRAM Limit ({dram_gb} GB)')

    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Sequence Length', fontsize=11)
    ax1.set_ylabel('Memory Usage (MB)', fontsize=11)
    ax1.set_title('Measured Memory vs Sequence Length', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize=8)

    # Right: Theoretical KV cache size
    ax2 = axes[1]
    theo_groups = {}
    for row in kv_data:
        key = f"{row['config_id']} (kv={row['bytes_per_kv']}B)"
        if key not in theo_groups:
            theo_groups[key] = {"x": [], "y": []}
        theo_groups[key]["x"].append(int(row["seq_len"]))
        theo_groups[key]["y"].append(float(row["theoretical_kv_mb"]))

    for idx, (label, g) in enumerate(sorted(theo_groups.items())):
        pairs = sorted(zip(g["x"], g["y"]))
        # Remove duplicates
        seen = {}
        for x, y in pairs:
            seen[x] = y
        x_vals = sorted(seen.keys())
        y_vals = [seen[x] for x in x_vals]
        ax2.plot(x_vals, y_vals, marker=markers[idx % len(markers)],
                 linestyle=linestyles[idx % len(linestyles)],
                 linewidth=1.5, markersize=5, label=label)

    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Sequence Length', fontsize=11)
    ax2.set_ylabel('Theoretical KV Cache (MB)', fontsize=11)
    ax2.set_title('Theoretical KV Cache Size', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2)
    ax2.legend(fontsize=8)

    fig.suptitle('KV Cache Memory Analysis', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"kv_cache_memory.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved kv_cache_memory.pdf/png")
    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
