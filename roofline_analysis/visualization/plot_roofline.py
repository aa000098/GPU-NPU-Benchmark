"""
Roofline Model Visualization
3 백엔드 (CPU/GPU/NPU) Roofline 그래프 생성.

Usage:
    python visualization/plot_roofline.py
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
from matplotlib.lines import Line2D
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


def draw_roofline(ax, peak_gflops, peak_bw, color, label, alpha=0.3):
    """Draw a single backend's roofline on the given axes."""
    ridge = peak_gflops / peak_bw if peak_bw > 0 else 1

    # X range (arithmetic intensity)
    x = np.logspace(-2, 4, 500)

    # Roofline: y = min(peak_gflops, peak_bw * x)
    y = np.minimum(peak_gflops, peak_bw * x)

    ax.loglog(x, y, color=color, linewidth=2, label=f"{label} ({peak_gflops:.1f} GFLOPS)")
    ax.fill_between(x, y, alpha=alpha * 0.3, color=color)

    # Mark ridge point
    ax.axvline(x=ridge, color=color, linestyle=':', alpha=0.5)
    ax.annotate(f'ridge={ridge:.2f}', xy=(ridge, peak_gflops * 0.7),
                fontsize=7, color=color, rotation=90, va='center')


def plot_combined_roofline(peaks, intensity_data, output_dir, colors):
    """Generate combined 3-backend roofline plot."""
    fig, ax = plt.subplots(figsize=(12, 8))

    backend_labels = {"cpu": "CPU (A76)", "gpu": "GPU (Mali-G610)", "npu": "NPU (RKNN)"}

    for backend, hw in peaks.items():
        if hw["peak_gflops"] > 0 and hw["peak_bandwidth_gb_s"] > 0:
            draw_roofline(ax, hw["peak_gflops"], hw["peak_bandwidth_gb_s"],
                          colors.get(backend, "#333"), backend_labels.get(backend, backend))

    # Plot arithmetic intensity points from LLM operations
    if intensity_data:
        markers = {"prefill": "o", "decode": "^"}
        quant_colors_map = {"W8A8": "#E74C3C", "W4A16": "#3498DB", "W4A8": "#2ECC71", "FP16": "#F39C12"}

        for entry in intensity_data:
            if entry["operation"] != "total":
                continue
            intensity = float(entry["intensity"])
            if intensity <= 0:
                continue

            phase = entry["phase"]
            quant = entry.get("quant", "")
            marker = markers.get(phase, "o")
            color = quant_colors_map.get(quant, "#333")

            # Y-axis: we don't have achieved GFLOPS, so place at a reasonable position
            # Use the roofline value as reference
            ax.scatter(intensity, intensity * 0.5, marker=marker, color=color,
                       s=40, edgecolors='black', linewidth=0.5, zorder=5, alpha=0.7)

    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
    ax.set_title('3-Backend Roofline Model — RK3588', fontsize=14, fontweight='bold')
    ax.set_xlim(0.01, 10000)
    ax.set_ylim(0.01, 100)
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(loc='lower right', fontsize=10)

    # Custom legend for phases
    phase_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, label='Prefill'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
               markersize=8, label='Decode'),
    ]
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.legend(handles=phase_handles, loc='upper left', fontsize=9)

    fig.tight_layout()

    for ext in ["pdf", "png"]:
        path = os.path.join(output_dir, f"roofline_combined.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved roofline_combined.pdf/png")


def plot_per_backend_roofline(peaks, output_dir, colors):
    """Generate individual roofline per backend."""
    backend_labels = {"cpu": "CPU (A76)", "gpu": "GPU (Mali-G610)", "npu": "NPU (RKNN)"}

    for backend, hw in peaks.items():
        if hw["peak_gflops"] <= 0 or hw["peak_bandwidth_gb_s"] <= 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))
        draw_roofline(ax, hw["peak_gflops"], hw["peak_bandwidth_gb_s"],
                      colors.get(backend, "#333"), backend_labels.get(backend, backend))

        ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12)
        ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
        ax.set_title(f'Roofline — {backend_labels.get(backend, backend)}', fontsize=14)
        ax.set_xlim(0.01, 10000)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend(fontsize=10)

        fig.tight_layout()
        for ext in ["pdf", "png"]:
            path = os.path.join(output_dir, f"roofline_{backend}.{ext}")
            fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved roofline_{backend}.pdf/png")


def main():
    parser = argparse.ArgumentParser(description="Roofline Visualization")
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

    # Load hardware peaks
    peaks = load_json(os.path.join(args.raw_dir, "hardware_peaks.json"))
    if not peaks:
        print("[WARN] No hardware peaks found. Run hardware benchmarks first.")
        print("       Using placeholder values for visualization.")
        peaks = {
            "cpu": {"peak_gflops": 25, "peak_bandwidth_gb_s": 15},
            "gpu": {"peak_gflops": 10, "peak_bandwidth_gb_s": 12},
            "npu": {"peak_gflops": 6, "peak_bandwidth_gb_s": 8},
        }

    # Load intensity data
    intensity_data = load_csv(os.path.join(args.raw_dir, "arithmetic_intensity.csv"))

    print(f"{'='*60}")
    print("Generating Roofline Plots")
    print(f"{'='*60}")

    plot_combined_roofline(peaks, intensity_data, output_dir, colors)
    plot_per_backend_roofline(peaks, output_dir, colors)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
