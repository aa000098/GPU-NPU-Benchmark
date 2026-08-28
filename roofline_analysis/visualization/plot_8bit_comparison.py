"""
W8A8 공정 비교: CPU vs GPU vs NPU
동일 W8A8 양자화, OpenCL 수정 후 진짜 GPU 데이터.

Usage:
    python visualization/plot_8bit_comparison.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# ── W8A8 실측 데이터 (OpenCL 수정 후) ──

DATA = {
    # MNN CPU W8A8 (M9)
    "M9_cpu": {
        "label": "CPU\n(M9 W8A8)",
        "backend": "CPU",
        "size_mb": 1185.1,
        "prefill_64": 154.62,
        "prefill_256": 143.99,
        "decode": 16.43,
        "ppl": 19.75,
    },
    # MNN GPU W8A8 (M9) — 진짜 Mali-G610 OpenCL
    "M9_gpu": {
        "label": "GPU\n(M9 W8A8)",
        "backend": "GPU",
        "size_mb": 1185.1,
        "prefill_64": 149.04,
        "prefill_256": 174.46,
        "decode": 13.66,
        "ppl": 19.75,
    },
    # RKNN NPU W8A8
    "R1": {
        "label": "NPU\n(R1 W8A8)",
        "backend": "NPU",
        "size_mb": 1704,
        "prefill_64": 1296.0,   # 64 / 0.0494s
        "prefill_256": 4971.0,  # 256 / 0.0515s
        "decode": 19.97,
        "ppl": 16.93,
    },
    "R2": {
        "label": "NPU\n(R2 G128)",
        "backend": "NPU",
        "size_mb": 1798,
        "prefill_64": 817.4,
        "prefill_256": 3184.1,
        "decode": 12.39,
        "ppl": 16.04,
    },
    "R3": {
        "label": "NPU\n(R3 G256)",
        "backend": "NPU",
        "size_mb": 1750,
        "prefill_64": 1002.3,
        "prefill_256": 3867.1,
        "decode": 15.22,
        "ppl": 16.11,
    },
    "R4": {
        "label": "NPU\n(R4 G512)",
        "backend": "NPU",
        "size_mb": 1727,
        "prefill_64": 1076.5,
        "prefill_256": 4116.4,
        "decode": 16.3,
        "ppl": 16.22,
    },
}

BACKEND_COLORS = {"CPU": "#3498DB", "GPU": "#2ECC71", "NPU": "#E74C3C"}
ALL_CONFIGS = ["M9_cpu", "M9_gpu", "R1", "R2", "R3", "R4"]
BEST_CONFIGS = ["M9_cpu", "M9_gpu", "R1"]  # channel-wise best per backend


def fig1_decode(output_dir):
    """Decode Speed 비교."""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [DATA[c]["label"] for c in ALL_CONFIGS]
    vals = [DATA[c]["decode"] for c in ALL_CONFIGS]
    colors = [BACKEND_COLORS[DATA[c]["backend"]] for c in ALL_CONFIGS]

    x = np.arange(len(ALL_CONFIGS))
    bars = ax.bar(x, vals, 0.65, color=colors, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                f'{v:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Decode Speed (tokens/sec)', fontsize=12)
    ax.set_title('W8A8 Decode Speed: CPU vs GPU vs NPU\n'
                 'Llama 3.2 1B on RK3588',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.2)
    ax.legend(handles=[Patch(facecolor=BACKEND_COLORS[b], label=b) for b in ["CPU", "GPU", "NPU"]],
              fontsize=11, loc='upper right')

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir, f"w8a8_decode.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved w8a8_decode.pdf/png")


def fig2_prefill(output_dir):
    """Prefill Speed 비교."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    labels = [DATA[c]["label"] for c in ALL_CONFIGS]
    colors = [BACKEND_COLORS[DATA[c]["backend"]] for c in ALL_CONFIGS]
    x = np.arange(len(ALL_CONFIGS))

    for ax, key, title in [(ax1, "prefill_64", "Prefill (p=64)"),
                            (ax2, "prefill_256", "Prefill (p=256)")]:
        vals = [DATA[c][key] for c in ALL_CONFIGS]
        bars = ax.bar(x, vals, 0.65, color=colors, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.01,
                    f'{v:.0f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel('Prefill Speed (tokens/sec)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.2)

    ax2.legend(handles=[Patch(facecolor=BACKEND_COLORS[b], label=b) for b in ["CPU", "GPU", "NPU"]],
               fontsize=10)

    fig.suptitle('W8A8 Prefill Speed: CPU vs GPU vs NPU\nLlama 3.2 1B on RK3588',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir, f"w8a8_prefill.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved w8a8_prefill.pdf/png")


def fig3_ppl_vs_speed(output_dir):
    """PPL vs Decode Speed scatter."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for key in ALL_CONFIGS:
        d = DATA[key]
        color = BACKEND_COLORS[d["backend"]]
        marker = {"CPU": "o", "GPU": "s", "NPU": "^"}[d["backend"]]

        ax.scatter(d["decode"], d["ppl"], c=color, marker=marker,
                   s=120, edgecolors="black", linewidth=0.8, zorder=3)
        ax.annotate(key, (d["decode"], d["ppl"]),
                    fontsize=9, ha="left", va="bottom",
                    xytext=(5, 3), textcoords="offset points")

    ax.set_xlabel('Decode Speed (tokens/sec)', fontsize=12)
    ax.set_ylabel('Perplexity (lower is better)', fontsize=12)
    ax.set_title('W8A8: Perplexity vs Decode Speed\n'
                 '(PPL: MNN vs RKNN runtime difference)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.legend(handles=[Patch(facecolor=BACKEND_COLORS[b], label=b) for b in ["CPU", "GPU", "NPU"]],
              fontsize=10)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir, f"w8a8_ppl_vs_speed.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved w8a8_ppl_vs_speed.pdf/png")


def fig4_summary(output_dir):
    """종합 4패널: best per backend (channel-wise)."""
    best = {b: DATA[c] for b, c in zip(
        ["CPU\n(M9 W8A8)", "GPU\n(M9 W8A8)", "NPU\n(R1 W8A8)"],
        BEST_CONFIGS
    )}

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    backend_order = list(best.keys())
    colors_list = [BACKEND_COLORS["CPU"], BACKEND_COLORS["GPU"], BACKEND_COLORS["NPU"]]

    metrics = [
        ("decode", "Decode Speed", "tokens/sec", 0.3),
        ("prefill_64", "Prefill (p=64)", "tokens/sec", 10),
        ("prefill_256", "Prefill (p=256)", "tokens/sec", 10),
        ("ppl", "Perplexity", "PPL (lower=better)", 0.2),
    ]

    for ax, (key, title, ylabel, offset) in zip(axes, metrics):
        vals = [best[b][key] for b in backend_order]
        bars = ax.bar(backend_order, vals, color=colors_list, edgecolor='white')
        for bar, v in zip(bars, vals):
            fmt = f'{v:.1f}' if key == "decode" else (f'{v:.2f}' if key == "ppl" else f'{v:.0f}')
            ax.text(bar.get_x() + bar.get_width()/2, v + offset,
                    fmt, ha='center', fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.2)

    fig.suptitle('W8A8 Fair Comparison: CPU vs GPU vs NPU\n'
                 'Llama 3.2 1B on RK3588 — Same W8A8, Real GPU (OpenCL fixed)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir, f"w8a8_summary.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved w8a8_summary.pdf/png")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(PROJECT_DIR, "results", f"w8a8_comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print("W8A8 Fair Comparison: CPU vs GPU vs NPU")
    print(f"  GPU: Real Mali-G610 OpenCL (fixed build)")
    print(f"{'='*60}")

    fig1_decode(output_dir)
    fig2_prefill(output_dir)
    fig3_ppl_vs_speed(output_dir)
    fig4_summary(output_dir)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
