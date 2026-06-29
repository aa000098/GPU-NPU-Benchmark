"""
논문용 그래프 생성 스크립트
collect_results.py가 생성한 full_results.csv를 읽어 논문 Figure 생성.

Usage:
    python analysis/generate_paper_figures.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_CSV = os.path.join(BASE_DIR, "results", "raw", "full_results.csv")


def load_results():
    """CSV에서 결과 로드."""
    results = []
    with open(RESULTS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse numeric fields safely
            for key in ["size_mb", "prefill_64_tok_s", "prefill_256_tok_s",
                        "decode_256_tok_s", "perplexity", "ttft_64_ms",
                        "ttft_256_ms", "decode_p256_tok_s"]:
                val = row.get(key)
                if val and val != "None" and val != "":
                    row[key] = float(val)
                else:
                    row[key] = None
            # Skip rows with no decode data at all
            if row["decode_256_tok_s"] is None:
                continue
            results.append(row)
    return results


def fig1_ppl_vs_speed(results, output_dir):
    """Fig 1: Perplexity vs Decode Speed scatter plot."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for r in results:
        if r["perplexity"] is None:
            continue

        color = "#3498DB" if r["quant_bit"] == "4bit" else "#E74C3C"
        marker = "o" if r["block_size"] == "channel" else "s"
        size = r["size_mb"] / 5  # Scale marker size by model size

        ax.scatter(r["decode_256_tok_s"], r["perplexity"],
                   c=color, marker=marker, s=size, edgecolors="black",
                   linewidth=0.8, zorder=3)
        ax.annotate(r["model"].replace("_direct", ""),
                    (r["decode_256_tok_s"], r["perplexity"]),
                    fontsize=8, ha="left", va="bottom",
                    xytext=(5, 3), textcoords="offset points")

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498DB",
               markersize=10, label="4-bit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E74C3C",
               markersize=10, label="8-bit"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=10, label="Channel-wise"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markersize=10, label="Block-wise"),
    ]
    ax.legend(handles=legend, loc="upper right")

    ax.set_xlabel("Decode Speed (tokens/sec, higher is better)", fontsize=12)
    ax.set_ylabel("Perplexity (lower is better)", fontsize=12)
    ax.set_title("Accuracy vs Speed: Quantization Configuration Comparison\n"
                 "Llama 3.2 1B on RK3588 (MNN, CPU 4 threads)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Lower PPL is better → top

    path = os.path.join(output_dir, "fig1_ppl_vs_speed.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig2_model_size_comparison(results, output_dir):
    """Fig 2: Model size bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = [r["model"].replace("_direct", "") for r in results]
    sizes = [r["size_mb"] for r in results]
    colors = ["#3498DB" if r["quant_bit"] == "4bit" else "#E74C3C" for r in results]

    bars = ax.bar(range(len(names)), sizes, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Model Size (MB)", fontsize=12)
    ax.set_title("Quantized Model Size Comparison", fontsize=13)

    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, val + 15,
                f"{val:.0f}", ha="center", fontsize=9)

    from matplotlib.patches import Patch
    legend = [Patch(facecolor="#3498DB", label="4-bit"),
              Patch(facecolor="#E74C3C", label="8-bit")]
    ax.legend(handles=legend)
    ax.grid(True, axis="y", alpha=0.3)

    path = os.path.join(output_dir, "fig2_model_size.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig3_prefill_decode_comparison(results, output_dir):
    """Fig 3: Prefill vs Decode speed grouped bar chart (MNN only)."""
    results = [r for r in results if r["framework"] == "mnn"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names = [r["model"].replace("_direct", "").replace("_q", "\nq") for r in results]
    x = np.arange(len(names))
    width = 0.35

    # Prefill
    pp64 = [r["prefill_64_tok_s"] for r in results]
    pp256 = [r["prefill_256_tok_s"] for r in results]
    ax1.bar(x - width/2, pp64, width, label="Prompt=64", color="#3498DB")
    ax1.bar(x + width/2, pp256, width, label="Prompt=256", color="#E67E22")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=8)
    ax1.set_ylabel("Prefill Speed (tokens/sec)")
    ax1.set_title("Prefill Performance")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    # Decode
    tg = [r["decode_256_tok_s"] for r in results]
    colors = ["#3498DB" if r["quant_bit"] == "4bit" else "#E74C3C" for r in results]
    bars = ax2.bar(x, tg, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=8)
    ax2.set_ylabel("Decode Speed (tokens/sec)")
    ax2.set_title("Decode Performance (generate 256 tokens)")
    ax2.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, tg):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                 f"{val:.1f}", ha="center", fontsize=9)

    fig.suptitle("Llama 3.2 1B on RK3588 — MNN CPU (4 threads)", fontsize=14)
    fig.tight_layout()

    path = os.path.join(output_dir, "fig3_prefill_decode.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig4_ppl_by_config(results, output_dir):
    """Fig 4: Perplexity bar chart by configuration (MNN only)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    valid = [r for r in results if r["perplexity"] is not None and r["framework"] == "mnn"]
    names = [r["model"].replace("_direct", "") for r in valid]
    ppls = [r["perplexity"] for r in valid]
    colors = ["#3498DB" if r["quant_bit"] == "4bit" else "#E74C3C" for r in valid]

    bars = ax.bar(range(len(names)), ppls, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Perplexity (lower is better)", fontsize=12)
    ax.set_title("Perplexity by Quantization Configuration\n"
                 "(wikitext-2, Llama 3.2 1B)", fontsize=13)

    for bar, val in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                f"{val:.2f}", ha="center", fontsize=9)

    from matplotlib.patches import Patch
    legend = [Patch(facecolor="#3498DB", label="4-bit"),
              Patch(facecolor="#E74C3C", label="8-bit")]
    ax.legend(handles=legend)
    ax.grid(True, axis="y", alpha=0.3)

    path = os.path.join(output_dir, "fig4_perplexity.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig5_tradeoff_radar(results, output_dir):
    """Fig 5: Radar chart comparing top configs on multiple metrics."""
    valid = [r for r in results if r["perplexity"] is not None and r["framework"] == "mnn"]
    if len(valid) < 3:
        print("  [SKIP] fig5: not enough data for radar chart")
        return

    categories = ["Decode Speed", "Prefill Speed", "Size Efficiency", "Accuracy"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(valid)))

    max_decode = max(r["decode_256_tok_s"] for r in valid)
    max_prefill = max(r["prefill_64_tok_s"] for r in valid)
    max_size = max(r["size_mb"] for r in valid)
    max_ppl = max(r["perplexity"] for r in valid)

    for i, r in enumerate(valid):
        values = [
            r["decode_256_tok_s"] / max_decode,
            r["prefill_64_tok_s"] / max_prefill,
            1 - (r["size_mb"] / max_size),  # Smaller is better
            1 - (r["perplexity"] / max_ppl),  # Lower is better
        ]
        values += values[:1]
        label = r["model"].replace("_direct", "")
        ax.plot(angles, values, "o-", linewidth=2, label=label, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.set_title("Multi-Metric Comparison (Normalized)", pad=20, fontsize=13)

    path = os.path.join(output_dir, "fig5_radar.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig6_cpu_vs_npu(results, output_dir):
    """Fig 6: MNN CPU vs RKNN-LLM NPU decode speed comparison."""
    mnn_8bit = [r for r in results if r["framework"] == "mnn" and r.get("quant_bit") == "8bit"]
    rknn = [r for r in results if r["framework"] == "rknn-llm"]

    if not rknn:
        print("  [SKIP] fig6: no RKNN-LLM data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Decode speed comparison
    labels = ["W8A8\n(channel)", "W8A8\nG128", "W8A8\nG256", "W8A8\nG512"]
    # MNN 8-bit: channel=M4, block64=M5 (closest to RKNN configs)
    mnn_speeds = [15.63, None, None, None]  # M4 channel-wise only
    rknn_speeds = [r["decode_256_tok_s"] for r in rknn]

    x = np.arange(len(labels))
    width = 0.35

    ax1.bar(x + width/2, rknn_speeds, width, label="RKNN-LLM (NPU)", color="#E74C3C", edgecolor="black", linewidth=0.5)

    # Add MNN channel-wise for comparison
    mnn_ch = [r for r in results if r["model"] == "M4_q8_ch_direct"]
    if mnn_ch:
        mnn_val = mnn_ch[0]["decode_256_tok_s"]
        ax1.bar(x[0] - width/2, mnn_val, width, label="MNN (CPU)", color="#3498DB", edgecolor="black", linewidth=0.5)

    for i, v in enumerate(rknn_speeds):
        ax1.text(x[i] + width/2, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("Decode Speed (tokens/sec)")
    ax1.set_title("Decode Speed: CPU vs NPU (W8A8)")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    # Right: TTFT comparison
    ttfts_64 = [r.get("ttft_64_ms", 0) for r in rknn]
    ttfts_256 = [r.get("ttft_256_ms", 0) for r in rknn]

    ax2.bar(x - width/2, ttfts_64, width, label="Prompt=64", color="#3498DB")
    ax2.bar(x + width/2, ttfts_256, width, label="Prompt=256", color="#E67E22")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("TTFT (ms, lower is better)")
    ax2.set_title("Time To First Token: NPU by Group Size")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("RKNN-LLM NPU Performance — Llama 3.2 1B on RK3588", fontsize=14)
    fig.tight_layout()

    path = os.path.join(output_dir, "fig6_cpu_vs_npu.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig7_cpu_gpu_npu_comparison(results, output_dir):
    """Fig 7: CPU vs GPU vs NPU 3-way decode speed comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Data: 8-bit channel-wise across 3 backends
    # MNN CPU, MNN GPU (OpenCL), RKNN-LLM NPU
    categories = ["4-bit\nchannel", "4-bit\nblock-64", "8-bit\nchannel"]
    cpu_speeds = [29.89, 24.59, 15.63]
    gpu_speeds = [9.57, 7.69, 6.53]
    npu_speeds = [19.97, None, 19.97]  # NPU only has W8A8 channel

    x = np.arange(len(categories))
    width = 0.25

    bars_cpu = ax.bar(x - width, cpu_speeds, width, label="MNN CPU (4 threads)",
                       color="#3498DB", edgecolor="black", linewidth=0.5)
    bars_gpu = ax.bar(x, gpu_speeds, width, label="MNN GPU (Mali G610 OpenCL)",
                       color="#2ECC71", edgecolor="black", linewidth=0.5)

    # NPU: only 8-bit channel available
    npu_vals = [0, 0, 19.97]
    bars_npu = ax.bar(x[2] + width, npu_vals[2], width, label="RKNN-LLM NPU (3 cores)",
                       color="#E74C3C", edgecolor="black", linewidth=0.5)

    # Add value labels
    for bars in [bars_cpu, bars_gpu]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                        f"{h:.1f}", ha="center", fontsize=9)
    ax.text(x[2] + width, 19.97 + 0.3, "20.0", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Decode Speed (tokens/sec, higher is better)", fontsize=12)
    ax.set_title("CPU vs GPU vs NPU: Decode Performance Comparison\n"
                 "Llama 3.2 1B on RK3588", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    path = os.path.join(output_dir, "fig7_cpu_gpu_npu.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_latex_table(results, output_dir):
    """LaTeX 테이블 생성."""
    path = os.path.join(output_dir, "table1_results.tex")
    with open(path, "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Quantization Benchmark Results — Llama 3.2 1B on RK3588}\n")
        f.write("\\label{tab:quant_results}\n")
        f.write("\\begin{tabular}{lcrrrr}\n\\hline\n")
        f.write("Config & Bit/Block & Size (MB) & PP64 (t/s) & TG256 (t/s) & PPL \\\\\n\\hline\n")
        for r in results:
            ppl_str = f"{r['perplexity']:.2f}" if r["perplexity"] else "—"
            pp64 = f"{r['prefill_64_tok_s']:.1f}" if r["prefill_64_tok_s"] else "—"
            tg = f"{r['decode_256_tok_s']:.2f}" if r["decode_256_tok_s"] else "—"
            block = r["block_size"] if r["block_size"] != "channel" else "ch"
            model_escaped = r['model'].replace('_', '\\_')
            line = f"{model_escaped} & {r['quant_bit']}/{block} & "
            line += f"{r['size_mb']:.0f} & {pp64} & {tg} & {ppl_str}"
            f.write(line + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  LaTeX saved: {path}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_DIR, "results", f"figures_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading results...")
    results = load_results()
    print(f"Loaded {len(results)} configurations\n")

    print("Generating figures...")
    fig1_ppl_vs_speed(results, output_dir)
    fig2_model_size_comparison(results, output_dir)
    fig3_prefill_decode_comparison(results, output_dir)
    fig4_ppl_by_config(results, output_dir)
    fig5_tradeoff_radar(results, output_dir)
    fig6_cpu_vs_npu(results, output_dir)
    fig7_cpu_gpu_npu_comparison(results, output_dir)

    print("\nGenerating LaTeX table...")
    generate_latex_table(results, output_dir)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
