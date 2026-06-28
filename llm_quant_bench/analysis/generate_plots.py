"""
Paper Figure Generator
실험 결과 JSON 파일들을 읽어 논문용 그래프 생성.

Usage:
    python generate_plots.py --results_dir ./results/raw --output_dir ./results/figures
"""
import os
import json
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.dpi'] = 150


def load_all_results(results_dir):
    """Load all benchmark and PPL result JSONs into a unified DataFrame."""
    records = []

    # Load benchmark results
    for fpath in glob.glob(os.path.join(results_dir, "*.json")):
        fname = os.path.basename(fpath)
        if fname.startswith("ppl_"):
            continue
        if "summary" in fname:
            continue

        with open(fpath) as f:
            data = json.load(f)

        # Handle list (multiple backends) or dict (single)
        items = data if isinstance(data, list) else [data]
        for item in items:
            framework = item.get("framework", "unknown")
            backend = item.get("backend", "unknown")
            model = item.get("model", item.get("model_dir", "unknown"))
            model_size = item.get("model_size_mb", 0)

            for bm in item.get("benchmarks", []):
                records.append({
                    "model": os.path.basename(str(model)),
                    "framework": framework,
                    "backend": backend,
                    "model_size_mb": model_size,
                    "prompt_length": bm.get("prompt_length", 0),
                    "ttft_ms": bm.get("ttft_ms_mean", bm.get("ttft_ms", None)),
                    "tokens_per_sec": bm.get("tokens_per_sec_mean", bm.get("tokens_per_sec", None)),
                    "tokens_per_sec_std": bm.get("tokens_per_sec_std", 0),
                    "peak_memory_mb": bm.get("peak_memory_mb", 0),
                })

    df = pd.DataFrame(records)

    # Load PPL results
    ppl_data = {}
    for fpath in glob.glob(os.path.join(results_dir, "ppl_*.json")):
        with open(fpath) as f:
            data = json.load(f)
        model = data.get("model", "unknown")
        ppl_data[model] = data.get("perplexity", None)

    # Merge PPL into df
    if not df.empty and ppl_data:
        df["perplexity"] = df["model"].map(ppl_data)

    return df


def fig1_pareto_frontier(df, output_dir):
    """Fig 1: Perplexity vs Tokens/sec scatter (Pareto frontier)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {"rknn-llm": "#E74C3C", "mnn": "#3498DB"}
    markers = {"npu": "o", "cpu": "s", "opencl": "D", "gpu": "D"}

    # Use prompt_length=64 data
    subset = df[df["prompt_length"] == 64].dropna(subset=["perplexity", "tokens_per_sec"])

    for _, row in subset.iterrows():
        color = colors.get(row["framework"], "gray")
        marker = markers.get(row["backend"], "x")
        label = f"{row['framework']}/{row['backend']}"
        ax.scatter(row["tokens_per_sec"], row["perplexity"],
                   c=color, marker=marker, s=100, edgecolors="black", linewidth=0.5)
        ax.annotate(row["model"][:15], (row["tokens_per_sec"], row["perplexity"]),
                    fontsize=7, ha="left", va="bottom")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C',
               markersize=10, label='RKNN-LLM (NPU)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498DB',
               markersize=10, label='MNN (CPU)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#3498DB',
               markersize=10, label='MNN (GPU)'),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_xlabel("Tokens/sec (higher is better)")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Accuracy vs Speed: Quantization Configuration Comparison")
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "fig1_pareto.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig2_model_size(df, output_dir):
    """Fig 2: Model size comparison by quantization bit-width."""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = df.drop_duplicates(subset=["model"]).sort_values("model_size_mb")
    if models.empty:
        return

    colors = []
    for m in models["model"]:
        if "q4" in m.lower() or "w4" in m.lower():
            colors.append("#3498DB")
        elif "q8" in m.lower() or "w8" in m.lower():
            colors.append("#E74C3C")
        else:
            colors.append("#95A5A6")

    bars = ax.barh(range(len(models)), models["model_size_mb"], color=colors)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models["model"], fontsize=9)
    ax.set_xlabel("Model Size (MB)")
    ax.set_title("Quantized Model Size Comparison")

    # Add value labels
    for bar, val in zip(bars, models["model_size_mb"]):
        ax.text(val + 5, bar.get_y() + bar.get_height()/2,
                f"{val:.0f} MB", va="center", fontsize=8)

    from matplotlib.patches import Patch
    legend = [Patch(facecolor="#3498DB", label="4-bit"),
              Patch(facecolor="#E74C3C", label="8-bit")]
    ax.legend(handles=legend)

    path = os.path.join(output_dir, "fig2_model_size.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig3_ttft_comparison(df, output_dir):
    """Fig 3: TTFT comparison across backends."""
    fig, ax = plt.subplots(figsize=(12, 6))

    subset = df[df["prompt_length"] == 64].dropna(subset=["ttft_ms"])
    if subset.empty:
        return

    backends = subset["backend"].unique()
    x = np.arange(len(subset["model"].unique()))
    width = 0.8 / len(backends)

    for i, backend in enumerate(sorted(backends)):
        data = subset[subset["backend"] == backend]
        ax.bar(x[:len(data)] + i * width, data["ttft_ms"], width,
               label=backend.upper())

    ax.set_xlabel("Model Configuration")
    ax.set_ylabel("TTFT (ms, lower is better)")
    ax.set_title("Time To First Token: NPU vs CPU vs GPU (prompt=64)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    path = os.path.join(output_dir, "fig3_ttft.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig4_ppl_vs_groupsize(df, output_dir):
    """Fig 4: Perplexity degradation vs group size."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract group size from model name
    subset = df.drop_duplicates(subset=["model"]).dropna(subset=["perplexity"]).copy()
    if subset.empty:
        return

    import re
    def extract_group_size(name):
        m = re.search(r'[Gg](\d+)', name)
        return int(m.group(1)) if m else 0

    subset["group_size"] = subset["model"].apply(extract_group_size)

    for framework in subset["framework"].unique():
        fw_data = subset[subset["framework"] == framework]
        for bit in ["4", "8"]:
            bit_data = fw_data[fw_data["model"].str.contains(f"q{bit}|w{bit}", case=False)]
            if len(bit_data) > 1:
                bit_data = bit_data.sort_values("group_size")
                label = f"{framework} {bit}-bit"
                ax.plot(bit_data["group_size"], bit_data["perplexity"],
                        marker="o", label=label, linewidth=2)

    ax.set_xlabel("Group Size (0 = channel-wise)")
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity vs Group/Block Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "fig4_ppl_groupsize.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig5_quant_algorithm_comparison(df, output_dir):
    """Fig 5: Quantization algorithm comparison (Direct vs AWQ vs Smooth vs HQQ)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter MNN 4-bit models with different algorithms
    subset = df[(df["framework"] == "mnn") & (df["prompt_length"] == 64)].copy()
    if subset.empty:
        return

    methods = ["direct", "awq", "smooth", "hqq"]
    method_colors = {"direct": "#3498DB", "awq": "#E74C3C", "smooth": "#2ECC71", "hqq": "#9B59B6"}

    # Extract method from model name
    def extract_method(name):
        for m in methods:
            if m in name.lower():
                return m
        return "direct"

    subset["method"] = subset["model"].apply(extract_method)
    algo_data = subset[subset["method"].isin(methods)]

    if not algo_data.empty:
        # Tokens/sec comparison
        for method in methods:
            data = algo_data[algo_data["method"] == method]
            if not data.empty:
                axes[0].bar(method.upper(), data["tokens_per_sec"].mean(),
                           color=method_colors[method])
        axes[0].set_ylabel("Tokens/sec")
        axes[0].set_title("Throughput by Quantization Algorithm")
        axes[0].grid(True, axis="y", alpha=0.3)

        # PPL comparison
        ppl_data = algo_data.dropna(subset=["perplexity"])
        for method in methods:
            data = ppl_data[ppl_data["method"] == method]
            if not data.empty:
                axes[1].bar(method.upper(), data["perplexity"].mean(),
                           color=method_colors[method])
        axes[1].set_ylabel("Perplexity")
        axes[1].set_title("Accuracy by Quantization Algorithm")
        axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("4-bit Quantization Algorithm Comparison (MNN, block=64)", fontsize=14)
    fig.tight_layout()

    path = os.path.join(output_dir, "fig5_quant_algorithms.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def fig6_radar_chart(df, output_dir):
    """Fig 6: Radar chart for top configurations."""
    subset = df[df["prompt_length"] == 64].drop_duplicates(subset=["model"]).dropna(
        subset=["tokens_per_sec", "model_size_mb"]
    )
    if len(subset) < 3:
        return

    # Select top 4 by tokens/sec
    top = subset.nlargest(4, "tokens_per_sec")

    categories = ["Speed", "Size (inv)", "Memory (inv)", "Accuracy (inv PPL)"]
    N = len(categories)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors = plt.cm.Set2(np.linspace(0, 1, len(top)))

    for i, (_, row) in enumerate(top.iterrows()):
        values = [
            row["tokens_per_sec"] / subset["tokens_per_sec"].max(),
            1 - (row["model_size_mb"] / subset["model_size_mb"].max()),
            1 - (row.get("peak_memory_mb", 0) / max(subset.get("peak_memory_mb", pd.Series([1])).max(), 1)),
            1 - (row.get("perplexity", 100) / max(subset.get("perplexity", pd.Series([100])).max(), 1)),
        ]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=row["model"][:20], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title("Top Configurations: Normalized Scores", pad=20)

    path = os.path.join(output_dir, "fig6_radar.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_summary_table(df, output_dir):
    """Generate Table 1: Full results summary as CSV and LaTeX."""
    subset = df[df["prompt_length"] == 64].copy()
    if subset.empty:
        subset = df.copy()

    cols = ["model", "framework", "backend", "model_size_mb",
            "peak_memory_mb", "ttft_ms", "tokens_per_sec", "perplexity"]
    available_cols = [c for c in cols if c in subset.columns]
    table = subset[available_cols].drop_duplicates(subset=["model", "backend"])
    table = table.sort_values(["framework", "model"])

    # CSV
    csv_path = os.path.join(output_dir, "table1_summary.csv")
    table.to_csv(csv_path, index=False)
    print(f"  Table saved: {csv_path}")

    # LaTeX
    latex_path = os.path.join(output_dir, "table1_summary.tex")
    table.to_latex(latex_path, index=False, float_format="%.2f")
    print(f"  LaTeX saved: {latex_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results_dir", type=str, default="./results/raw")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./results/figures_{timestamp}"

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading results...")
    df = load_all_results(args.results_dir)

    if df.empty:
        print("No results found. Run benchmarks first.")
        return

    print(f"Loaded {len(df)} records")
    print(f"Models: {df['model'].nunique()}")
    print(f"Frameworks: {df['framework'].unique()}")

    print("\nGenerating figures...")
    fig1_pareto_frontier(df, args.output_dir)
    fig2_model_size(df, args.output_dir)
    fig3_ttft_comparison(df, args.output_dir)
    fig4_ppl_vs_groupsize(df, args.output_dir)
    fig5_quant_algorithm_comparison(df, args.output_dir)
    fig6_radar_chart(df, args.output_dir)

    print("\nGenerating summary table...")
    generate_summary_table(df, args.output_dir)

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
