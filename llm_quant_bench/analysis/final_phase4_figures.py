"""
Phase 4 최종 종합 figure 생성

데이터:
- matmul_scaling_fixed_freq.txt : Shape A (K=2048) context sweep
- gqa_shapes.txt : Shape A/B/C 비교
- dispatch_overhead.txt : per-call dispatch cost

생성물:
- fig_phase4_scaling.pdf : Shape A linear fit + 이론 비교 (이미 있음)
- fig_phase4_decomposition.pdf : decode time breakdown with dispatch vs compute
- fig_phase4_shape_comparison.pdf : 3 shapes slopes + K-proportionality
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MB = os.path.join(BASE, "results", "figures_microbench")


# ── Data ──────────────────────────────────────────────────────────
# Shape A: (1, 2048) × (2048, N)
shape_A = {
    "context": [32, 64, 128, 256, 512, 1024, 2048, 4096],
    "min_us": [42.58, 53.96, 77.58, 124.54, 204.46, 391.72, 767.97, 1518.16],
}
# Shape B: (1, 512) × (512, N)
shape_B = {
    "context": [32, 64, 128, 256, 512, 1024, 2048, 4096],
    "min_us": [18.08, 21.00, 27.13, 38.79, 62.13, 108.79, 203.30, 391.43],
}
# Shape C: (1, 64) × (64, N)
shape_C = {
    "context": [32, 64, 128, 256, 512, 1024, 2048, 4096],
    "min_us": [15.17, 13.71, 16.62, 18.37, 21.29, 27.12, 39.08, 62.71],
}

# Dispatch overhead data
dispatch_data = [
    ("Tiny (1×32×16)",   21.29, 512,    0.09),
    ("Small (1×64×32)",  25.67, 2048,   0.37),
    ("Med (1×128×64)",   32.38, 8192,   1.49),
    ("Med (1×512×64)",   35.58, 32768,  5.96),
    ("FullK (1×2048×32)",35.58, 65536,  11.9),
    ("FullK (1×2048×128)",63.58,262144, 47.7),
    ("FullK (1×2048×512)",203.88,1048576,190.65),
    ("FullK (1×2048×4096)",1518.45,8388608,1525.2),
]

# Original Phase 1 measurement: NPU decode slope was 9.26 µs/tok
ORIG_DECODE_SLOPE = 9.26
ORIG_FIXED_OVERHEAD = 50.18

# Llama 3.2 1B architecture
LAYERS = 16
ATTN_MATMUL_PER_LAYER = 2  # QK^T + AV


# ── Figure 1: Shape comparison + K-proportionality ────────────────
def fig_shape_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for data, label, color in [
        (shape_A, "Shape A: K=2048 (hidden)", "#E74C3C"),
        (shape_B, "Shape B: K=512 (KV dim)",   "#2ECC71"),
        (shape_C, "Shape C: K=64 (per-head)",  "#3498DB"),
    ]:
        ctx = np.array(data["context"])
        us = np.array(data["min_us"])
        slope, inter = np.polyfit(ctx, us, 1)
        ax1.plot(ctx, us, 'o-', markersize=8, color=color,
                  label=f'{label}: {inter:.1f} + {slope*1000:.1f}ns×ctx')

    ax1.set_xscale('log')
    ax1.set_xticks([32, 64, 128, 256, 512, 1024, 2048, 4096])
    ax1.set_xticklabels(['32', '64', '128', '256', '512', '1024', '2048', '4096'])
    ax1.set_xlabel('Context Length (N)', fontsize=12)
    ax1.set_ylabel('Matmul Latency (µs)', fontsize=12)
    ax1.set_title('NPU Matmul (1, K) × (K, N): Linear scaling in N across shapes',
                   fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right panel: slope vs K proportionality
    K_values = [64, 512, 2048]
    slopes = []
    for data in [shape_C, shape_B, shape_A]:
        ctx = np.array(data["context"])
        us = np.array(data["min_us"])
        s, _ = np.polyfit(ctx, us, 1)
        slopes.append(s * 1000)  # ns/tok

    ax2.plot(K_values, slopes, 'o-', markersize=12, linewidth=2, color='#9B59B6')
    # Fit y = c × K
    c = slopes[-1] / K_values[-1]
    x_ideal = np.array([32, 2048])
    ax2.plot(x_ideal, c * x_ideal, '--', color='gray', alpha=0.7,
              label=f'Ideal y = {c*1000:.3f} pns/MAC × K')

    for k, s in zip(K_values, slopes):
        ax2.annotate(f'K={k}\n{s:.1f} ns/tok',
                      (k, s), xytext=(8, 8), textcoords='offset points', fontsize=10)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('K (matmul inner dimension)', fontsize=12)
    ax2.set_ylabel('Per-token slope (ns/tok)', fontsize=12)
    ax2.set_title('Slope scales linearly with K\n(confirms compute-bound behavior)',
                   fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, which='both', alpha=0.3)

    fig.suptitle('NPU Matmul Compute Scaling — Shape & K Dependence',
                  fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(MB, "fig_shape_comparison.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 2: Decode time decomposition ───────────────────────────
def fig_decode_decomposition():
    """Decompose measured NPU decode into compute vs dispatch components."""
    fig, ax = plt.subplots(figsize=(11, 6))

    # Measured Phase 1 NPU decode
    ctx = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    measured_ms = [50.0, 50.0, 51.0, 54.9, 55.2, 58.8, 68.5, 88.5]

    # Decomposition
    # - Fixed overhead (non-attention): ~40 ms (Q/K/V/O projection + MLP, context-invariant)
    # - Per-matmul dispatch: 25 µs × 144 matmuls = 3.6 ms (context-invariant)
    # - Attention compute: 32 matmuls × (slope_A × ctx + fixed) = context-dependent
    slope_A = 0.367  # µs per ctx-token per matmul
    inter_A = 27.4   # µs per matmul (fixed overhead from shape A)
    DISPATCH_PER_MATMUL = 25.0  # µs
    NON_ATTN_MATMULS = 144 - 32  # rough estimate: 7 non-attention matmul / layer × 16 layers
    ATTN_MATMULS = LAYERS * ATTN_MATMUL_PER_LAYER  # 32

    ctx_arr = np.array(ctx)
    # Per decode token:
    attn_compute_ms = (slope_A * ATTN_MATMULS * ctx_arr) / 1000  # context-dependent
    attn_fixed_ms = (inter_A * ATTN_MATMULS) / 1000  # ~0.88ms context-invariant
    non_attn_ms = np.full_like(ctx_arr, 40, dtype=float)  # rough baseline
    dispatch_ms = np.full_like(ctx_arr, 144 * DISPATCH_PER_MATMUL / 1000, dtype=float)

    # Stacked area
    x = np.arange(len(ctx))
    ax.bar(x, non_attn_ms, label='Non-attention (QKV proj + MLP)', color='#34495E')
    ax.bar(x, dispatch_ms, bottom=non_attn_ms,
            label=f'Dispatch overhead ({DISPATCH_PER_MATMUL:.0f}µs × 144 ops)', color='#95A5A6')
    ax.bar(x, attn_fixed_ms, bottom=non_attn_ms+dispatch_ms,
            label='Attention fixed overhead', color='#F39C12')
    ax.bar(x, attn_compute_ms, bottom=non_attn_ms+dispatch_ms+attn_fixed_ms,
            label=f'Attention compute (O(ctx), slope={slope_A:.2f}µs×32 matmul)',
            color='#E74C3C')

    # Measured overlay
    ax.plot(x, measured_ms, 'ko-', linewidth=2.5, markersize=10,
             label='Measured NPU decode', zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in ctx])
    ax.set_xlabel('Context Length', fontsize=12)
    ax.set_ylabel('Decode time per token (ms)', fontsize=12)
    ax.set_title('Phase 4: Decode Time Decomposition from Microbench\n'
                  'Attention compute O(ctx) explains context-proportional slope',
                  fontsize=13)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)

    path = os.path.join(MB, "fig_decode_decomposition.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Figure 3: Dispatch vs compute bars ────────────────────────────
def fig_dispatch_vs_compute():
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [d[0].replace("Tiny ", "").replace("Small ", "").replace("Med ", "").replace("FullK ", "") for d in dispatch_data]
    min_us = [d[1] for d in dispatch_data]
    compute_us = [d[3] for d in dispatch_data]
    dispatch_us = [max(m - c, 0) for m, c in zip(min_us, compute_us)]

    x = np.arange(len(labels))
    width = 0.7

    ax.bar(x, dispatch_us, width, label='Dispatch overhead (residual)', color='#95A5A6')
    ax.bar(x, compute_us, width, bottom=dispatch_us,
            label='Compute @ 11 GFLOPS sustained', color='#E74C3C')

    for i, (m, d, c) in enumerate(zip(min_us, dispatch_us, compute_us)):
        ax.text(i, m + 10, f"{m:.0f}", ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Matmul Latency (µs)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Phase 4-5: NPU Dispatch Overhead Isolation\n'
                  'Per-call dispatch ~20-30 µs regardless of compute volume',
                  fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', which='both', alpha=0.3)

    path = os.path.join(MB, "fig_dispatch_vs_compute.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    os.makedirs(MB, exist_ok=True)
    fig_shape_comparison()
    fig_decode_decomposition()
    fig_dispatch_vs_compute()


if __name__ == "__main__":
    main()
