"""
NPU Decode Overhead Decomposition
=================================

NPU decode time = fixed_overhead + weights_read + f(context)

목표: f(context)의 형태를 실측으로 확인하여, KV bandwidth 모델과 비교.
  - 만약 f(context) ≈ KV / BW 이면 KV bandwidth 가설 지지
  - 만약 f(context) >> KV / BW 이면 scheduling overhead 가설 지지 (Option B thesis)

방법:
  여러 context 길이에서 decode time을 측정하고,
  linear regression으로 intercept(fixed) + slope(context-proportional) 분리.

Usage:
    tmux new -s npu_overhead 'bash scripts/run_npu_overhead.sh'
    python analysis/measure_npu_overhead.py --input_csv <logged_data>
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LLAMA_32_1B = {
    "num_hidden_layers": 16,
    "num_key_value_heads": 8,
    "head_dim": 64,
}


def kv_bytes_per_token(arch=LLAMA_32_1B, kv_bits=16):
    """KV bytes added per context token (write) and per step (read)."""
    return (
        2  # K + V
        * arch["num_hidden_layers"]
        * arch["num_key_value_heads"]
        * arch["head_dim"]
        * (kv_bits // 8)
    )


def analyze_npu_overhead(context_lens, decode_ms, output_dir):
    """Linear regression: decode_ms = a + b * context_len"""
    ctx = np.array(context_lens, dtype=np.float64)
    ms = np.array(decode_ms, dtype=np.float64)

    # Fit linear model
    coef = np.polyfit(ctx, ms, 1)
    slope, intercept = coef[0], coef[1]

    print(f"\n=== NPU Decode Time Linear Fit ===")
    print(f"decode_ms ≈ {intercept:.2f} + {slope:.6f} × context_len")
    print(f"  Fixed overhead: {intercept:.2f} ms/step")
    print(f"  Context-proportional slope: {slope*1000:.3f} us/token")

    # Compare with KV bandwidth model
    kv_b_per_token = kv_bytes_per_token()
    print(f"\n=== KV Bandwidth Hypothesis Check ===")
    print(f"  KV bytes per context token: {kv_b_per_token} B ({kv_b_per_token/1024:.2f} KB)")

    for bw_gbs in [10, 20, 30, 50]:
        kv_us_per_token_at_bw = (kv_b_per_token / (bw_gbs * 1e9)) * 1e6
        ratio = (slope * 1000) / kv_us_per_token_at_bw if kv_us_per_token_at_bw > 0 else 0
        print(f"  At BW={bw_gbs} GB/s: pure KV read predicts {kv_us_per_token_at_bw:.3f} us/token")
        print(f"    → Measured slope is {ratio:.1f}x higher than pure KV bandwidth model")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: linear fit
    ax1.scatter(ctx, ms, s=100, c='red', edgecolors='black', zorder=5,
                 label='NPU measured')
    x_fit = np.linspace(0, max(ctx) * 1.05, 100)
    y_fit = intercept + slope * x_fit
    ax1.plot(x_fit, y_fit, 'k--', linewidth=2,
              label=f'Linear fit: {intercept:.1f} + {slope*1000:.2f}us × ctx')

    # Overlay pure KV bandwidth predictions
    for bw_gbs, color in [(10, '#90CAF9'), (30, '#64B5F6'), (50, '#1976D2')]:
        kv_us_per_token = (kv_b_per_token / (bw_gbs * 1e9)) * 1e6
        y_kv = intercept + (kv_us_per_token / 1000) * x_fit
        ax1.plot(x_fit, y_kv, ':', color=color, alpha=0.7,
                  label=f'Pure KV@{bw_gbs}GB/s (slope={kv_us_per_token:.2f}us)')

    ax1.set_xlabel('Context Length', fontsize=12)
    ax1.set_ylabel('Decode time per token (ms)', fontsize=12)
    ax1.set_title('NPU Decode Time: Context-Proportional Overhead\n'
                  'measured slope vs pure KV bandwidth predictions', fontsize=12)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Panel 2: excess over pure KV model
    excess_at_bw = {}
    for bw_gbs in [30]:
        kv_us_per_token = (kv_b_per_token / (bw_gbs * 1e9)) * 1e6
        excess = (slope * 1000) - kv_us_per_token
        excess_at_bw[bw_gbs] = excess

        # Plot stacked bars
        pure_kv_contrib = (kv_us_per_token / 1000) * ctx
        excess_contrib = (excess / 1000) * ctx
        fixed_contrib = np.full_like(ctx, intercept)

        width = np.array([max(ctx) * 0.05] * len(ctx))
        ax2.bar(ctx, fixed_contrib, width, label='Fixed overhead', color='#95A5A6')
        ax2.bar(ctx, pure_kv_contrib, width, bottom=fixed_contrib,
                 label=f'Pure KV read @ {bw_gbs}GB/s', color='#E74C3C')
        ax2.bar(ctx, excess_contrib, width,
                 bottom=fixed_contrib + pure_kv_contrib,
                 label='Excess overhead\n(scheduling/access pattern)', color='#9B59B6')
        ax2.plot(ctx, ms, 'o-', color='black', linewidth=2, markersize=8,
                  label='Measured total', zorder=5)

    ax2.set_xlabel('Context Length', fontsize=12)
    ax2.set_ylabel('Decode time per token (ms)', fontsize=12)
    ax2.set_title('NPU Decode Time Decomposition\n'
                  '(Fixed + Pure KV BW + Excess)', fontsize=12)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Testing the "KV Bandwidth is the Bottleneck" Hypothesis — NPU',
                  fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, "npu_overhead_decomposition.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")

    return {
        "fixed_overhead_ms": intercept,
        "slope_us_per_token": slope * 1000,
        "kv_bw_predicted_slope_us_at_30gbs": (kv_b_per_token / 30e9) * 1e6,
        "excess_ratio": (slope * 1000) / ((kv_b_per_token / 30e9) * 1e6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                         default="./results/figures_roofline")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Measured NPU decode (from our benchmarks)
    context_lens = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    npu_tok_s = [20.0, 20.0, 19.6, 18.2, 18.1, 17.0, 14.6, 11.3]
    npu_ms = [1000 / v for v in npu_tok_s]

    cpu_tok_s = [15.49, 16.55, 15.75, 16.68, 16.61, 16.38, 16.46, 15.78]
    cpu_ms = [1000 / v for v in cpu_tok_s]

    gpu_tok_s = [13.87, 13.83, 12.79, 13.02, 14.15, 14.41, 13.76, 14.09]
    gpu_ms = [1000 / v for v in gpu_tok_s]

    print("\n\n################ NPU ################")
    npu_result = analyze_npu_overhead(context_lens, npu_ms, args.output_dir)

    # CPU same analysis
    print("\n\n################ CPU ################")
    cpu_ctx = np.array(context_lens, dtype=np.float64)
    cpu_ms_arr = np.array(cpu_ms, dtype=np.float64)
    cpu_coef = np.polyfit(cpu_ctx, cpu_ms_arr, 1)
    print(f"CPU decode_ms ≈ {cpu_coef[1]:.2f} + {cpu_coef[0]*1000:.4f} us × context_len")
    print(f"  Fixed overhead: {cpu_coef[1]:.2f} ms/step")
    print(f"  Context slope: {cpu_coef[0]*1000:.4f} us/token")
    print(f"  → CPU is nearly context-invariant (slope ~ 0)")

    # GPU same analysis
    print("\n################ GPU ################")
    gpu_ctx = np.array(context_lens, dtype=np.float64)
    gpu_ms_arr = np.array(gpu_ms, dtype=np.float64)
    gpu_coef = np.polyfit(gpu_ctx, gpu_ms_arr, 1)
    print(f"GPU decode_ms ≈ {gpu_coef[1]:.2f} + {gpu_coef[0]*1000:.4f} us × context_len")
    print(f"  Fixed overhead: {gpu_coef[1]:.2f} ms/step")
    print(f"  Context slope: {gpu_coef[0]*1000:.4f} us/token")

    # Comparison plot
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(context_lens, npu_ms, 'o-', color='#E74C3C', linewidth=2, markersize=9,
             label=f'NPU (slope={npu_result["slope_us_per_token"]:.2f} us/tok, {npu_result["excess_ratio"]:.1f}x over pure KV BW)')
    ax.plot(context_lens, cpu_ms, 's-', color='#3498DB', linewidth=2, markersize=9,
             label=f'CPU (slope={cpu_coef[0]*1000:.2f} us/tok, near-zero)')
    ax.plot(context_lens, gpu_ms, '^-', color='#2ECC71', linewidth=2, markersize=9,
             label=f'GPU (slope={gpu_coef[0]*1000:.2f} us/tok)')
    ax.set_xscale('log')
    ax.set_xlabel('Context Length')
    ax.set_ylabel('Decode time per token (ms)')
    ax.set_title('Context-Proportional Overhead: NPU vs CPU vs GPU\n'
                  '(Only NPU exhibits significant context-dependent slow-down)')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    path = os.path.join(args.output_dir, "backend_overhead_comparison.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Save combined summary
    summary = {
        "npu": npu_result,
        "cpu": {
            "fixed_overhead_ms": cpu_coef[1],
            "slope_us_per_token": cpu_coef[0] * 1000,
        },
        "gpu": {
            "fixed_overhead_ms": gpu_coef[1],
            "slope_us_per_token": gpu_coef[0] * 1000,
        },
    }
    json_path = os.path.join(args.output_dir, "backend_overhead_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {json_path}")


if __name__ == "__main__":
    main()
