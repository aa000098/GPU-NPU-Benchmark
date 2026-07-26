"""
Roofline Analysis for LLM Decode on RK3588

NPU decode degradation (20 tok/s → 11.3 tok/s at p4096)의 원인이
KV cache memory bandwidth bottleneck임을 증명하는 분석.

입력:
  - 기존 벤치마크 결과 (results/raw/full_results.csv)
  - 시퀀스 길이별 decode 속도 (p32~p4096)
  - RK3588 메모리 대역폭 (측정 또는 spec 값)

출력:
  - KV cache size vs context length table
  - Arithmetic intensity per decode step
  - Roofline plot (log-log: intensity vs throughput)
  - Bytes-per-token estimation (KV read dominant)

Usage:
    python analysis/roofline_analysis.py --output_dir ./results/figures_roofline
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Llama 3.2 1B Instruct architecture constants ─────────────────────
LLAMA_32_1B = {
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,   # GQA with 8 KV heads
    "head_dim": 64,
    "intermediate_size": 8192,
    "vocab_size": 128256,
}

# RK3588 spec (can be overridden by measurement)
RK3588_MEMORY_BANDWIDTH_GBS = {
    "lpddr4x_theoretical": 34.1,   # LPDDR4x-4266, 64-bit × 2 channels
    "lpddr5_theoretical": 51.2,    # LPDDR5-6400 if applicable
    "cpu_stream_measured": None,   # Phase 1-2에서 채울 예정
    "npu_effective": None,          # NPU 실효 대역폭 (추정)
}

# RK3588 compute spec
RK3588_COMPUTE_TOPS = {
    "cpu_fp16_peak": 0.48,  # Cortex-A76 x4 × 2GHz × 4 FMA/cyc × 16 NEON lanes (대략)
    "cpu_int8_peak": 1.92,  # sdot은 4x throughput
    "npu_int8_peak": 6.0,   # RK3588 NPU 3코어 × 2 TOPS each
    "gpu_fp16_peak": 0.45,  # Mali G610 MC4
}


def kv_cache_size_bytes(context_len: int, dtype_bytes: int = 2,
                         arch: dict = LLAMA_32_1B) -> int:
    """Compute total KV cache size in bytes for given context length.

    KV cache = 2 (K and V) × num_layers × num_kv_heads × head_dim × context_len × bytes_per_element
    """
    return (
        2  # K and V
        * arch["num_hidden_layers"]
        * arch["num_key_value_heads"]
        * arch["head_dim"]
        * context_len
        * dtype_bytes
    )


def weight_size_bytes(arch: dict = LLAMA_32_1B, quant_bits: int = 8) -> int:
    """Rough model weight size (for memory-bound analysis).

    Approximation: attention (Q,K,V,O) + MLP (gate,up,down) + embeddings
    """
    d = arch["hidden_size"]
    n_layers = arch["num_hidden_layers"]
    n_heads = arch["num_attention_heads"]
    n_kv = arch["num_key_value_heads"]
    head_dim = arch["head_dim"]
    ff = arch["intermediate_size"]
    vocab = arch["vocab_size"]

    # Attention per layer
    q_proj = d * (n_heads * head_dim)
    k_proj = d * (n_kv * head_dim)
    v_proj = d * (n_kv * head_dim)
    o_proj = (n_heads * head_dim) * d
    attn_per_layer = q_proj + k_proj + v_proj + o_proj

    # MLP per layer (SwiGLU)
    gate = d * ff
    up = d * ff
    down = ff * d
    mlp_per_layer = gate + up + down

    # Total
    layer_weights = (attn_per_layer + mlp_per_layer) * n_layers
    embed = vocab * d  # tie_word_embeddings이면 1번만
    total_params = layer_weights + embed

    return total_params * (quant_bits // 8) if quant_bits >= 8 else total_params * quant_bits // 8


def decode_step_bytes_read(context_len: int, arch: dict = LLAMA_32_1B,
                            weight_bits: int = 8, kv_bits: int = 16) -> dict:
    """Bytes that must be read per decode step.

    A decode step reads:
    1. All model weights (once)
    2. Full KV cache (K and V for all layers)
    3. Input activation (small, 1 × d elements)
    """
    weights = weight_size_bytes(arch, quant_bits=weight_bits)
    kv = kv_cache_size_bytes(context_len, dtype_bytes=kv_bits // 8, arch=arch)
    activation = arch["hidden_size"] * 2  # FP16 input embedding for 1 token

    return {
        "weights_bytes": weights,
        "kv_bytes": kv,
        "activation_bytes": activation,
        "total_bytes": weights + kv + activation,
        "kv_ratio": kv / (weights + kv + activation),
    }


def decode_step_flops(context_len: int, arch: dict = LLAMA_32_1B) -> int:
    """Rough FLOPs per decode step.

    Decode (single output token, context of length n):
    1. QKV projection: 2 × d × (n_heads + 2*n_kv) × head_dim
    2. Attention score: 2 × n × head_dim × n_heads (GQA)
    3. Attention × V: 2 × n × head_dim × n_heads
    4. Output projection: 2 × n_heads × head_dim × d
    5. MLP: 2 × 3 × d × ff (gate, up, down)
    6. LM head: 2 × d × vocab (large but once)

    All per layer for attention/mlp.
    """
    d = arch["hidden_size"]
    n_layers = arch["num_hidden_layers"]
    n_heads = arch["num_attention_heads"]
    n_kv = arch["num_key_value_heads"]
    head_dim = arch["head_dim"]
    ff = arch["intermediate_size"]
    vocab = arch["vocab_size"]
    n = context_len

    qkv_proj = 2 * d * (n_heads + 2 * n_kv) * head_dim
    attn_score = 2 * n * head_dim * n_heads
    attn_out = 2 * n * head_dim * n_heads
    o_proj = 2 * n_heads * head_dim * d
    mlp = 2 * 3 * d * ff
    per_layer = qkv_proj + attn_score + attn_out + o_proj + mlp

    lm_head = 2 * d * vocab

    return per_layer * n_layers + lm_head


def arithmetic_intensity(context_len: int, arch: dict = LLAMA_32_1B,
                          weight_bits: int = 8, kv_bits: int = 16) -> float:
    """Arithmetic intensity (FLOPs / bytes) for decode step."""
    flops = decode_step_flops(context_len, arch)
    bytes_read = decode_step_bytes_read(context_len, arch, weight_bits, kv_bits)["total_bytes"]
    return flops / bytes_read


def print_kv_table(context_lengths, arch=LLAMA_32_1B):
    """Table 출력: context length별 KV 크기, 읽어야 할 바이트, intensity."""
    print(f"\n=== KV Cache Analysis: {arch['num_hidden_layers']}-layer model, "
          f"{arch['num_key_value_heads']} KV heads × {arch['head_dim']} head_dim ===\n")
    print(f"{'Context':>8} {'KV (MB)':>10} {'Weights (MB)':>14} "
          f"{'Total Read (MB)':>16} {'KV ratio':>10} {'FLOPs (G)':>10} {'AI (FLOP/B)':>12}")
    print("-" * 95)

    rows = []
    for n in context_lengths:
        bytes_info = decode_step_bytes_read(n, arch)
        flops = decode_step_flops(n, arch)
        ai = arithmetic_intensity(n, arch)

        kv_mb = bytes_info["kv_bytes"] / 1e6
        w_mb = bytes_info["weights_bytes"] / 1e6
        total_mb = bytes_info["total_bytes"] / 1e6
        kv_ratio = bytes_info["kv_ratio"]
        flops_g = flops / 1e9

        print(f"{n:>8} {kv_mb:>10.2f} {w_mb:>14.2f} {total_mb:>16.2f} "
              f"{kv_ratio:>9.1%} {flops_g:>10.3f} {ai:>12.3f}")
        rows.append({
            "context_len": n,
            "kv_mb": kv_mb,
            "weights_mb": w_mb,
            "total_read_mb": total_mb,
            "kv_ratio": kv_ratio,
            "flops_g": flops_g,
            "arithmetic_intensity": ai,
        })
    return rows


def plot_roofline(context_lengths, measured_decode_tok_s, output_dir,
                   bandwidth_gbs=30.0, peak_tops_int8=1.92,
                   arch=LLAMA_32_1B, label="CPU W8A8"):
    """Roofline plot with measured points.

    bandwidth_gbs: RK3588 실효 메모리 대역폭 (GB/s)
    peak_tops_int8: INT8 peak compute (TOPS)
    measured_decode_tok_s: {context_len: tok/s}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Roofline curves
    ai_range = np.logspace(-2, 2, 500)
    memory_bound = ai_range * bandwidth_gbs  # GFLOP/s = AI × BW(GB/s) = FLOP/B × B/s
    compute_bound = np.full_like(ai_range, peak_tops_int8 * 1000)  # TOPS → GFLOPS

    roofline = np.minimum(memory_bound, compute_bound)

    ax.loglog(ai_range, roofline, 'k--', linewidth=2, label=f'Roofline (BW={bandwidth_gbs} GB/s, peak={peak_tops_int8} TOPS)')
    ax.loglog(ai_range, memory_bound, 'b:', alpha=0.5, label='Memory-bound')
    ax.loglog(ai_range, compute_bound, 'r:', alpha=0.5, label='Compute-bound')

    # Measured points: compute achieved GFLOPS from tok/s
    for n in context_lengths:
        if n not in measured_decode_tok_s:
            continue
        flops_per_token = decode_step_flops(n, arch)
        tok_s = measured_decode_tok_s[n]
        gflops_achieved = flops_per_token * tok_s / 1e9
        ai = arithmetic_intensity(n, arch)

        ax.scatter(ai, gflops_achieved, s=100, zorder=5,
                   edgecolor='black', label=None)
        ax.annotate(f'p{n}\n{tok_s:.1f}t/s',
                    (ai, gflops_achieved), fontsize=8,
                    xytext=(8, -3), textcoords='offset points')

    ax.set_xlabel('Arithmetic Intensity (FLOP / byte)', fontsize=12)
    ax.set_ylabel('Performance (GFLOP/s)', fontsize=12)
    ax.set_title(f'Roofline Analysis: Llama-3.2-1B Decode on RK3588 ({label})', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

    path = os.path.join(output_dir, f"roofline_{label.replace(' ', '_').lower()}.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_kv_vs_decode(context_lengths, measured_decode_tok_s,
                        output_dir, bandwidth_gbs=30.0, arch=LLAMA_32_1B):
    """Figure: KV cache size and predicted KV-read time vs measured decode time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    kv_mbs = []
    predicted_kv_read_ms = []
    measured_decode_ms = []

    for n in context_lengths:
        if n not in measured_decode_tok_s:
            continue
        kv_bytes = kv_cache_size_bytes(n, dtype_bytes=2, arch=arch)
        kv_mbs.append(kv_bytes / 1e6)
        # Time to read KV cache at given bandwidth
        kv_time_s = kv_bytes / (bandwidth_gbs * 1e9)
        predicted_kv_read_ms.append(kv_time_s * 1000)
        # Measured decode time per token
        decode_ms = 1000 / measured_decode_tok_s[n]
        measured_decode_ms.append(decode_ms)

    valid_ctx = [n for n in context_lengths if n in measured_decode_tok_s]

    ax1.semilogx(valid_ctx, kv_mbs, 'o-', linewidth=2, markersize=8, label='KV cache size')
    ax1.set_xlabel('Context Length')
    ax1.set_ylabel('KV Cache (MB)')
    ax1.set_title('KV Cache Size vs Context Length')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.semilogx(valid_ctx, measured_decode_ms, 'o-', linewidth=2, markersize=8,
                 label='Measured decode time')
    ax2.semilogx(valid_ctx, predicted_kv_read_ms, 's--', linewidth=2, markersize=8,
                 label=f'Predicted KV read time\n(BW={bandwidth_gbs}GB/s)')
    ax2.set_xlabel('Context Length')
    ax2.set_ylabel('Time per decode step (ms)')
    ax2.set_title('Decode Time vs KV Read Time (predicted)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Does KV cache explain decode degradation?', fontsize=13)
    fig.tight_layout()

    path = os.path.join(output_dir, "kv_vs_decode.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Roofline analysis for LLM decode")
    parser.add_argument("--output_dir", type=str,
                        default="./results/figures_roofline")
    parser.add_argument("--bandwidth_gbs", type=float, default=30.0,
                        help="Effective memory bandwidth in GB/s")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    context_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    # Print table
    rows = print_kv_table(context_lengths)

    # Save table as CSV
    csv_path = os.path.join(args.output_dir, "kv_cache_analysis.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n  CSV saved: {csv_path}")

    # Measured decode tok/s (from previous benchmarks)
    cpu_decode = {32: 15.49, 64: 16.55, 128: 15.75, 256: 16.68,
                   512: 16.61, 1024: 16.38, 2048: 16.46, 4096: 15.78}
    gpu_decode = {32: 13.87, 64: 13.83, 128: 12.79, 256: 13.02,
                   512: 14.15, 1024: 14.41, 2048: 13.76, 4096: 14.09}
    npu_decode = {32: 20.0, 64: 20.0, 128: 19.6, 256: 18.2,
                   512: 18.1, 1024: 17.0, 2048: 14.6, 4096: 11.3}

    # Plots
    print("\nGenerating roofline plots...")
    plot_roofline(context_lengths, cpu_decode, args.output_dir,
                   bandwidth_gbs=args.bandwidth_gbs, peak_tops_int8=1.92,
                   label="CPU W8A8")
    plot_roofline(context_lengths, npu_decode, args.output_dir,
                   bandwidth_gbs=args.bandwidth_gbs, peak_tops_int8=6.0,
                   label="NPU W8A8")

    plot_kv_vs_decode(context_lengths, npu_decode,
                       args.output_dir, bandwidth_gbs=args.bandwidth_gbs)

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
