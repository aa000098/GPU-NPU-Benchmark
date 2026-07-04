"""
Decode Bottleneck Analysis for RK3588 NPU
=========================================

기존 가설: KV cache bandwidth가 NPU decode degradation의 원인
재검토 필요: Llama 3.2 1B는 GQA로 KV가 작음 (p4096에서도 134MB vs weights 1.2GB)

실제 원인 후보:
1. Weights reload per step (NPU의 on-chip memory 한계 때문)
2. Attention O(n) compute 자체
3. NPU kernel launch/dispatch overhead
4. KV write overhead (새 token마다 K/V append)

본 분석은:
  실측 decode time을 각 부분으로 decompose하고,
  bandwidth vs compute 기여도를 정량화한다.

Usage:
    python analysis/decode_bottleneck_analysis.py
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LLAMA_32_1B = {
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "intermediate_size": 8192,
    "vocab_size": 128256,
}


def attention_flops_decode(n_context, arch=LLAMA_32_1B):
    """Attention FLOPs for decode single token.

    GQA: Q is [n_heads, head_dim], K/V cache is [n_context, n_kv_heads, head_dim]
    Q @ K^T:  [n_heads × head_dim] @ [head_dim × n_context] for each kv_head group
              = 2 × n_heads × head_dim × n_context
    Attn @ V: [n_heads × n_context] @ [n_context × head_dim]
              = 2 × n_heads × head_dim × n_context
    """
    n_h = arch["num_attention_heads"]
    d_h = arch["head_dim"]
    return 2 * 2 * n_h * d_h * n_context  # QK^T + (softmax×V)


def attention_bytes_decode(n_context, arch=LLAMA_32_1B, kv_bits=16):
    """Bytes to read from KV cache for attention computation in decode step.

    K: n_context × n_kv × head_dim × bytes
    V: same
    Q: n_heads × head_dim × bytes (from activation, small)
    """
    n_kv = arch["num_key_value_heads"]
    d_h = arch["head_dim"]
    kv_bytes = 2 * n_context * n_kv * d_h * (kv_bits // 8)  # K + V per layer
    return kv_bytes


def mlp_bytes_decode(arch=LLAMA_32_1B, weight_bits=8):
    """Bytes to read for MLP per decode step per layer.

    SwiGLU: gate/up/down weights, each d × ff or ff × d.
    All weights loaded every step (decode is token-by-token).
    """
    d = arch["hidden_size"]
    ff = arch["intermediate_size"]
    w_bytes = (d * ff + d * ff + ff * d) * (weight_bits // 8)
    return w_bytes


def attn_proj_bytes_decode(arch=LLAMA_32_1B, weight_bits=8):
    """Bytes for Q/K/V/O projection weights per layer."""
    d = arch["hidden_size"]
    n_h = arch["num_attention_heads"]
    n_kv = arch["num_key_value_heads"]
    d_h = arch["head_dim"]
    qkv_out = (d * n_h * d_h) + 2 * (d * n_kv * d_h) + (n_h * d_h * d)
    return qkv_out * (weight_bits // 8)


def total_decode_bytes(n_context, arch=LLAMA_32_1B, weight_bits=8, kv_bits=16):
    """Total bytes touched per decode step.

    If NPU must reload all weights per step (streaming),
    weights dominate. If weights stay on-chip, KV dominates for long context.
    """
    n_layers = arch["num_hidden_layers"]
    vocab = arch["vocab_size"]
    d = arch["hidden_size"]

    # Weights per layer
    w_per_layer = (attn_proj_bytes_decode(arch, weight_bits)
                    + mlp_bytes_decode(arch, weight_bits))
    total_weights = w_per_layer * n_layers
    lm_head_bytes = d * vocab * (weight_bits // 8)

    # KV per layer for attention
    kv_per_layer = attention_bytes_decode(n_context, arch, kv_bits)
    total_kv = kv_per_layer * n_layers

    return {
        "weights": total_weights + lm_head_bytes,
        "kv": total_kv,
        "total": total_weights + lm_head_bytes + total_kv,
    }


def predict_decode_time_ms(n_context, bandwidth_gbs=30.0, peak_tops=6.0,
                            arch=LLAMA_32_1B, weight_bits=8, kv_bits=16,
                            weights_streamed=True, overhead_ms=5.0):
    """Predict decode time under memory+compute+overhead model.

    weights_streamed: True면 매 step마다 weights를 DRAM에서 읽는다고 가정 (NPU).
                      False면 on-chip cache에 있다고 가정 (이상적).
    overhead_ms: kernel launch, synchronization 등 고정 오버헤드.
    """
    bytes_info = total_decode_bytes(n_context, arch, weight_bits, kv_bits)
    if weights_streamed:
        bytes_to_read = bytes_info["total"]
    else:
        bytes_to_read = bytes_info["kv"]

    # Memory read time (ms)
    mem_time_ms = (bytes_to_read / (bandwidth_gbs * 1e9)) * 1000

    # Compute time (ms) - for decode, mostly matmul with sizes ~(1, d)×(d, ff)
    # Total decode FLOPs
    n_layers = arch["num_hidden_layers"]
    d = arch["hidden_size"]
    ff = arch["intermediate_size"]
    n_h = arch["num_attention_heads"]
    n_kv = arch["num_key_value_heads"]
    d_h = arch["head_dim"]
    vocab = arch["vocab_size"]

    flops_per_layer = (
        2 * d * (n_h + 2 * n_kv) * d_h       # QKV proj
        + attention_flops_decode(n_context, arch)
        + 2 * n_h * d_h * d                     # O proj
        + 2 * 3 * d * ff                        # MLP
    )
    flops_total = flops_per_layer * n_layers + 2 * d * vocab

    compute_time_ms = (flops_total / (peak_tops * 1e12)) * 1000

    # Dominant time + overhead
    total_ms = max(mem_time_ms, compute_time_ms) + overhead_ms

    return {
        "mem_time_ms": mem_time_ms,
        "compute_time_ms": compute_time_ms,
        "overhead_ms": overhead_ms,
        "total_ms": total_ms,
        "memory_bound": mem_time_ms > compute_time_ms,
    }


def analyze_vs_measured(output_dir):
    """Compare predicted vs measured decode times."""
    os.makedirs(output_dir, exist_ok=True)

    context_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    # Measured (from previous benchmarks)
    cpu_decode_ms = {n: 1000 / v for n, v in
                      {32: 15.49, 64: 16.55, 128: 15.75, 256: 16.68,
                       512: 16.61, 1024: 16.38, 2048: 16.46, 4096: 15.78}.items()}
    npu_decode_ms = {n: 1000 / v for n, v in
                      {32: 20.0, 64: 20.0, 128: 19.6, 256: 18.2,
                       512: 18.1, 1024: 17.0, 2048: 14.6, 4096: 11.3}.items()}

    # Predict under different assumptions
    scenarios = [
        ("NPU weights streamed (BW=30GB/s)",  {"bandwidth_gbs": 30, "peak_tops": 6.0,  "weights_streamed": True,  "overhead_ms": 5}),
        ("NPU weights on-chip (BW=30GB/s)",   {"bandwidth_gbs": 30, "peak_tops": 6.0,  "weights_streamed": False, "overhead_ms": 5}),
        ("NPU weights streamed (BW=20GB/s)",  {"bandwidth_gbs": 20, "peak_tops": 6.0,  "weights_streamed": True,  "overhead_ms": 5}),
        ("CPU weights streamed (BW=20GB/s)",  {"bandwidth_gbs": 20, "peak_tops": 1.92, "weights_streamed": True,  "overhead_ms": 10}),
    ]

    print("\n=== Predicted vs Measured Decode Time ===\n")
    print(f"{'Context':>8} " + " ".join(f"{s[0][:28]:>30}" for s in scenarios) +
          f"{'NPU measured':>15} {'CPU measured':>15}")
    print("-" * 200)

    results = {name: [] for name, _ in scenarios}
    for n in context_lengths:
        row = f"{n:>8} "
        for name, cfg in scenarios:
            pred = predict_decode_time_ms(n, arch=LLAMA_32_1B, **cfg)
            results[name].append(pred)
            row += f"{pred['total_ms']:>30.1f}"
        row += f"{npu_decode_ms[n]:>15.1f} {cpu_decode_ms[n]:>15.1f}"
        print(row)

    # Plot: predicted vs measured
    fig, ax = plt.subplots(figsize=(11, 7))

    # Measured
    ax.plot(context_lengths, [npu_decode_ms[n] for n in context_lengths],
             'o-', color='red', linewidth=2.5, markersize=9, label='NPU measured', zorder=5)
    ax.plot(context_lengths, [cpu_decode_ms[n] for n in context_lengths],
             's-', color='blue', linewidth=2.5, markersize=9, label='CPU measured', zorder=5)

    # Predictions
    colors = ['lightcoral', 'mistyrose', 'lightsalmon', 'lightsteelblue']
    for (name, _), color in zip(scenarios, colors):
        pred_ms = [p["total_ms"] for p in results[name]]
        ax.plot(context_lengths, pred_ms, '--', alpha=0.7, color=color, label=name)

    ax.set_xscale('log')
    ax.set_xticks(context_lengths)
    ax.set_xticklabels([str(n) for n in context_lengths])
    ax.set_xlabel('Context Length', fontsize=12)
    ax.set_ylabel('Decode Time per Token (ms)', fontsize=12)
    ax.set_title('Decode Time: Measured vs Predicted under Different Bandwidth Assumptions',
                  fontsize=13)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "decode_time_prediction.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")

    # Plot 2: breakdown of predicted decode time for NPU streamed
    fig, ax = plt.subplots(figsize=(11, 6))
    preds = results["NPU weights streamed (BW=30GB/s)"]

    weights_mem = []
    kv_mem = []
    compute = []
    overhead = []
    for n, p in zip(context_lengths, preds):
        bi = total_decode_bytes(n)
        bw = 30e9
        w_ms = (bi["weights"] / bw) * 1000
        kv_ms = (bi["kv"] / bw) * 1000
        # If memory-bound, split mem into weights + kv
        # If compute-bound, compute dominates
        if p["memory_bound"]:
            weights_mem.append(w_ms)
            kv_mem.append(kv_ms)
            compute.append(0)
        else:
            weights_mem.append(0)
            kv_mem.append(0)
            compute.append(p["compute_time_ms"])
        overhead.append(p["overhead_ms"])

    x = np.arange(len(context_lengths))
    width = 0.6
    ax.bar(x, weights_mem, width, label='Weights mem read (streamed)', color='#3498DB')
    ax.bar(x, kv_mem, width, bottom=weights_mem, label='KV cache mem read', color='#E74C3C')
    ax.bar(x, compute, width, bottom=[w + k for w, k in zip(weights_mem, kv_mem)],
           label='Compute', color='#2ECC71')
    ax.bar(x, overhead, width,
           bottom=[w + k + c for w, k, c in zip(weights_mem, kv_mem, compute)],
           label='Overhead', color='#95A5A6')

    # Overlay measured NPU
    npu_ms_list = [npu_decode_ms[n] for n in context_lengths]
    ax.plot(x, npu_ms_list, 'ko-', linewidth=2, markersize=10,
             label='NPU measured', zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in context_lengths])
    ax.set_xlabel('Context Length', fontsize=12)
    ax.set_ylabel('Time per decode step (ms)', fontsize=12)
    ax.set_title('Predicted Decode Time Breakdown (NPU, weights streamed)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    path = os.path.join(output_dir, "decode_breakdown.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    output_dir = "/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/results/figures_roofline"
    analyze_vs_measured(output_dir)


if __name__ == "__main__":
    main()
