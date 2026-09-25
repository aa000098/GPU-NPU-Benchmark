"""Phase 3: Simulate Llama-3.2-1B forward on NPU at various batch sizes.

Per layer matmul ops (excluding attention which we keep on CPU):
- QKV proj: M × 2048 → 3072 (combined Q+K+V)
- O proj: M × 2048 → 2048
- FFN gate: M × 2048 → 8192
- FFN up: M × 2048 → 8192
- FFN down: M × 8192 → 2048

Total per layer: 5 matmul ops. Times 16 layers.

This simulates the verify phase of speculative decoding where M=K candidate
tokens go through the model in one batched pass.

Compare to:
- CPU sequential M decode passes (CPU MNN measured 67ms/token = 67M ms total)
- Pure NPU per-layer time × 16 + overhead

NOTE: This is matmul-only timing. Real verify includes attention compute on CPU
and norm/residual/RoPE which are tiny (<2 ms per token regardless of M).
"""
import os, sys, csv, time
sys.path.insert(0, '/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark')
import numpy as np
from my_rknn.matmul_rknn import matmul_rknn_f16

OUT = '/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/paper/v9/v9_data/npu_layer_sim.csv'

# Llama-3.2-1B per-layer matmuls
LAYER_OPS = [
    ('QKV_proj', 2048, 3072),  # K, N
    ('O_proj',   2048, 2048),
    ('FFN_gate', 2048, 8192),
    ('FFN_up',   2048, 8192),
    ('FFN_down', 8192, 2048),
]
N_LAYERS = 16
LM_HEAD_K, LM_HEAD_N = 2048, 128256

# CPU MNN baseline (per single-token decode, measured)
CPU_PER_TOKEN_MS = 67.5  # measured at ctx=1024 W8A8

print(f"NPU full-Llama layer simulator (Llama-3.2-1B, 16 layers + LM head)")
print(f"{'M':>4} {'layer ms':>12} {'lm_head ms':>12} {'total ms':>12} {'ms/tok':>10} {'vs CPU':>12}")

with open(OUT, 'w') as f:
    w = csv.writer(f)
    w.writerow(['M', 'layer_ms', 'lm_head_ms', 'total_ms', 'ms_per_tok', 'speedup_vs_cpu'])

    for M in [1, 2, 4, 8, 16, 32]:
        # Pre-build NPU handles for each op (one-time setup)
        # We run each op iters=20 to amortize launch
        total_layer_ms = 0.0
        for name, K, N in LAYER_OPS:
            X = np.random.randn(M, K).astype(np.float16)
            W = np.random.randn(K, N).astype(np.float16)
            ms, _ = matmul_rknn_f16(X, W, iters=15)
            total_layer_ms += ms
        layer_ms = total_layer_ms  # one full layer (5 matmuls)

        # 16 layers
        all_layers_ms = layer_ms * N_LAYERS

        # LM head once at end (only need final logits — but for verify of M tokens
        # we need M logit vectors, so it's M × LM head amortized)
        Xlm = np.random.randn(M, LM_HEAD_K).astype(np.float16)
        Wlm = np.random.randn(LM_HEAD_K, LM_HEAD_N).astype(np.float16)
        lm_ms, _ = matmul_rknn_f16(Xlm, Wlm, iters=10)

        total_ms = all_layers_ms + lm_ms
        ms_per_tok = total_ms / M
        cpu_total_ms = CPU_PER_TOKEN_MS * M
        speedup = cpu_total_ms / total_ms

        print(f"{M:>4} {layer_ms:>12.3f} {lm_ms:>12.3f} {total_ms:>12.3f} {ms_per_tok:>10.3f} {speedup:>11.2f}x")
        w.writerow([M, layer_ms, lm_ms, total_ms, ms_per_tok, speedup])

print(f"\nSaved: {OUT}")
print()
print("Interpretation:")
print("  ms_per_tok < CPU 67.5 ms = NPU verify viable for speculative decoding")
print("  speedup = how much faster verify-M-tokens is vs sequential CPU-M-tokens")
