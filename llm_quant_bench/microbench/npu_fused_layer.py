"""Bonus: test if fusing 5 matmuls into 1 reduces NPU launch overhead.

Per-Llama-layer ops:
  QKV proj: 2048 → 3072
  O proj: 2048 → 2048
  FFN gate: 2048 → 8192
  FFN up: 2048 → 8192
  FFN down: 8192 → 2048

Fused: pack into one big matmul where output is concat of all.
Issue: input dimensions differ (4 with 2048 input, 1 with 8192 input).
We test fusing the 4 same-input ops: QKV + O + gate + up = 2048 → (3072+2048+8192+8192) = 21504

Compare:
A) 4 separate matmuls (2048→3072, 2048→2048, 2048→8192, 2048→8192)
B) 1 fused matmul (2048 → 21504)

If fusion saves launch overhead (4×L → 1×L), should see ~3×L speedup at small M.
"""
import os, sys, time, ctypes
sys.path.insert(0, '/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark')
import numpy as np
from my_rknn.matmul_rknn import matmul_rknn_f16

K = 2048
N_OUTPUTS = [3072, 2048, 8192, 8192]
N_FUSED = sum(N_OUTPUTS)  # 21504

print("=== Fused vs separate matmul (4 ops with same input dim 2048) ===")
print(f"Fused output dim: {N_FUSED}")
print(f"{'M':>4} {'4 separate ms':>14} {'1 fused ms':>12} {'speedup':>10}")

for M in [1, 4, 8, 16, 32]:
    X = np.random.randn(M, K).astype(np.float16)

    # Separate
    t_sep = 0.0
    for N in N_OUTPUTS:
        W = np.random.randn(K, N).astype(np.float16)
        ms, _ = matmul_rknn_f16(X, W, iters=10)
        t_sep += ms

    # Fused
    Wf = np.random.randn(K, N_FUSED).astype(np.float16)
    t_fused, _ = matmul_rknn_f16(X, Wf, iters=10)

    speedup = t_sep / t_fused
    print(f"{M:>4} {t_sep:>14.3f} {t_fused:>12.3f} {speedup:>9.2f}x")
