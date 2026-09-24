"""Phase 2b: Test NPU matmul speedup vs M (batch size).

Hypothesis: NPU GEMV (M=1) loses but NPU batched matmul (M=4,8,16,32) wins.
This determines whether speculative decoding (batched verify) is viable.
"""
import os, sys, csv
sys.path.insert(0, '/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark')
import numpy as np
from my_rknn.matmul_rknn import matmul_rknn_f16

OUT = '/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/paper/v9/v9_data/npu_batched_sweep.csv'

# For each shape, sweep M from 1 to 64
SHAPES = [
    ('QKV_proj',   2048, 3072),
    ('FFN_gate',   2048, 8192),
    ('FFN_down',   8192, 2048),
    ('LM_head',    2048, 128256),
]

M_LIST = [1, 2, 4, 8, 16, 32, 64]

print(f"{'Op':<14} {'M':>4} {'shape':<22} {'NPU_ms':>10} {'NPU_GFLOPS':>12} {'ms_per_M':>10} {'BW_GB/s':>10}")

with open(OUT, 'w') as f:
    w = csv.writer(f)
    w.writerow(['op', 'M', 'K', 'N', 'npu_ms', 'npu_gflops', 'ms_per_M', 'bw_gb_s'])
    for name, K, N in SHAPES:
        for M in M_LIST:
            X = np.random.randn(M, K).astype(np.float16)
            W = np.random.randn(K, N).astype(np.float16)
            try:
                ms, _ = matmul_rknn_f16(X, W, iters=20)
                flops = 2.0 * M * K * N
                gflops = flops / (ms * 1e6)
                bytes_ = (M * K + K * N + M * N) * 2  # fp16
                bw = bytes_ / (ms * 1e6)
                ms_per = ms / M
                shape_str = f"{M}x{K} @ {K}x{N}"
                print(f"{name:<14} {M:>4} {shape_str:<22} {ms:>10.3f} {gflops:>12.2f} {ms_per:>10.4f} {bw:>10.2f}")
                w.writerow([name, M, K, N, ms, gflops, ms_per, bw])
            except Exception as e:
                print(f"{name:<14} {M:>4} FAIL: {e}")
                w.writerow([name, M, K, N, 'FAIL', '', '', ''])
        print()

print(f"Saved: {OUT}")
