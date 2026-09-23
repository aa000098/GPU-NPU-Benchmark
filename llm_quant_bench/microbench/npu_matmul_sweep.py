"""Phase 2: NPU matmul characterization for Llama-3.2-1B decode shapes.

Measures NPU matmul latency for each op type at M=1 (decode GEMV pattern).
Compares to CPU MNN measured per-call latency from MNN_OP_TIMING earlier.
"""
import os, sys, csv, time
sys.path.insert(0, '/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark')
import numpy as np
from my_rknn.matmul_rknn import matmul_rknn_f16

OUT = '/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/paper/v9/v9_data/npu_matmul_sweep.csv'

# Llama-3.2-1B shapes (M=1 decode GEMV)
SHAPES = [
    # (name, M, K, N, cpu_ms_W8_mnn)
    ('QKV_proj',   1, 2048, 3072,  0.41),    # CPU MNN W8 measured
    ('O_proj',     1, 2048, 2048,  0.41),    # CPU MNN W8 (same as QKV avg)
    ('FFN_gate',   1, 2048, 8192,  0.81),    # CPU MNN W8
    ('FFN_up',     1, 2048, 8192,  0.81),
    ('FFN_down',   1, 8192, 2048,  0.81),
    ('LM_head',    1, 2048, 128256, 12.57),  # CPU MNN W8 - the big one
]

def bench_shape(name, M, K, N, iters=30, warmup=5):
    X = np.random.randn(M, K).astype(np.float16)
    W = np.random.randn(K, N).astype(np.float16)
    try:
        ms, C = matmul_rknn_f16(X, W, iters=iters)
        bw_gb_s = (M * K + K * N + M * N) * 2 / (ms * 1e6)  # bytes / ms = MB/ms = GB/s
        flops = 2.0 * M * K * N
        gflops = flops / (ms * 1e6)
        return ms, bw_gb_s, gflops
    except Exception as e:
        return None, None, str(e)

print(f"NPU matmul sweep — Llama-3.2-1B decode shapes (M=1 GEMV)")
print(f"{'Op':<14} {'shape':<22} {'NPU_ms':>10} {'NPU_BW':>10} {'NPU_GFLOPS':>12} {'CPU_ms':>10} {'NPU vs CPU':>12}")

with open(OUT, 'w') as f:
    w = csv.writer(f)
    w.writerow(['op', 'M', 'K', 'N', 'npu_ms', 'npu_bw_gb_s', 'npu_gflops', 'cpu_ms_mnn_w8', 'speedup_npu_over_cpu'])
    for name, M, K, N, cpu_ms in SHAPES:
        shape_str = f"{M}x{K} @ {K}x{N}"
        ms, bw, gflops = bench_shape(name, M, K, N)
        if ms is None:
            print(f"{name:<14} {shape_str:<22} FAIL: {gflops}")
            w.writerow([name, M, K, N, 'FAIL', 'FAIL', 'FAIL', cpu_ms, ''])
        else:
            ratio = cpu_ms / ms
            print(f"{name:<14} {shape_str:<22} {ms:>10.3f} {bw:>10.2f} {gflops:>12.2f} {cpu_ms:>10.2f} {ratio:>11.2f}x")
            w.writerow([name, M, K, N, ms, bw, gflops, cpu_ms, ratio])

print(f"\nSaved: {OUT}")
