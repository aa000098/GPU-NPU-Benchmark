"""Phase 3-pivot: Analyze NPU matmul launch overhead.

Use the EXISTING matmul_rknn_f16_new (which uses persistent handle internally)
and matmul_rknn_f16 (also uses persistent + warmup) to measure latencies.
Fit linear model latency = L + M*C to extract launch overhead L.
"""
import os, sys, csv, time
sys.path.insert(0, '/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark')
import numpy as np
from my_rknn.matmul_rknn import matmul_rknn_f16

OUT = '/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/paper/v9/v9_data/npu_overhead.csv'

# CPU MNN W8 reference (per single-token decode):
CPU_MS = {'QKV_proj': 0.41, 'FFN_gate': 0.81, 'FFN_down': 0.81, 'LM_head': 12.57}

SHAPES = [(2048, 3072, "QKV_proj"), (2048, 8192, "FFN_gate"),
          (8192, 2048, "FFN_down"), (2048, 128256, "LM_head")]

print("=== NPU matmul latency vs M ===")
print("Linear fit: latency_ms = L + M × C  (L = launch overhead, C = per-token compute)")
print()

with open(OUT, 'w') as f:
    w = csv.writer(f)
    w.writerow(['op', 'K', 'N', 'M', 'ms', 'gflops'])

    for K, N, name in SHAPES:
        print(f"--- {name} (shape MxK @ K x{N}) ---")
        print(f"{'M':>4} {'ms':>10} {'GFLOPS':>10}")
        Ms = []
        ts = []
        for M in [1, 2, 4, 8, 16, 32]:
            X = np.random.randn(M, K).astype(np.float16)
            W = np.random.randn(K, N).astype(np.float16)
            try:
                ms, _ = matmul_rknn_f16(X, W, iters=15)
                gflops = 2.0 * M * K * N / (ms * 1e6)
                print(f"{M:>4} {ms:>10.4f} {gflops:>10.2f}")
                w.writerow([name, K, N, M, ms, gflops])
                Ms.append(M); ts.append(ms)
            except Exception as e:
                print(f"{M:>4} FAIL: {e}")

        # Fit linear model
        if len(Ms) >= 2:
            Ms_arr = np.array(Ms)
            ts_arr = np.array(ts)
            A = np.vstack([np.ones_like(Ms_arr, dtype=float), Ms_arr.astype(float)]).T
            (L, C), *_ = np.linalg.lstsq(A, ts_arr, rcond=None)
            cpu_per_token = CPU_MS[name]
            # Crossover M where NPU < CPU per-token: L + M*C < M * cpu_per_token
            # L < M * (cpu_per_token - C)
            # M > L / (cpu_per_token - C)
            if cpu_per_token > C:
                crossover_M = L / (cpu_per_token - C)
                cstr = f"{crossover_M:.1f}"
            else:
                cstr = "infeasible (NPU per-tok cost > CPU per-tok)"
            print(f"\n  Fit: L = {L:.3f} ms (launch), C = {C:.4f} ms/token")
            print(f"  CPU MNN baseline: {cpu_per_token:.2f} ms/token")
            print(f"  NPU breakeven M: {cstr}")
        print()

print(f"Saved: {OUT}")
