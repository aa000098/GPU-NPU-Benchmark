"""Quick: test if NPU int8 matmul has different launch overhead profile.
Per spec, RK3588 NPU: 6 TOPS int8, ~3 TFLOPS fp16. Maybe int8 path is faster.
"""
import os, sys, time, ctypes
sys.path.insert(0, '/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark')
import numpy as np
from my_rknn import matmul_rknn

_lib = matmul_rknn._lib

def bench_i8(M, K, N, iters=20):
    X = np.random.randint(-128, 128, (M, K), dtype=np.int8)
    X = np.ascontiguousarray(X)
    W = np.random.randint(-128, 128, (K, N), dtype=np.int8)
    W = np.ascontiguousarray(W)
    h = _lib.matmul_rknn_i8_create(M, K, N, 0)
    if not h:
        raise RuntimeError("create failed")
    try:
        ret = _lib.matmul_rknn_i8_set_B(h, ctypes.c_void_p(W.ctypes.data))
        if ret != 0:
            raise RuntimeError(f"set_B failed: {ret}")
        C = np.empty((M, N), dtype=np.int32, order="C")
        # warmup
        for _ in range(3):
            _lib.matmul_rknn_i8_run(h,
                ctypes.c_void_p(X.ctypes.data),
                ctypes.c_void_p(C.ctypes.data))
        t0 = time.perf_counter()
        for _ in range(iters):
            _lib.matmul_rknn_i8_run(h,
                ctypes.c_void_p(X.ctypes.data),
                ctypes.c_void_p(C.ctypes.data))
        t1 = time.perf_counter()
    finally:
        _lib.matmul_rknn_i8_destroy(h)
    return (t1 - t0) * 1000.0 / iters

print("=== NPU int8 matmul ===")
print(f"{'op':<14} {'M':>4} {'shape':<22} {'ms':>10} {'GOPS':>10}")

SHAPES = [(2048, 3072, "QKV_proj"), (2048, 8192, "FFN_gate"),
          (8192, 2048, "FFN_down"), (2048, 128256, "LM_head")]
M_LIST = [1, 4, 8, 16, 32]

for K, N, name in SHAPES:
    for M in M_LIST:
        try:
            ms = bench_i8(M, K, N)
            gops = 2.0 * M * K * N / (ms * 1e6)
            shape_str = f"{M}x{K} @ {K}x{N}"
            print(f"{name:<14} {M:>4} {shape_str:<22} {ms:>10.3f} {gops:>10.2f}")
        except Exception as e:
            print(f"{name:<14} {M:>4} FAIL: {e}")
    print()
