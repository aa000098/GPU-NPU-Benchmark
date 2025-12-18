import os, time, ctypes
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(HERE, "build/libmatmul_rknn_f16.so")
_lib = ctypes.CDLL(LIB_PATH)

_lib.matmul_rknn_f16_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.matmul_rknn_f16_create.restype  = ctypes.c_void_p

_lib.matmul_rknn_f16_set_B.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.matmul_rknn_f16_set_B.restype  = ctypes.c_int

_lib.matmul_rknn_f16_run.argtypes   = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
_lib.matmul_rknn_f16_run.restype    = ctypes.c_int

_lib.matmul_rknn_f16_destroy.argtypes = [ctypes.c_void_p]
_lib.matmul_rknn_f16_destroy.restype  = None


def matmul_rknn_f16(X, W, iters=30):
    X = np.asarray(X, dtype=np.float16, order="C")
    W = np.asarray(W, dtype=np.float16, order="C")
    warmup = 3
    core_mask = 0  # use core 0 and core 1

    if X.ndim != 2 or W.ndim != 2:
        raise ValueError("X, W must be 2D")
    M, K1 = X.shape
    K2, N = W.shape
    if K1 != K2:
        raise ValueError(f"shape mismatch: X({M},{K1}) vs W({K2},{N})")

    h = _lib.matmul_rknn_f16_create(M, K1, N, int(core_mask))
    if not h:
        raise RuntimeError("matmul_rknn_f16_create failed")

    try:
        ret = _lib.matmul_rknn_f16_set_B(h, ctypes.c_void_p(W.ctypes.data))
        if ret != 0:
            raise RuntimeError(f"set_B failed: {ret}")

        C = np.empty((M, N), dtype=np.float32, order="C")

        # warmup
        for _ in range(warmup):
            ret = _lib.matmul_rknn_f16_run(
                h,
                ctypes.c_void_p(X.ctypes.data),
                ctypes.c_void_p(C.ctypes.data),
            )
            if ret != 0:
                raise RuntimeError(f"run failed: {ret}")

        t0 = time.perf_counter()
        for _ in range(iters):
            ret = _lib.matmul_rknn_f16_run(
                h,
                ctypes.c_void_p(X.ctypes.data),
                ctypes.c_void_p(C.ctypes.data),
            )
            if ret != 0:
                raise RuntimeError(f"run failed: {ret}")
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0 / iters
        return latency_ms, C

    finally:
        _lib.matmul_rknn_f16_destroy(h)

