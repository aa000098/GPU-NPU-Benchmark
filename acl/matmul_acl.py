import ctypes
import os
import time

import numpy as np

# ACL shared library 경로
HERE = os.path.dirname(os.path.abspath(__file__))
MATMUL_ACL_PATH = os.path.join(HERE, "build/libmatmul_acl_f16.so")

_handles = {}

def matmul_acl_f16(X, W, iters=50):
    global _handles
    _acl = ctypes.CDLL(MATMUL_ACL_PATH)

    # 함수 시그니처 정의
    _acl.matmul_acl_f16_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _acl.matmul_acl_f16_create.restype = ctypes.c_void_p
    _acl.matmul_acl_f16_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16),
                                    ctypes.POINTER(ctypes.c_float)] 
    _acl.matmul_acl_f16_run.restype = ctypes.c_int
    _acl.matmul_acl_f16_set_B.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16)]
    _acl.matmul_acl_f16_set_B.restype = ctypes.c_int

    X = np.asarray(X, dtype=np.float16, order="C")
    W = np.asarray(W, dtype=np.float16, order="C")
    M, K = X.shape
    _, N = W.shape

    X_u16 = X.view(np.uint16)
    W_u16 = W.view(np.uint16)
    C = np.empty((M, N), dtype=np.float32, order="C")
    C_u32 = C.view(np.float32)

    a_ptr = X_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    b_ptr = W_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    c_ptr = C_u32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # 1. 핸들 캐싱 (M, N, K가 같으면 재사용)
    key = (M, N, K)
    if key not in _handles:
        handle = _acl.matmul_acl_f16_create(M, N, K)
        _acl.matmul_acl_f16_set_B(handle, b_ptr)
        _handles[key] = handle
    handle = _handles[key]


    # warm-up (이미 구성된 핸들을 쓰므로 훨씬 빠름)
    for _ in range(5):
        _acl.matmul_acl_f16_run(handle, a_ptr, c_ptr)

    # 측정
    start = time.time()
    for _ in range(iters):
        _acl.matmul_acl_f16_run(handle, a_ptr, c_ptr)
    end = time.time()

    latency_ms = (end - start) * 1000.0 / iters
    return latency_ms, C.copy()


def matmul_acl_f16_old(X, W, iters=50):
    """
    ACL(CL GEMM, FP16)을 사용한 matmul.
    - X: np.ndarray, shape [M, K], dtype float16/float32
    - W: np.ndarray, shape [K, N], dtype float16/float32

    return:
        latency_ms, C (np.float16, [M, N])    if return_output=True
        latency_ms                            if return_output=False
    """
    _acl = ctypes.CDLL(MATMUL_ACL_PATH)

    _acl.matmul_acl_f16.argtypes = [
        ctypes.c_int,  # M
        ctypes.c_int,  # N
        ctypes.c_int,  # K
        ctypes.POINTER(ctypes.c_uint16),  # A
        ctypes.POINTER(ctypes.c_uint16),  # B
        ctypes.POINTER(ctypes.c_float),  # C
    ]
    _acl.matmul_acl_f16.restype = ctypes.c_int

    X = np.asarray(X, dtype=np.float16, order="C")
    W = np.asarray(W, dtype=np.float16, order="C")

    if X.ndim != 2 or W.ndim != 2:
        raise ValueError("X, W must be 2D")

    M, K1 = X.shape
    K2, N = W.shape
    if K1 != K2:
        raise ValueError(f"shape mismatch: X ({M},{K1}), W ({K2},{N})")

    # FP16 → uint16 비트패턴 뷰
    X_u16 = X.view(np.uint16)
    W_u16 = W.view(np.uint16)
    C = np.empty((M, N), dtype=np.float16)
    C_u16 = C.view(np.uint16)

    a_ptr = X_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    b_ptr = W_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    c_ptr = C_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))

    # warm-up
    for _ in range(5):
        ret = _acl.matmul_acl_f16(M, N, K1, a_ptr, b_ptr, c_ptr)
        if ret != 0:
            raise RuntimeError(f"matmul_acl_f16 failed in warmup, code={ret}")

    # 측정
    start = time.time()
    for _ in range(iters):
        ret = _acl.matmul_acl_f16(M, N, K1, a_ptr, b_ptr, c_ptr)
        if ret != 0:
            raise RuntimeError(f"matmul_acl_f16 failed in loop, code={ret}")
    end = time.time()

    latency_ms = (end - start) * 1000.0 / iters

    return latency_ms, C.copy()


def matmul_acl_int8(X, W, iters=50):
    """
    ACL(CL GEMM, INT8)을 사용한 matmul.
    - X: np.ndarray, shape [M, K], dtype int8
    - W: np.ndarray, shape [K, N], dtype int8

    return:
        latency_ms, C (np.int32, [M, N])    if return_output=True
        latency_ms                            if return_output=False
    """
    _acl = ctypes.CDLL(MATMUL_ACL_PATH)

    _acl.matmul_acl_int8.argtypes = [
        ctypes.c_int,  # M
        ctypes.c_int,  # N
        ctypes.c_int,  # K
        ctypes.POINTER(ctypes.c_int8),  # A
        ctypes.POINTER(ctypes.c_int8),  # B
        ctypes.POINTER(ctypes.c_int32),  # C
    ]
    _acl.matmul_acl_int8.restype = ctypes.c_int



    X = np.asarray(X, dtype=np.int8, order="C")
    W = np.asarray(W, dtype=np.int8, order="C")

    if X.ndim != 2 or W.ndim != 2:
        raise ValueError("X, W must be 2D")

    M, K1 = X.shape
    K2, N = W.shape
    if K1 != K2:
        raise ValueError(f"shape mismatch: X ({M},{K1}), W ({K2},{N})")

    C = np.empty((M, N), dtype=np.int32)

    a_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    b_ptr = W.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    c_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


    # warm-up
    for _ in range(5):
        ret = _acl.matmul_acl_f16(M, N, K1, a_ptr, b_ptr, c_ptr)
        if ret != 0:
            raise RuntimeError(f"matmul_acl_f16 failed in warmup, code={ret}")

    # 측정
    start = time.time()
    for _ in range(iters):
        ret = _acl.matmul_acl_f16(M, N, K1, a_ptr, b_ptr, c_ptr)
        if ret != 0:
            raise RuntimeError(f"matmul_acl_f16 failed in loop, code={ret}")
    end = time.time()

    latency_ms = (end - start) * 1000.0 / iters

    return latency_ms, C.copy()


