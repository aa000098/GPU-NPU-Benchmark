import ctypes
import os
import time
import numpy as np

# softmax shared library 경로
HERE = os.path.dirname(os.path.abspath(__file__))
SOFTMAX_ACL_PATH = os.path.join(HERE, "build/libsoftmax_acl.so")

_acl = ctypes.CDLL(SOFTMAX_ACL_PATH)

_acl.softmax_acl_f16.argtypes = [
    ctypes.c_int,                         # M
    ctypes.c_int,                         # N
    ctypes.POINTER(ctypes.c_uint16),      # X (FP16 bitpattern)
    ctypes.POINTER(ctypes.c_uint16),      # Y (FP16 bitpattern)
    ctypes.c_float,                       # beta
    ctypes.c_int,                         # axis
]
_acl.softmax_acl_f16.restype = ctypes.c_int

_acl.softmax_acl_i8.argtypes = [
    ctypes.c_int,                         # M
    ctypes.c_int,                         # N
    ctypes.POINTER(ctypes.c_int8),       # X (INT8)
    ctypes.POINTER(ctypes.c_int8),       # Y (INT8)
    ctypes.c_float,                       # beta
    ctypes.c_int,                         # axis
]
_acl.softmax_acl_i8.restype = ctypes.c_int


def softmax_acl_f16(X, beta=1.0, axis=0, iters=50):
    """
    ACL(CL Softmax, FP16)을 사용한 softmax.
    - X: np.ndarray, shape [M, N], dtype float16/float32 (row-major)
    - beta: scaling factor
    - axis: ACL TensorShape(N, M) 기준 축
        * axis=0 : N 방향(가로) softmax  => 보통 row-wise softmax 용도
        * axis=1 : M 방향(세로) softmax

    return:
        latency_ms, Y (np.float16, [M, N])
    """
    X = np.asarray(X, dtype=np.float16, order="C")
    if X.ndim != 2:
        raise ValueError("X must be 2D [M, N]")

    M, N = X.shape

    # FP16 → uint16 비트패턴 뷰
    X_u16 = X.view(np.uint16)
    Y = np.empty((M, N), dtype=np.float16)
    Y_u16 = Y.view(np.uint16)

    x_ptr = X_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
    y_ptr = Y_u16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))

    # warm-up
    for _ in range(5):
        ret = _acl.softmax_acl_f16(M, N, x_ptr, y_ptr, ctypes.c_float(beta), axis)
        if ret != 0:
            raise RuntimeError(f"softmax_acl_f16 failed in warmup, code={ret}")

    # timing
    start = time.time()
    for _ in range(iters):
        ret = _acl.softmax_acl_f16(M, N, x_ptr, y_ptr, ctypes.c_float(beta), axis)
        if ret != 0:
            raise RuntimeError(f"softmax_acl_f16 failed in loop, code={ret}")
    end = time.time()

    latency_ms = (end - start) * 1000.0 / iters
    return latency_ms, Y.copy()
def softmax_acl_i8(X, beta=1.0, axis=0, iters=50):
    """
    ACL(CL Softmax, INT8)을 사용한 softmax.
    - X: np.ndarray, shape [M, N], dtype int8 (row-major)
    - beta: scaling factor
    - axis: ACL TensorShape(N, M) 기준 축
        * axis=0 : N 방향(가로) softmax  => 보통 row-wise softmax 용도
        * axis=1 : M 방향(세로) softmax
    return:
        latency_ms, Y (np.int8, [M, N])
    """
    X = np.asarray(X, dtype=np.int8, order="C")
    if X.ndim != 2:
        raise ValueError("X must be 2D [M, N]")

    M, N = X.shape

    Y = np.empty((M, N), dtype=np.int8)

    x_ptr = X.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
    y_ptr = Y.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))

    # warm-up
    for _ in range(5):
        ret = _acl.softmax_acl_i8(M, N, x_ptr, y_ptr, ctypes.c_float(beta), axis)
        if ret != 0:
            raise RuntimeError(f"softmax_acl_i8 failed in warmup, code={ret}")

    # timing
    start = time.time()
    for _ in range(iters):
        ret = _acl.softmax_acl_i8(M, N, x_ptr, y_ptr, ctypes.c_float(beta), axis)
        if ret != 0:
            raise RuntimeError(f"softmax_acl_i8 failed in loop, code={ret}")
    end = time.time()

    latency_ms = (end - start) * 1000.0 / iters
    return latency_ms, Y.copy()
