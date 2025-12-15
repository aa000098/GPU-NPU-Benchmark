import ctypes
import os
import time

# ACL shared library 경로
HERE = os.path.dirname(os.path.abspath(__file__))
MATMUL_ACL_PATH = os.path.join(HERE, "build/libmatmul_acl.so")

_acl = ctypes.CDLL(MATMUL_ACL_PATH)
#_acl.acl_gemm_fp16.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
#_acl.acl_gemm_fp16.restype  = ctypes.c_int

def matmul_acl(M, N, K):
    """
    Arm Compute Library(CL GEMM)을 사용한 matmul.
    시간 측정은 Python에서 수행.
    """

    iters = 50

    # optional: warm-up
    for _ in range(5):
        ret = _acl.matmul_acl(M, N, K)
        if ret != 0:
            raise RuntimeError(f"matmul_acl failed with code {ret}")

    start = time.time()
    for _ in range(iters):
        ret = _acl.matmul_acl(M, N, K)
        if ret != 0:
            raise RuntimeError(f"matmul_acl failed with code {ret}")
    end = time.time()

    latency_ms = (end - start) * 1000.0 / iters

    return latency_ms

if __name__=="__main__":
    matmul_acl(32, 32, 32)
