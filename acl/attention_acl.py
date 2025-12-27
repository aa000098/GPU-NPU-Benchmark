import ctypes
import os
import time
import numpy as np

# -----------------------------------------------------------------------------
# 1. ACL 라이브러리(.so) 로드
# -----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
# 빌드된 라이브러리 경로 (환경에 따라 다를 수 있으니 확인 필요)
# 보통 acl/build/libmatmul_acl.so 에 생성됩니다.
LIB_PATH = os.path.join(HERE, "build/libattention_acl.so")

# 만약 acl 폴더 바로 아래가 아니라면 상위 폴더의 build도 확인
if not os.path.exists(LIB_PATH):
    LIB_PATH_ALT = os.path.join(os.path.dirname(HERE), "build/libattention_acl.so")
    if os.path.exists(LIB_PATH_ALT):
        LIB_PATH = LIB_PATH_ALT
    else:
        # 파일이 없으면 에러 발생
        raise FileNotFoundError(f"ACL Shared Library not found at {LIB_PATH}. Please compile C++ code first.")

_acl = ctypes.CDLL(LIB_PATH)

# -----------------------------------------------------------------------------
# 2. C 함수 시그니처 정의 (ctypes)
# -----------------------------------------------------------------------------

# --- FP16 Functions ---
try:
    _acl.attention_acl_f16_create.argtypes = [ctypes.c_int, ctypes.c_int]
    _acl.attention_acl_f16_create.restype = ctypes.c_void_p

    _acl.attention_acl_f16_run.argtypes = [
        ctypes.c_void_p, # handle
        ctypes.c_void_p, # q_ptr
        ctypes.c_void_p, # k_ptr
        ctypes.c_void_p, # v_ptr
        ctypes.c_void_p  # out_ptr
    ]
    _acl.attention_acl_f16_run.restype = ctypes.c_int
except AttributeError:
    print("[Warning] FP16 Attention functions not found in library.")

# --- INT8 Functions ---
try:
    _acl.attention_acl_i8_create.argtypes = [ctypes.c_int, ctypes.c_int]
    _acl.attention_acl_i8_create.restype = ctypes.c_void_p

    _acl.attention_acl_i8_run.argtypes = [
        ctypes.c_void_p, # handle
        ctypes.c_void_p, # q_ptr
        ctypes.c_void_p, # k_ptr
        ctypes.c_void_p, # v_ptr
        ctypes.c_void_p  # out_ptr
    ]
    _acl.attention_acl_i8_run.restype = ctypes.c_int
except AttributeError:
    print("[Warning] INT8 Attention functions not found in library.")

# -----------------------------------------------------------------------------
# 3. Handle Cache (파이프라인 재사용용)
# -----------------------------------------------------------------------------
_handles_f16 = {}
_handles_i8 = {}

# -----------------------------------------------------------------------------
# 4. Python Wrapper Functions
# -----------------------------------------------------------------------------

def attention_acl_f16(Q, K, V, iters=50):
    """
    [GPU Fused FP16] Softmax((Q @ K.T)/sqrt(d)) @ V
    Input: float32 arrays (will be cast to float16 internaly)
    Output: float32 array
    """
    global _handles_f16

    # 1. Data Type Casting (FP32 -> FP16)
    # ACL은 C-order 메모리 레이아웃을 선호합니다.
    Q_fp16 = np.asarray(Q, dtype=np.float16, order="C")
    K_fp16 = np.asarray(K, dtype=np.float16, order="C")
    V_fp16 = np.asarray(V, dtype=np.float16, order="C")

    M, D = Q_fp16.shape
    
    # Output Buffer 생성
    Out_fp16 = np.empty((M, D), dtype=np.float16, order="C")

    # 2. Get Pointers
    q_ptr = Q_fp16.ctypes.data_as(ctypes.c_void_p)
    k_ptr = K_fp16.ctypes.data_as(ctypes.c_void_p)
    v_ptr = V_fp16.ctypes.data_as(ctypes.c_void_p)
    out_ptr = Out_fp16.ctypes.data_as(ctypes.c_void_p)

    # 3. Handle Management (Create or Retrieve)
    key = (M, D)
    if key not in _handles_f16:
        if not hasattr(_acl, 'attention_acl_f16_create'):
             raise RuntimeError("ACL Library missing FP16 Attention function")
             
        handle = _acl.attention_acl_f16_create(M, D)
        if not handle:
            raise RuntimeError("Failed to create ACL F16 Attention Handle")
        _handles_f16[key] = handle
    
    handle = _handles_f16[key]

    # 4. Warm-up
    for _ in range(3):
        _acl.attention_acl_f16_run(handle, q_ptr, k_ptr, v_ptr, out_ptr)

    # 5. Benchmark
    start = time.time()
    for _ in range(iters):
        _acl.attention_acl_f16_run(handle, q_ptr, k_ptr, v_ptr, out_ptr)
    end = time.time()

    latency_ms = (end - start) * 1000.0 / iters

    # 결과를 다시 FP32로 변환해서 반환 (Python 호환성 위해)
    return latency_ms, Out_fp16.astype(np.float32)


def attention_acl_i8(Q, K, V, iters=50):
    """
    [GPU Fused INT8] Softmax((Q @ K.T)) @ V
    Input: int8 arrays
    Output: int8 array
    """
    global _handles_i8

    # 1. Ensure INT8 & C-order
    Q_i8 = np.asarray(Q, dtype=np.int8, order="C")
    K_i8 = np.asarray(K, dtype=np.int8, order="C")
    V_i8 = np.asarray(V, dtype=np.int8, order="C")

    M, D = Q_i8.shape
    
    # Output Buffer
    Out_i8 = np.empty((M, D), dtype=np.int8, order="C")

    # 2. Get Pointers
    q_ptr = Q_i8.ctypes.data_as(ctypes.c_void_p)
    k_ptr = K_i8.ctypes.data_as(ctypes.c_void_p)
    v_ptr = V_i8.ctypes.data_as(ctypes.c_void_p)
    out_ptr = Out_i8.ctypes.data_as(ctypes.c_void_p)

    # 3. Handle Management
    key = (M, D)
    if key not in _handles_i8:
        if not hasattr(_acl, 'attention_acl_i8_create'):
             raise RuntimeError("ACL Library missing INT8 Attention function")

        handle = _acl.attention_acl_i8_create(M, D)
        if not handle:
            raise RuntimeError("Failed to create ACL INT8 Attention Handle")
        _handles_i8[key] = handle
    
    handle = _handles_i8[key]

    # 4. Warm-up
    for _ in range(3):
        _acl.attention_acl_i8_run(handle, q_ptr, k_ptr, v_ptr, out_ptr)

    # 5. Benchmark
    start = time.time()
    for _ in range(iters):
        _acl.attention_acl_i8_run(handle, q_ptr, k_ptr, v_ptr, out_ptr)
    end = time.time()

    latency_ms = (end - start) * 1000.0 / iters

    return latency_ms, Out_i8
