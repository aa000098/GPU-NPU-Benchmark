# profile_attention.py

import time

# GPU(OpenCL)
from opencl.matmul_opencl import matmul_opencl
from opencl.softmax_opencl import softmax_opencl

# NPU(RKNN)
from my_rknn.matmul_rknn import matmul_rknn
from my_rknn.softmax_rknn import softmax_rknn

def round_summary(s, nd=3):
    return {
        k: {kk: round(vv, nd) for kk, vv in dv.items()}
        for k, dv in s.items()
    }

def profile_attention(seq=2048, head_dim=1024):
    M = seq
    K = head_dim
    N = head_dim

    print()
    print("=== GPU(OpenCL) ===")
    qkv_gpu = matmul_opencl(M, 3*head_dim, head_dim)
    qk_gpu  = matmul_opencl(seq, seq, head_dim)
    sm_gpu  = softmax_opencl(seq, seq)
    av_gpu  = matmul_opencl(seq, head_dim, seq)

    print()
    print(f"QKV (GPU): {qkv_gpu:.3f} ms")
    print(f"QKᵀ (GPU): {qk_gpu:.3f} ms")
    print(f"softmax (GPU): {sm_gpu:.3f} ms")
    print(f"AV (GPU): {av_gpu:.3f} ms")

    print("\n=== NPU(RKNN) ===")
    qkv_npu = matmul_rknn(M, 3*head_dim, head_dim)
    qk_npu  = matmul_rknn(seq, seq, head_dim)
    sm_npu  = softmax_rknn(seq, seq)
    av_npu  = matmul_rknn(seq, head_dim, seq)

    print()
    print(f"QKV (NPU): {qkv_npu:.3f} ms")
    print(f"QKᵀ (NPU): {qk_npu:.3f} ms")
    print(f"softmax (NPU): {sm_npu:.3f} ms")
    print(f"AV (NPU): {av_npu:.3f} ms")

    print("\n=== Summary ===")
    gpu_results = {"GPU": {"QKV": qkv_gpu, "QK": qk_gpu, "softmax": sm_gpu, "AV": av_gpu}}
    npu_results = {"NPU": {"QKV": qkv_npu, "QK": qk_npu, "softmax": sm_npu, "AV": av_npu}}
    print(round_summary(gpu_results))
    print(round_summary(npu_results))
    print()

if __name__ == "__main__":
    profile_attention()

