# profile_attention.py

import sys

# GPU(OpenCL)
from opencl.matmul_opencl import matmul_opencl
from opencl.softmax_opencl import softmax_opencl

# GPU(ACL CLGEMM matmul)
from acl.matmul_acl import matmul_acl 

# NPU(RKNN)
from my_rknn.matmul_rknn import matmul_rknn
from my_rknn.softmax_rknn import softmax_rknn

def round_summary(s, nd=3):
    return {
        k: {kk: round(vv, nd) for kk, vv in dv.items()}
        for k, dv in s.items()
    }

def profile_attention(seq=640, head_dim=320):
    M = seq
    K = head_dim
    N = head_dim

    print()
    print("=== GPU(OpenCL) ===")
#    qkv_gpu = matmul_opencl(M, 3*head_dim, head_dim)
#    qk_gpu  = matmul_opencl(seq, seq, head_dim)
    qkv_gpu = matmul_acl(M, 3*head_dim, head_dim)
    qk_gpu  = matmul_acl(seq, seq, head_dim)
    sm_gpu  = softmax_opencl(seq, seq)
#    av_gpu  = matmul_opencl(seq, head_dim, seq)
    av_gpu  = matmul_acl(seq, head_dim, seq)

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
    print("seq : ", seq, ", head_dim : ", head_dim)
    gpu_results = {"GPU": {"QKV": qkv_gpu, "QK": qk_gpu, "softmax": sm_gpu, "AV": av_gpu}}
    npu_results = {"NPU": {"QKV": qkv_npu, "QK": qk_npu, "softmax": sm_npu, "AV": av_npu}}
    print(round_summary(gpu_results))
    print(round_summary(npu_results))
    print()

if __name__ == "__main__":
    # 사용 예:
    #   python profile_attention.py           -> 기본값 seq=640, head_dim=320
    #   python profile_attention.py 128 64   -> seq=128, head_dim=64
    if len(sys.argv) == 1:
        seq = 640
        head_dim = 320
    elif len(sys.argv) == 3:
        seq = int(sys.argv[1])
        head_dim = int(sys.argv[2])
    else:
        print(f"Usage: {sys.argv[0]} [seq head_dim]")
        sys.exit(1)

    profile_attention(seq, head_dim)
