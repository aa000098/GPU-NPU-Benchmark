# profile_attention.py

import sys
import numpy as np

# GPU(OpenCL)
# from opencl.matmul_opencl import matmul_opencl
# from opencl.softmax_opencl import softmax_opencl

# GPU(ACL CLGEMM matmul)
from acl.matmul_acl import matmul_acl_f16, matmul_acl_int8 
from acl.softmax_acl import softmax_acl_f16

# NPU(RKNN)
from my_rknn.matmul_rknn import matmul_rknn_f16
from my_rknn.softmax_rknn import softmax_rknn

def round_summary(s, nd=3):
    return {
        k: {kk: round(vv, nd) for kk, vv in dv.items()}
        for k, dv in s.items()
    }

# softmax: row-wise softmax on [M, M]
def softmax_np(x, axis=-1):
    x = x.astype(np.float32)
    x = np.clip(x, -50.0, 50.0)

    x_max = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - x_max)
    return ex / np.sum(ex, axis=axis, keepdims=True)

def profile_attention(seq=32, head_dim=64):
    M = seq
    D = head_dim

    X = np.random.randn(M, D).astype(np.float32)
    W_qkv  = np.random.rand(D, D*3).astype(np.float32) / np.sqrt(D)
    W_out = np.random.randn(M, D*3).astype(np.float32)

    QKV_ref = X @ W_qkv                # [M, 3D]
    Q_ref, K_ref, V_ref = np.split(QKV_ref, 3, axis=1)  # 각각 [M,D]

    scale = 1.0 / np.sqrt(D)
    QK_ref = Q_ref @ K_ref.T           # [M, M]
    P_ref  = softmax_np(QK_ref * scale, axis=-1)
    Y_ref = P_ref @ V_ref             # [M, D]

    print()
    print("\n=== GPU (ACL FP16, Mali G610 MC4) ===")

    qkv_lat_gpu, QKV = matmul_acl_f16(X, W_qkv)
    Q, K_mat, V = np.split(QKV.astype(np.float32), 3, axis=1)

    qk_lat_gpu, QK = matmul_acl_f16(Q, K_mat.T)

    sm_lat_gpu, P = softmax_acl_f16(X=QK*scale, beta=1, axis=0)  # ACL 버전 latency 측정
#    P = softmax_np(QK * scale, axis=-1)  # numpy 버전으로 우선

    av_lat_gpu, Y_gpu = matmul_acl_f16(P, V)
    maxdiff_y_gpu = np.max(np.abs(Y_gpu - Y_ref))

    print(f"QKV (GPU): {qkv_lat_gpu} ms")
    print(f"QKᵀ (GPU): {qk_lat_gpu} ms")
    print(f"softmax (GPU): {sm_lat_gpu} ms")
    print(f"AV  (GPU): {av_lat_gpu} ms")
    print(f"Y  max|diff| = {maxdiff_y_gpu:.3e}")


    print("\n=== NPU (RKNN FP16, RK3588) ===")

    # RKNN 쪽은 IO float32로 던지고 내부에서 FP16
    qkv_lat_npu, QKV = matmul_rknn_f16(X, W_qkv)
    #qkv_lat_npu, QKV_T = matmul_rknn_f16(W_qkv.T, X.T)  # RKNN matmul은 (N, K)x(K, M)에서 더 높은 성능
    Q, K_mat, V = np.split(QKV.astype(np.float32), 3, axis=1)
    #Q, K_mat, V = np.split(QKV_T.T.astype(np.float32), 3, axis=1)

    qk_lat_npu, QK = matmul_rknn_f16(Q, K_mat.T)

    sm_lat_npu = softmax_rknn(M, M)  # 현재는 latency만
    P  = softmax_np(QK * scale, axis=-1)

    av_lat_npu, Y_npu  = matmul_rknn_f16(P, V)
    maxdiff_y_npu = np.max(np.abs(Y_gpu - Y_ref))

    print(f"QKV (NPU): {qkv_lat_npu:.3f} ms")
    print(f"QKᵀ (NPU): {qk_lat_npu:.3f} ms")
    print(f"softmax (NPU): {sm_lat_npu:.3f} ms")
    print(f"AV  (NPU): {av_lat_npu:.3f} ms")
    print(f"Y  max|diff| = {maxdiff_y_npu:.3e}")

    print("\n=== Summary ===")
    print("seq : ", seq, ", head_dim : ", head_dim)
    gpu_results = {"GPU_FP16": {"QKV": qkv_lat_gpu, "QK": qk_lat_gpu, "softmax": sm_lat_gpu, "AV": av_lat_gpu,}}
    npu_results = {"NPU_FP16": {"QKV": qkv_lat_npu, "QK": qk_lat_npu, "softmax": sm_lat_npu, "AV": av_lat_npu,}}
    print(round_summary(gpu_results))
    print(round_summary(npu_results))
    print()
#    print("Y_ref: \n", Y_ref)
#    print("Y_gpu: \n", Y_gpu)
#    print("Y_npu: \n", Y_npu)


if __name__ == "__main__":
    # 사용 예:
    #   python profile_attention.py           -> 기본값 seq=640, head_dim=320
    #   python profile_attention.py 128 64   -> seq=128, head_dim=64
    if len(sys.argv) == 1:
        seq = 2048
        head_dim = 2048
    elif len(sys.argv) == 3:
        seq = int(sys.argv[1])
        head_dim = int(sys.argv[2])
    else:
        print(f"Usage: {sys.argv[0]} [seq head_dim]")
        sys.exit(1)

    profile_attention(seq, head_dim)
