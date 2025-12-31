# profile_attention.py

import sys
import numpy as np

# GPU(OpenCL)
# from opencl.matmul_opencl import matmul_opencl
from opencl.softmax_opencl import softmax_opencl

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

def calc_diff(name, a, b):
    diff = np.abs(a - b)
    max_diff = np.max(np.abs(diff))
    mean_diff = np.mean(np.abs(diff))
    return max_diff, mean_diff

def profile_attention(seq=32, head_dim=64):
    M = seq
    D = head_dim

    np.random.seed(0)

    X = np.random.randn(M, D).astype(np.float32)
    print("Input X :", X)
    W_qkv  = np.random.rand(D, D*3).astype(np.float32) / np.sqrt(D)
    print("W_qkv(Ref) :", W_qkv)
    W_out = np.random.randn(M, D*3).astype(np.float32)
    print("W_out(Ref) :", W_out)

    QKV_ref = X @ W_qkv                # [M, 3D]
    print("QKV_ref :", QKV_ref)
    Q_ref, K_ref, V_ref = np.split(QKV_ref, 3, axis=1)  # 각각 [M,D]
    print("Q_ref :", Q_ref)
    print("K_ref :", K_ref)
    print("V_ref :", V_ref)

    scale = 1.0 / np.sqrt(D)
    QK_ref = Q_ref @ K_ref.T           # [M, M]
    print("QK_ref :", QK_ref)
    P_ref  = softmax_np(QK_ref * scale, axis=-1)
    print("P_ref :", P_ref)
    Y_ref = P_ref @ V_ref             # [M, D]
    print("Y_ref :", Y_ref)

    gpu_acc = {}
    npu_acc = {}

    print()
    print("\n=== GPU (ACL FP16, Mali G610 MC4) ===")

    qkv_lat_gpu, QKV = matmul_acl_f16(X, W_qkv)
    Q, K_mat, V = np.split(QKV.astype(np.float32), 3, axis=1)
    gpu_acc["QKV"] = calc_diff("QKV", QKV.astype(np.float32), QKV_ref)
    print("QKV (GPU) :", QKV)

    qk_lat_gpu, QK = matmul_acl_f16(Q, K_mat.T)
    gpu_acc["QK"] = calc_diff("QK", QK.astype(np.float32), QK_ref)
    print("QK (GPU) :", QK)

    sm_lat_gpu, P = softmax_acl_f16(X=QK*scale, beta=1, axis=0)  # ACL 버전 latency 측정
#    P = softmax_np(QK * scale, axis=-1)  # numpy 버전으로 우선
    gpu_acc["P"] = calc_diff("P", P.astype(np.float32), P_ref)
    print("P (GPU) :", P)

    av_lat_gpu, Y_gpu = matmul_acl_f16(P, V)
    gpu_acc["Y"] = calc_diff("Y", Y_gpu.astype(np.float32), Y_ref)
    print("Y (GPU) :", Y_gpu)


    print(f"QKV (GPU): {qkv_lat_gpu} ms")
    print(f"QKᵀ (GPU): {qk_lat_gpu} ms")
    print(f"softmax (GPU): {sm_lat_gpu} ms")
    print(f"AV  (GPU): {av_lat_gpu} ms")


    print("\n=== NPU (RKNN FP16, RK3588) ===")

    # RKNN 쪽은 IO float32로 던지고 내부에서 FP16
    qkv_lat_npu, QKV = matmul_rknn_f16(X, W_qkv)
    Q, K_mat, V = np.split(QKV.astype(np.float32), 3, axis=1)
    npu_acc["QKV"] = calc_diff("QKV", QKV.astype(np.float32), QKV_ref)
    print("QKV (NPU) :", QKV)

    qk_lat_npu, QK = matmul_rknn_f16(Q, K_mat.T)
    npu_acc["QK"] = calc_diff("QK", QK.astype(np.float32), QK_ref)
    print("QK (NPU) :", QK)

    sm_lat_npu = softmax_rknn(M, M)  # 현재는 latency만
    P  = softmax_np(QK * scale, axis=-1)
    npu_acc["P"] = calc_diff("P", P.astype(np.float32), P_ref)
    print("P (NPU) :", P)

    av_lat_npu, Y_npu  = matmul_rknn_f16(P, V)
    npu_acc["Y"] = calc_diff("Y", Y_npu.astype(np.float32), Y_ref)
    print("Y (NPU) :", Y_npu)


    print(f"QKV (NPU): {qkv_lat_npu:.3f} ms")
    print(f"QKᵀ (NPU): {qk_lat_npu:.3f} ms")
    print(f"softmax (NPU): {sm_lat_npu:.3f} ms")
    print(f"AV  (NPU): {av_lat_npu:.3f} ms")

    print("\n=== Summary ===")
    print("seq : ", seq, ", head_dim : ", head_dim)

    print("=== GPU vs Ref Precision Comparison ===")
    for name, (max_diff, mean_diff) in gpu_acc.items():
        print(f"GPU {name}: max_diff = {max_diff:.3e}, mean_diff = {mean_diff:.3e}")
    print()
    print("=== NPU vs Ref Precision Comparison ===")
    for name, (max_diff, mean_diff) in npu_acc.items():
        print(f"NPU {name}: max_diff = {max_diff:.3e}, mean_diff = {mean_diff:.3e}")
    print()
    gpu_results = {"GPU_FP16": {"QKV": qkv_lat_gpu, "QK": qk_lat_gpu, "softmax": sm_lat_gpu, "AV": av_lat_gpu,}}
    npu_results = {"NPU_FP16": {"QKV": qkv_lat_npu, "QK": qk_lat_npu, "softmax": sm_lat_npu, "AV": av_lat_npu,}}


if __name__ == "__main__":
    # 사용 예:
    #   python profile_attention.py           -> 기본값 seq=640, head_dim=320
    #   python profile_attention.py 128 64   -> seq=128, head_dim=64
    if len(sys.argv) == 1:
        seq = 320
        head_dim = 640
    elif len(sys.argv) == 3:
        seq = int(sys.argv[1])
        head_dim = int(sys.argv[2])
    else:
        print(f"Usage: {sys.argv[0]} [seq head_dim]")
        sys.exit(1)

    profile_attention(seq, head_dim)
