# profile_attention.py

import numpy as np

import sys
import os

# ---------------------------------------------------------------------
# 현재 파일(benchmark 폴더 내)의 부모 디렉토리(Project_Root)를 경로에 추가
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__)) # benchmark 폴더 경로
parent_dir = os.path.dirname(current_dir)                # 상위 폴더(Project Root) 경로
sys.path.append(parent_dir)

# GPU(OpenCL)
# from opencl.matmul_opencl import matmul_opencl
# from opencl.softmax_opencl import softmax_opencl

# GPU(ACL CLGEMM matmul)
from acl.matmul_acl import matmul_acl_f16, matmul_acl_i8
from acl.softmax_acl import softmax_acl_f16, softmax_acl_i8
from acl.attention_acl import attention_acl_f16, attention_acl_i8

# NPU(RKNN)
from my_rknn.matmul_rknn import matmul_rknn_f16, matmul_rknn_i8
from my_rknn.softmax_rknn import softmax_rknn_f16, softmax_rknn_i8, softmax_rknn

from my_rknn.attention_rknn import attention_rknn_f16, attention_rknn_i8

def round_summary(s, nd=3):
    return { k: round(v, nd) for k, v in s.items() }

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

    X = np.random.randn(M, D).astype(np.float32) / np.sqrt(D)
    W_qkv  = np.random.randn(D, D*3).astype(np.float32) / np.sqrt(D)
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
    total_gpu = qkv_lat_gpu + qk_lat_gpu + sm_lat_gpu + av_lat_gpu
    maxdiff_y_gpu = np.max(np.abs(Y_gpu - Y_ref))

    fused_core_gpu_lat, Y_gpu_fused = attention_acl_f16(Q, K_mat, V)
    gpu_fused_total = qkv_lat_gpu + fused_core_gpu_lat
    maxdiff_y_gpu_fused = np.max(np.abs(Y_gpu_fused - Y_ref))

    print(f"QKV (GPU): {qkv_lat_gpu} ms")
    print(f"QKᵀ (GPU): {qk_lat_gpu} ms")
    print(f"softmax (GPU): {sm_lat_gpu} ms")
    print(f"AV  (GPU): {av_lat_gpu} ms")
    print(f"Total (GPU): {total_gpu} ms")
    print(f"Fused Attention Core (GPU): {fused_core_gpu_lat:.3f} ms")
    print(f"Fused total (GPU): {gpu_fused_total:.3f} ms")
    print(f"Y  max|diff| = {maxdiff_y_gpu:.3e}")
    print(f"Y_fused  max|diff| = {maxdiff_y_gpu_fused:.3e}")


    print("\n=== NPU (RKNN FP16, RK3588) ===")

    # RKNN 쪽은 IO float32로 던지고 내부에서 FP16
    #if M < D:
    #    qkv_lat_npu, QKV = matmul_rknn_f16(X, W_qkv)
    qkv_lat_npu, QKV = matmul_rknn_f16(X, W_qkv)
    #else:
    #    qkv_lat_npu, QKV_T = matmul_rknn_f16(W_qkv.T, X.T)  # RKNN matmul은 (N, K)x(K, M)에서 더 높은 성능
    #    QKV = QKV_T.T
    Q, K_mat, V = np.split(QKV.astype(np.float32), 3, axis=1)

    #if M < D:
    #    qk_lat_npu, QK = matmul_rknn_f16(Q, K_mat.T)
    qk_lat_npu, QK = matmul_rknn_f16(Q, K_mat.T)
    #else:
    #    qk_lat_npu, QK_T = matmul_rknn_f16(K_mat, Q.T)
    #    QK = QK_T.T

    sm_lat_npu, P = softmax_rknn_f16(QK)  # 현재는 latency만
    #P  = softmax_np(QK * scale, axis=-1)

    #if M < D: 
    #    av_lat_npu, Y_npu  = matmul_rknn_f16(P, V)
    av_lat_npu, Y_npu  = matmul_rknn_f16(P, V)
    #else:
    #    av_lat_npu, Y_npu  = matmul_rknn_f16(V.T, P.T)
    #    Y_npu = Y_npu.T
    total_npu = qkv_lat_npu + qk_lat_npu + sm_lat_npu + av_lat_npu
    maxdiff_y_npu = np.max(np.abs(Y_npu - Y_ref))

    # Fused Attention Core 실행 (Q, K, V 입력)
    fused_core_npu_lat, Y_npu_fused = attention_rknn_f16(Q, K_mat, V)
    npu_fused_total = qkv_lat_npu + fused_core_npu_lat
    maxdiff_y_npu_fused = np.max(np.abs(Y_npu_fused - Y_ref))

    print(f"QKV (NPU): {qkv_lat_npu:.3f} ms")
    print(f"QKᵀ (NPU): {qk_lat_npu:.3f} ms")
    print(f"softmax (NPU): {sm_lat_npu:.3f} ms")
    print(f"AV  (NPU): {av_lat_npu:.3f} ms")
    print(f"Total (NPU): {total_npu:.3f} ms")
    print(f"Fused Attention Core (NPU): {fused_core_npu_lat:.3f} ms")
    print(f"Fused total (NPU): {npu_fused_total:.3f} ms")
    print(f"Y  max|diff| = {maxdiff_y_npu:.3e}")
    print(f"Y_fused  max|diff| = {maxdiff_y_npu_fused:.3e}")

    print("\n=== Summary ===")
    print("seq : ", seq, ", head_dim : ", head_dim)
    gpu_results = {
        "qkv": qkv_lat_gpu,
        "qk": qk_lat_gpu,
        "softmax": sm_lat_gpu,
        "av": av_lat_gpu,
        "total": total_gpu,
        "fused_core": fused_core_gpu_lat,
        "fused_total": gpu_fused_total
    }
    npu_results = {
        "qkv": qkv_lat_npu,
        "qk": qk_lat_npu,
        "softmax": sm_lat_npu,
        "av": av_lat_npu,
        "total": total_npu,
        "fused_core": fused_core_npu_lat,
        "fused_total": npu_fused_total
    }
    print("GPU_FLOAT16: ", round_summary(gpu_results))
    print("NPU_FLOAT16: ", round_summary(npu_results))
    print()
#    print("Y_ref: \n", Y_ref)
#    print("Y_gpu: \n", Y_gpu)
#    print("Y_npu: \n", Y_npu)
    return gpu_results, npu_results

def profile_attention_i8(seq=32, head_dim=64):
    M = seq
    D = head_dim

    X = np.random.randn(M, D).astype(np.float32) / np.sqrt(D)
    W_qkv  = np.random.randn(D, D*3).astype(np.float32) / np.sqrt(D)
    W_out = np.random.randn(M, D*3).astype(np.float32)

    QKV_ref = X @ W_qkv                # [M, 3D]
    Q_ref, K_ref, V_ref = np.split(QKV_ref, 3, axis=1)  # 각각 [M,D]

    scale = 1.0 / np.sqrt(D)
    QK_ref = Q_ref @ K_ref.T           # [M, M]
    P_ref  = softmax_np(QK_ref * scale, axis=-1)
    Y_ref = P_ref @ V_ref             # [M, D]

    print()
    print("\n=== GPU (ACL INT8, Mali G610 MC4) ===")
    qkv_lat_gpu, QKV = matmul_acl_i8(X, W_qkv)
    Q, K_mat, V = np.split(QKV.astype(np.float32), 3, axis=1)
    qk_lat_gpu, QK = matmul_acl_i8(Q, K_mat.T)
    sm_lat_gpu, P = softmax_acl_i8(X=QK*scale, beta=1, axis=0)  # ACL 버전 latency 측정
    av_lat_gpu, Y_gpu = matmul_acl_i8(P, V)
    total_gpu = qkv_lat_gpu + qk_lat_gpu + sm_lat_gpu + av_lat_gpu
    maxdiff_y_gpu = np.max(np.abs(Y_gpu - Y_ref))
    fused_core_gpu_lat, Y_gpu_fused = attention_acl_i8(Q, K_mat, V)
    gpu_fused_total = qkv_lat_gpu + fused_core_gpu_lat
    maxdiff_y_gpu_fused = np.max(np.abs(Y_gpu_fused - Y_ref))

    print(f"QKV (GPU INT8): {qkv_lat_gpu} ms")
    print(f"QKᵀ (GPU INT8): {qk_lat_gpu} ms")
    print(f"softmax (GPU INT8): {sm_lat_gpu} ms")
    print(f"AV  (GPU INT8): {av_lat_gpu} ms")
    print(f"Total (GPU INT8): {total_gpu} ms")
    print(f"Y  max|diff| = {maxdiff_y_gpu:.3e}")
    print(f"Fused Attention Core (GPU INT8): {fused_core_gpu_lat:.3f} ms")
    print(f"Fused total (GPU INT8): {gpu_fused_total:.3f} ms")
    print(f"Y_fused  max|diff| = {maxdiff_y_gpu_fused:.3e}")

    print()
    print("\n=== NPU (RKNN INT8, RK3588) ===")
    qkv_lat_npu, QKV = matmul_rknn_i8(X, W_qkv)
    Q, K_mat, V = np.split(QKV.astype(np.float32), 3, axis=1)
    qk_lat_npu, QK = matmul_rknn_i8(Q, K_mat.T)
    sm_lat_npu, P = softmax_rknn_i8(QK)
    #P  = softmax_np(QK * scale, axis=-1)
    av_lat_npu, Y_npu  = matmul_rknn_i8(P, V)
    total_npu = qkv_lat_npu + qk_lat_npu + sm_lat_npu + av_lat_npu
    maxdiff_y_npu = np.max(np.abs(Y_npu - Y_ref))

    # Fused Attention Core 실행 (Q, K, V 입력)
    fused_core_npu_lat, Y_fused = attention_rknn_i8(Q, K_mat, V)
    npu_fused_total = qkv_lat_npu + fused_core_npu_lat
    maxdiff_y_npu_fused = np.max(np.abs(Y_fused - Y_ref))

    print(f"QKV (NPU INT8): {qkv_lat_npu:.3f} ms")
    print(f"QKᵀ (NPU INT8): {qk_lat_npu:.3f} ms")
    print(f"softmax (NPU INT8): {sm_lat_npu:.3f} ms")
    print(f"AV  (NPU INT8): {av_lat_npu:.3f} ms")
    print(f"Total (NPU INT8): {total_npu:.3f} ms")
    print(f"Fused Attention Core (NPU INT8): {fused_core_npu_lat:.3f} ms")
    print(f"Fused total (NPU INT8): {npu_fused_total:.3f} ms")
    print(f"Y  max|diff| = {maxdiff_y_npu:.3e}")
    print(f"Y_fused  max|diff| = {maxdiff_y_npu_fused:.3e}")

    print("\n=== Summary INT8 ===")
    print("seq : ", seq, ", head_dim : ", head_dim)
    gpu_results = {
        "qkv": qkv_lat_gpu,
        "qk": qk_lat_gpu,
        "softmax": sm_lat_gpu,
        "av": av_lat_gpu,
        "total": total_gpu,
        "fused_core": fused_core_gpu_lat,
        "fused_total": gpu_fused_total
    }
    npu_results = {
        "qkv": qkv_lat_npu,
        "qk": qk_lat_npu,
        "softmax": sm_lat_npu,
        "av": av_lat_npu,
        "total": total_npu,
        "fused_core": fused_core_npu_lat,
        "fused_total": npu_fused_total

    }
    print("GPU_INT8: ", round_summary(gpu_results))
    print("NPU_INT8: ", round_summary(npu_results))
    print()

    return gpu_results, npu_results


if __name__ == "__main__":
    # 사용 예:
    #   python profile_attention.py           -> 기본값 seq=640, head_dim=320
    #   python profile_attention.py 128 64   -> seq=128, head_dim=64
    if len(sys.argv) == 1:
        seq = 100
        head_dim = 1024
    elif len(sys.argv) == 3:
        seq = int(sys.argv[1])
        head_dim = int(sys.argv[2])
    else:
        print(f"Usage: {sys.argv[0]} [seq head_dim]")
        sys.exit(1)

    profile_attention(seq, head_dim)
    profile_attention_i8(seq, head_dim)
