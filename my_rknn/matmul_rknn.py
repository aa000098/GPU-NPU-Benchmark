# my_rknn/matmul_rknn.py

import os
import time
import numpy as np
import onnx
from onnx import helper, TensorProto
from rknn.api import RKNN   # ✅ Lite가 아니라 Full RKNN 사용

HERE = os.path.dirname(os.path.abspath(__file__))

def generate_matmul_onnx(K, M, N):
    # shape별로 다른 onnx 파일 사용
    onnx_path = os.path.join(HERE, f"matmul_M{M}_N{N}_K{K}.onnx")

    if os.path.exists(onnx_path):
        #print(">> use existing ONNX:", onnx_path)
        return onnx_path

    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [M, K])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [K, N])
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [M, N])

    node = helper.make_node("MatMul", ["A", "B"], ["C"])
    graph = helper.make_graph([node], "MatMulGraph", [A, B], [C])
    model = helper.make_model(graph)
    onnx.save(model, onnx_path)
    return onnx_path

def get_rknn_path(M, N, K, quant=False):
    suffix = "int8" if quant else "fp16"
    return os.path.join(HERE, f"matmul_M{M}_N{N}_K{K}_{suffix}.rknn")


def build_dummy_dataset(M, N, K, num_samples=32):
    """
    MatMul용 dummy calibration dataset 생성
    각 줄:  A_i.npy B_i.npy
    """
    dataset_dir = os.path.join(HERE, f"matmul_calib_M{M}_N{N}_K{K}")
    os.makedirs(dataset_dir, exist_ok=True)

    dataset_txt = os.path.join(dataset_dir, "dataset.txt")
    if os.path.exists(dataset_txt):
        # 이미 만들어져 있으면 재사용
        return dataset_txt

    lines = []
    for i in range(num_samples):
        A = np.random.rand(M, K).astype(np.float32)
        B = np.random.rand(K, N).astype(np.float32)

        a_path = os.path.join(dataset_dir, f"A_{i}.npy")
        b_path = os.path.join(dataset_dir, f"B_{i}.npy")
        np.save(a_path, A)
        np.save(b_path, B)

        # 여러 입력인 경우 "a.npy b.npy" 형식으로 한 줄에 써야 함
        lines.append(f"{a_path} {b_path}\n")

    with open(dataset_txt, "w") as f:
        f.writelines(lines)

    return dataset_txt


def build_matmul_rknn(M, N, K, quant=False):
    rknn_path = get_rknn_path(M, N, K, quant=quant)

    if os.path.exists(rknn_path):
        #print(">> use existing RKNN:", rknn_path)
        return rknn_path

    print(">> build new RKNN:", rknn_path)
    onnx_path = generate_matmul_onnx(K, M, N)

    rknn = RKNN()

    print(">> RKNN config")
    ret = rknn.config(
        target_platform="rk3588",
#        mean_values=[[0], [0]],   # input 0, input 1
#        std_values=[[1], [1]],
    )
    if ret != 0:
        raise RuntimeError(f"RKNN config failed: {ret}")

    print(">> load ONNX:", onnx_path)
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        raise RuntimeError(f"RKNN load_onnx failed: {ret}")

    if quant:
        # INT8 quantization: generate dummy dataset
        dataset_txt = build_dummy_dataset(M, N, K)
        print(">> build RKNN INT8")
        ret = rknn.build(do_quantization=True, dataset=dataset_txt)
    else:
        # FP16 경로: do_quantization=False (Rockchip 문서 기준 float32→float16 변환만 수행)
        print(">> build RKNN FP16")
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        raise RuntimeError(f"RKNN build failed: {ret}")

    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"RKNN export failed: {ret}")

    rknn.release()

    return rknn_path


def matmul_rknn_f16(X, W, iters=30):
    """
    RKNN(FP16 내부) MatMul
    - X: [M, K], float32/float16
    - W: [K, N], float32/float16

    return:
        (C, latency_ms)  if return_output=True
        latency_ms       if return_output=False
    """
    X = np.asarray(X, dtype=np.float32, order="C")
    W = np.asarray(W, dtype=np.float32, order="C")

    if X.ndim != 2 or W.ndim != 2:
        raise ValueError("X, W must be 2D")

    M, K1 = X.shape
    K2, N = W.shape
    if K1 != K2:
        raise ValueError(f"shape mismatch: X ({M},{K1}), W ({K2},{N})")

    rknn_path = build_matmul_rknn(M, N, K1, quant=False)

    rknn = RKNN()
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")

    ret = rknn.init_runtime(target="rk3588")
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    last_out = None

    # warm-up
    for _ in range(3):
        outputs = rknn.inference(inputs=[X, W])
        last_out = outputs[0]

    # benchmark
    start = time.time()
    for _ in range(iters):
        outputs = rknn.inference(inputs=[X, W])
        last_out = outputs[0]
    end = time.time()

    rknn.release()

    latency_ms = (end - start) * 1000.0 / iters
    C = np.array(last_out)  # RKNN output -> numpy

    return latency_ms, C


def matmul_rknn_int8(X, W, iters=30):
    """
    RKNN(INT8 quantized) MatMul
    - X: [M, K], float32 (또는 float16)
    - W: [K, N], float32 (또는 float16)

    RKNN은 내부에서 INT8 quant/dequant 수행, Python 단에서는 float32만 던져주면 됨.
    """
    X = np.asarray(X, dtype=np.float32, order="C")
    W = np.asarray(W, dtype=np.float32, order="C")

    if X.ndim != 2 or W.ndim != 2:
        raise ValueError("X, W must be 2D")

    M, K1 = X.shape
    K2, N = W.shape
    if K1 != K2:
        raise ValueError(f"shape mismatch: X ({M},{K1}), W ({K2},{N})")

    rknn_path = build_matmul_rknn(M, N, K1, quant=True)

    rknn = RKNN()
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")

    ret = rknn.init_runtime(target="rk3588")
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    last_out = None

    # warm-up
    for _ in range(3):
        outputs = rknn.inference(inputs=[X, W])
        last_out = outputs[0]

    # benchmark
    start = time.time()
    for _ in range(iters):
        outputs = rknn.inference(inputs=[X, W])
        last_out = outputs[0]
    end = time.time()

    rknn.release()

    latency_ms = (end - start) * 1000.0 / iters
    C = np.array(last_out)

    return latency_ms, C
