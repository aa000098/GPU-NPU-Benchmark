# rknn/matmul_rknn.py
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

def get_rknn_path(M, N, K):
    return os.path.join(HERE, f"matmul_M{M}_N{N}_K{K}.rknn")

def build_matmul_rknn(M, N, K):
    rknn_path = get_rknn_path(M, N, K)

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

    print(">> build RKNN (FP32/FP16, no quant)")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise RuntimeError(f"RKNN build failed: {ret}")

    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"RKNN export failed: {ret}")

    rknn.release()

    return rknn_path


def matmul_rknn(M, N, K, iters=30):

    rknn_path = build_matmul_rknn(M, N, K)

    rknn = RKNN()
    rknn.load_rknn(rknn_path)

    print(">> init runtime")
    # target 지정 안 해도 보드에서 돌면 알아서 rk3588로 잡히는 경우가 많음
    ret = rknn.init_runtime(target="rk3588")
    if ret != 0:
        raise RuntimeError(f"RKNN init_runtime failed: {ret}")

    # 2) 입력 데이터 준비
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)

    # 3) warm-up
    for _ in range(3):
        outputs = rknn.inference(
            inputs=[A, B], 
            data_format=['nhwc', 'nhwc']
        )

    # 4) 측정
    start = time.time()
    for _ in range(iters):
        outputs = rknn.inference(
            inputs=[A, B], 
            data_format=['nhwc', 'nhwc']
        )
    end = time.time()

    rknn.release()

    latency_ms = (end - start) * 1000 / iters
    return latency_ms

