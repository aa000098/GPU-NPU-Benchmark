# my_rknn/softmax_rknn.py

import os
import time
import numpy as np
import onnx
from onnx import helper, TensorProto
from rknn.api import RKNN

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_softmax_onnx(L, D, onnx_path):
    """
    입력:  X [L, D]
    출력:  Y [L, D]
    축:    axis=1 (D 방향 softmax, row-wise)
    """
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [L, D])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [L, D])

    node = helper.make_node(
        "Softmax",
        inputs=["X"],
        outputs=["Y"],
        axis=1,          # row-wise softmax
    )

    graph = helper.make_graph(
        [node],
        "SoftmaxGraph",
        [X],
        [Y],
    )
    model = helper.make_model(graph)
    onnx.save(model, onnx_path)


def build_softmax_rknn(L, D, do_quantization=False):
    suffix = "i8" if do_quantization else "f16"
    onnx_path = os.path.join(HERE, f"softmax_L{L}_D{D}.onnx")
    rknn_path = os.path.join(HERE, f"softmax_L{L}_D{D}_{suffix}.rknn")

    if os.path.exists(rknn_path):
       # print(">> use existing Softmax RKNN:", rknn_path)
        return rknn_path

    print(">> build new Softmax RKNN:", rknn_path)
    generate_softmax_onnx(L, D, onnx_path)

    rknn = RKNN()

    # INT8 양자화를 위해 더미 데이터셋 생성
    dataset_path = os.path.join(HERE, "dataset.txt")
    if do_quantization:
        dummy_data = np.random.rand(L, D).astype(np.float32)
        np.save(os.path.join(HERE, "dummy_data.npy"), dummy_data)
        with open(dataset_path, "w") as f:
            f.write(os.path.join(HERE, "dummy_data.npy"))

    ret = rknn.config(
        target_platform="rk3588",
        optimization_level=3,
        #quantized_dtype="float16",
        #target_ops_steps=0,
    )
    if ret != 0:
        raise RuntimeError(f"RKNN config failed: {ret}")

    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        raise RuntimeError(f"RKNN load_onnx failed: {ret}")

    #ret = rknn.build(do_quantization=False)
    ret = rknn.build(do_quantization=do_quantization, dataset=dataset_path if do_quantization else None)
    if ret != 0:
        raise RuntimeError(f"RKNN build failed: {ret}")

    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"RKNN export_rknn failed: {ret}")

    rknn.release()
    return rknn_path

def _softmax_rknn_base(X, do_quantization, iters=30):
    """
    X를 입력받아 NPU에서 Softmax를 수행하고 (latency, 결과값)을 반환
    """
    L, D = X.shape
    rknn_path = build_softmax_rknn(L, D, do_quantization=do_quantization)

    rknn = RKNN()
    ret = rknn.load_rknn(rknn_path)
    if ret != 0: raise RuntimeError(f"load_rknn failed: {ret}")

    ret = rknn.init_runtime(target="rk3588")
    if ret != 0: raise RuntimeError(f"init_runtime failed: {ret}")

    # RKNN은 float32 입력을 권장 (내부에서 FP16 변환)
    if X.dtype != np.float32:
        X = X.astype(np.float32)

    # warm-up
    for _ in range(5):
        outputs = rknn.inference(inputs=[X], data_format=['nhwc'])

    # 측정 및 실제 결과 획득
    start = time.perf_counter()
    for _ in range(iters):
        outputs = rknn.inference(inputs=[X], data_format=['nhwc'])
    end = time.perf_counter()

    rknn.release()

    latency_ms = (end - start) * 1000 / iters
    return latency_ms, outputs[0] # latency와 결과 행렬을 같이 반환

def softmax_rknn_f16(X, iters=30):
    return _softmax_rknn_base(X, do_quantization=False, iters=iters)

def softmax_rknn_i8(X, iters=30):
    # 입력이 float라면 int8로 스케일링이 필요할 수 있으나, 
    # RKNN inference가 float를 받아 내부 양자화를 수행하므로 그대로 전달합니다.
    return _softmax_rknn_base(X, do_quantization=True, iters=iters)

def softmax_rknn(L, D):
    """
    NPU에서 Softmax [L, D] latency 측정
    """
    rknn_path = build_softmax_rknn(L, D)

    rknn = RKNN()
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")

    ret = rknn.init_runtime(target="rk3588")
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    X = np.random.rand(L, D).astype(np.float32)

    # warm-up
    for _ in range(5):
        _ = rknn.inference(
                inputs=[X],
                 data_format=['nhwc']
            )

    # 측정
    start = time.time()
    for _ in range(30):
        _ = rknn.inference(
                inputs=[X],
                 data_format=['nhwc']
            )
    end = time.time()

    rknn.release()

    latency_ms = (end - start) * 1000 / 30
    return latency_ms

