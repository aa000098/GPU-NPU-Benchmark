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


def get_softmax_rknn_path(L, D):
    return os.path.join(HERE, f"softmax_L{L}_D{D}.rknn")

def build_softmax_rknn(L, D):
    onnx_path = os.path.join(HERE, f"softmax_L{L}_D{D}.onnx")
    rknn_path = get_softmax_rknn_path(L, D)

    if os.path.exists(rknn_path):
       # print(">> use existing Softmax RKNN:", rknn_path)
        return rknn_path

    print(">> build new Softmax RKNN:", rknn_path)
    generate_softmax_onnx(L, D, onnx_path)

    rknn = RKNN()

    ret = rknn.config(
        target_platform="rk3588",
        optimization_level=3,
        quantized_dtype="float16",
        target_ops_steps=0,
    )
    if ret != 0:
        raise RuntimeError(f"RKNN config failed: {ret}")

    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        raise RuntimeError(f"RKNN load_onnx failed: {ret}")

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise RuntimeError(f"RKNN build failed: {ret}")

    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"RKNN export_rknn failed: {ret}")

    rknn.release()
    return rknn_path

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

