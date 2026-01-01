import os
import time
import numpy as np
import onnx
from onnx import helper, TensorProto
from rknn.api import RKNN

# 현재 파일 위치
HERE = os.path.dirname(os.path.abspath(__file__))

def generate_attention_onnx(seq_len, head_dim):
    """
    Attention Core (SDPA) ONNX 모델 생성
    Inputs: Q, K, V (Shape: [seq_len, head_dim])
    Output: Context Vector (Shape: [seq_len, head_dim])
    Logic: Softmax( (Q @ K.T) / sqrt(d) ) @ V
    """
    M = seq_len
    D = head_dim
    scale_val = 1.0 / np.sqrt(D)

    onnx_filename = f"attention_core_S{M}_D{D}.onnx"
    onnx_path = os.path.join(HERE, onnx_filename)

    if os.path.exists(onnx_path):
        return onnx_path

    # 1. Define Inputs
    Q = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [M, D])
    K = helper.make_tensor_value_info("K", TensorProto.FLOAT, [M, D])
    V = helper.make_tensor_value_info("V", TensorProto.FLOAT, [M, D])
    
    # 2. Define Output
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, D])

    # 3. Create Nodes
    # Node A: Transpose K -> [D, M]
    node_transpose = helper.make_node(
        "Transpose", ["K"], ["K_T"], perm=[1, 0], name="node_transpose_k"
    )

    # Node B: MatMul (Q @ K_T) -> [M, M]
    node_matmul1 = helper.make_node(
        "MatMul", ["Q", "K_T"], ["QK"], name="node_matmul_qk"
    )

    # Node C: Scale (QK * scale) -> [M, M]
    # Scale 상수를 텐서로 생성
    scale_tensor = helper.make_tensor("scale_const", TensorProto.FLOAT, [], [scale_val])
    node_mul = helper.make_node(
        "Mul", ["QK", "scale_const"], ["QK_scaled"], name="node_scale"
    )

    # Node D: Softmax -> [M, M]
    node_softmax = helper.make_node(
        "Softmax", ["QK_scaled"], ["Softmax_out"], axis=-1, name="node_softmax"
    )

    # Node E: MatMul (Softmax_out @ V) -> [M, D]
    node_matmul2 = helper.make_node(
        "MatMul", ["Softmax_out", "V"], ["Y"], name="node_matmul_av"
    )

    # 4. Create Graph & Model
    # initializer에 scale 상수 포함
    graph = helper.make_graph(
        [node_transpose, node_matmul1, node_mul, node_softmax, node_matmul2],
        "AttentionCoreGraph",
        [Q, K, V],
        [Y],
        initializer=[scale_tensor]
    )
    
    # Opset version 12 이상 추천 (MatMul, Softmax 호환성)
    opset = helper.make_opsetid("", 14)
    model = helper.make_model(graph, opset_imports=[opset])
    
    onnx.save(model, onnx_path)
    # print(f">> Generated ONNX: {onnx_path}")
    return onnx_path

def get_rknn_path(seq_len, head_dim, quant=False):
    suffix = "int8" if quant else "fp16"
    return os.path.join(HERE, f"attention_core_S{seq_len}_D{head_dim}_{suffix}.rknn")

def build_dummy_dataset_attention(seq_len, head_dim, num_samples=16):
    """
    INT8 양자화를 위한 Calibration 데이터셋 생성 (Q, K, V)
    """
    dataset_dir = os.path.join(HERE, f"attn_calib_S{seq_len}_D{head_dim}")
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_txt = os.path.join(dataset_dir, "dataset.txt")

    if os.path.exists(dataset_txt):
        return dataset_txt

    lines = []
    for i in range(num_samples):
        # 0~1 사이 랜덤 값 생성
        Q = np.random.rand(seq_len, head_dim).astype(np.float32)
        K = np.random.rand(seq_len, head_dim).astype(np.float32)
        V = np.random.rand(seq_len, head_dim).astype(np.float32)

        q_path = os.path.join(dataset_dir, f"Q_{i}.npy")
        k_path = os.path.join(dataset_dir, f"K_{i}.npy")
        v_path = os.path.join(dataset_dir, f"V_{i}.npy")

        np.save(q_path, Q)
        np.save(k_path, K)
        np.save(v_path, V)

        # space로 구분하여 나열
        lines.append(f"{q_path} {k_path} {v_path}\n")

    with open(dataset_txt, "w") as f:
        f.writelines(lines)
    
    return dataset_txt

def build_attention_rknn(seq_len, head_dim, quant=False):
    rknn_path = get_rknn_path(seq_len, head_dim, quant)
    if os.path.exists(rknn_path):
        return rknn_path

    #print(f">> [Build] New Attention RKNN (Quant={quant}): {rknn_path}")
    
    # 1. Generate ONNX
    onnx_path = generate_attention_onnx(seq_len, head_dim)

    # 2. RKNN Config
    rknn = RKNN()
    rknn.config(target_platform="rk3588") # 필요한 경우 optimization_level 설정

    # 3. Load ONNX
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        raise RuntimeError("Load ONNX failed")

    # 4. Build
    if quant:
        dataset_txt = build_dummy_dataset_attention(seq_len, head_dim)
        ret = rknn.build(do_quantization=True, dataset=dataset_txt)
    else:
        ret = rknn.build(do_quantization=False)
    
    if ret != 0:
        raise RuntimeError("RKNN Build failed")

    # 5. Export
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError("RKNN Export failed")
    
    rknn.release()
    return rknn_path

def run_attention_rknn(Q, K, V, quant=False, iters=20):
    """
    통합 Attention RKNN 실행 함수
    Inputs: Q, K, V (numpy arrays, float32)
    Returns: (latency_ms, Output_Result)
    """
    # Type Casting
    Q = np.asarray(Q, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    V = np.asarray(V, dtype=np.float32)

    seq_len, head_dim = Q.shape
    
    # 1. Build or Load RKNN
    rknn_path = build_attention_rknn(seq_len, head_dim, quant=quant)

    # 2. Init Runtime
    rknn = RKNN()
    rknn.load_rknn(rknn_path)
    ret = rknn.init_runtime(target="rk3588")
    if ret != 0:
        raise RuntimeError("Init Runtime failed")

    # 3. Warmup
    for _ in range(3):
        _ = rknn.inference(inputs=[Q, K, V])

    # 4. Benchmark
    start = time.time()
    last_out = None
    for _ in range(iters):
        outputs = rknn.inference(inputs=[Q, K, V])
        last_out = outputs[0]
    end = time.time()

    latency_ms = (end - start) * 1000.0 / iters
    rknn.release()

    return latency_ms, last_out

# --- Wrapper Functions for external calls ---

def attention_rknn_f16(Q, K, V, iters=30):
    return run_attention_rknn(Q, K, V, quant=False, iters=iters)

def attention_rknn_i8(Q, K, V, iters=30):
    return run_attention_rknn(Q, K, V, quant=True, iters=iters)
