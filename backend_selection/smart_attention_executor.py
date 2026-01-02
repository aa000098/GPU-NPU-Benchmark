import pandas as pd
import numpy as np
import os
import sys
import math
import time
import ctypes
import onnx
from onnx import helper, TensorProto
from pycaret.classification import load_model, predict_model

from rknn.api import RKNN

# === 경로 설정 ===
HERE = os.path.dirname(os.path.abspath(__file__))

# 라이브러리 경로 (사용자 환경에 맞게 수정 필요)
# 예: build/libmatmul_acl.so, build/libmatmul_rknn.so 등
LIB_ACL_MATMUL_PATH = os.path.join(HERE, "../acl/build/libmatmul_acl.so")
LIB_RKNN_MATMUL_PATH = os.path.join(HERE, "../my_rknn/build/libmatmul_rknn.so")
LIB_ACL_ATTN_PATH = os.path.join(HERE, "../acl/build/libattention_acl.so")
#LIB_RKNN_ATTN_PATH = os.path.join(HERE, "../my_rknn/build/libattention_rknn.so")
# RKNN Attention 라이브러리 경로가 있다면 추가

def generate_attention_onnx(seq_len, head_dim):
    """Attention Core (SDPA) ONNX 모델 생성"""
    M, D = seq_len, head_dim
    scale_val = 1.0 / np.sqrt(D)
    
    # 모델 저장용 폴더 생성
    model_dir = os.path.join(HERE, "models_cache")
    os.makedirs(model_dir, exist_ok=True)
    
    onnx_filename = f"attention_core_S{M}_D{D}.onnx"
    onnx_path = os.path.join(model_dir, onnx_filename)

    if os.path.exists(onnx_path):
        return onnx_path

    # Define Graph (Q, K, V -> Y)
    Q = helper.make_tensor_value_info("Q", TensorProto.FLOAT, [M, D])
    K = helper.make_tensor_value_info("K", TensorProto.FLOAT, [M, D])
    V = helper.make_tensor_value_info("V", TensorProto.FLOAT, [M, D])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [M, D])

    # Nodes
    node_transpose = helper.make_node("Transpose", ["K"], ["K_T"], perm=[1, 0], name="node_transpose_k")
    node_matmul1 = helper.make_node("MatMul", ["Q", "K_T"], ["QK"], name="node_matmul_qk")
    scale_tensor = helper.make_tensor("scale_const", TensorProto.FLOAT, [], [scale_val])
    node_mul = helper.make_node("Mul", ["QK", "scale_const"], ["QK_scaled"], name="node_scale")
    node_softmax = helper.make_node("Softmax", ["QK_scaled"], ["Softmax_out"], axis=-1, name="node_softmax")
    node_matmul2 = helper.make_node("MatMul", ["Softmax_out", "V"], ["Y"], name="node_matmul_av")

    graph = helper.make_graph(
        [node_transpose, node_matmul1, node_mul, node_softmax, node_matmul2],
        "AttentionCoreGraph", [Q, K, V], [Y], initializer=[scale_tensor]
    )
    
    opset = helper.make_opsetid("", 14)
    model = helper.make_model(graph, opset_imports=[opset])
    onnx.save(model, onnx_path)
    return onnx_path

def build_dummy_dataset_attention(seq_len, head_dim, num_samples=5): # 샘플 수 줄임
    model_dir = os.path.join(HERE, "models_cache")
    dataset_dir = os.path.join(model_dir, f"attn_calib_S{seq_len}_D{head_dim}")
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_txt = os.path.join(dataset_dir, "dataset.txt")

    if os.path.exists(dataset_txt): return dataset_txt

    lines = []
    for i in range(num_samples):
        Q = np.random.rand(seq_len, head_dim).astype(np.float32)
        K = np.random.rand(seq_len, head_dim).astype(np.float32)
        V = np.random.rand(seq_len, head_dim).astype(np.float32)
        q_p, k_p, v_p = os.path.join(dataset_dir, f"Q_{i}.npy"), os.path.join(dataset_dir, f"K_{i}.npy"), os.path.join(dataset_dir, f"V_{i}.npy")
        np.save(q_p, Q); np.save(k_p, K); np.save(v_p, V)
        lines.append(f"{q_p} {k_p} {v_p}\n")

    with open(dataset_txt, "w") as f: f.writelines(lines)
    return dataset_txt

def get_rknn_path(seq_len, head_dim, quant=False):
    model_dir = os.path.join(HERE, "models_cache")
    os.makedirs(model_dir, exist_ok=True)
    suffix = "int8" if quant else "fp16"
    return os.path.join(model_dir, f"attention_core_S{seq_len}_D{head_dim}_{suffix}.rknn")

def build_attention_rknn(seq_len, head_dim, quant=False):
    rknn_path = get_rknn_path(seq_len, head_dim, quant)
    if os.path.exists(rknn_path): return rknn_path

    print(f"    [Build] Compiling new RKNN model (Quant={quant})... This may take a while.")
    onnx_path = generate_attention_onnx(seq_len, head_dim)
    rknn = RKNN()
    rknn.config(target_platform="rk3588")
    if rknn.load_onnx(model=onnx_path) != 0: raise RuntimeError("Load ONNX failed")
    
    dataset = build_dummy_dataset_attention(seq_len, head_dim) if quant else None
    if rknn.build(do_quantization=quant, dataset=dataset) != 0: raise RuntimeError("RKNN Build failed")
    if rknn.export_rknn(rknn_path) != 0: raise RuntimeError("RKNN Export failed")
    rknn.release()
    return rknn_path

def run_attention_rknn_single(Q, K, V, quant=False):
    """단일 실행용 래퍼 (루프 제거)"""
    seq_len, head_dim = Q.shape
    rknn_path = build_attention_rknn(seq_len, head_dim, quant)
    
    rknn = RKNN()
    rknn.load_rknn(rknn_path)
    if rknn.init_runtime(target="rk3588") != 0: raise RuntimeError("Init Runtime failed")
    
    # 1회 실행
    outputs = rknn.inference(inputs=[Q, K, V])
    rknn.release()
    return outputs[0]

class SmartAttentionExecutor:
    def __init__(self, qkv_model='selector_qkv', attn_model='selector_attn'):
        """초기화: 모델 로드 및 공유 라이브러리(C++) 연결"""
        
        # 1. PyCaret 모델 로드
        print("[Init] Loading Prediction Models...")
        try:
            self.brain_qkv = load_model(os.path.join(HERE, qkv_model), verbose=False)
            self.brain_attn = load_model(os.path.join(HERE, attn_model), verbose=False)
        except:
            print("[Warning] Models not found. Using default logic.")
            self.brain_qkv = None
            self.brain_attn = None

        # 2. C 라이브러리 로드 & 시그니처 정의
        self._init_c_libraries()
        
        # 3. 핸들 캐싱 (ACL 등 재사용 가능한 컨텍스트 저장)
        self._handles = {}
        self._rknn_pool = {}

    def _init_c_libraries(self):
        """ctypes 함수 시그니처 정의 (벤치마크 코드에서 추출)"""
        # --- ACL MatMul ---
        try:
            self.lib_acl_mm = ctypes.CDLL(LIB_ACL_MATMUL_PATH)
            # FP16
            self.lib_acl_mm.matmul_acl_f16_create.restype = ctypes.c_void_p
            self.lib_acl_mm.matmul_acl_f16_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float)]
            self.lib_acl_mm.matmul_acl_f16_set_B.restype = ctypes.c_int
            # INT8
            self.lib_acl_mm.matmul_acl_i8_create.restype = ctypes.c_void_p
            self.lib_acl_mm.matmul_acl_i8_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32)]
            self.lib_acl_mm.matmul_acl_i8_set_B.restype = ctypes.c_int
        except OSError as e:
            print(f"[Error] Failed to load ACL MatMul library: {e}")

        # --- RKNN MatMul ---
        try:
            self.lib_rknn_mm = ctypes.CDLL(LIB_RKNN_MATMUL_PATH)
            # FP16
            self.lib_rknn_mm.matmul_rknn_f16_create.restype = ctypes.c_void_p
            self.lib_rknn_mm.matmul_rknn_f16_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib_rknn_mm.matmul_rknn_f16_destroy.argtypes = [ctypes.c_void_p]
            # INT8
            self.lib_rknn_mm.matmul_rknn_i8_create.restype = ctypes.c_void_p
            self.lib_rknn_mm.matmul_rknn_i8_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            self.lib_rknn_mm.matmul_rknn_i8_destroy.argtypes = [ctypes.c_void_p]
        except OSError as e:
            print(f"[Error] Failed to load ACL MatMul library: {e}")
            
        # --- ACL Attention ---
        try:
            self.lib_acl_attn = ctypes.CDLL(LIB_ACL_ATTN_PATH)
            # FP16
            self.lib_acl_attn.attention_acl_f16_create.restype = ctypes.c_void_p
            self.lib_acl_attn.attention_acl_f16_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            # INT8
            self.lib_acl_attn.attention_acl_i8_create.restype = ctypes.c_void_p
            self.lib_acl_attn.attention_acl_i8_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        except OSError as e:
            print(f"[Error] Failed to load ACL MatMul library: {e}")

        # --- RKNN Attention ---
#        try:
#            self.lib_rknn_attn = ctypes.CDLL(LIB_RKNN_ATTN_PATH)
            # FP16
#            self.lib_rknn_attn.attention_acl_f16_create.restype = ctypes.c_void_p
#            self.lib_rknn_attn.attention_acl_f16_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
            # INT8
#            self.lib_rknn_attn.attention_acl_i8_create.restype = ctypes.c_void_p
#            self.lib_rknn_attn.attention_acl_i8_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
#        except OSError as e:
#            print(f"[Error] Failed to load ACL MatMul library: {e}")


    # ==========================================================
    # 1. Prediction (Brain)
    # ==========================================================
    def predict_backend(self, n, d, is_int8):
        if not self.brain_qkv or not self.brain_attn:
            return "NPU", "NPU" # Default

        # Feature Calculation
        def calc_pad(x, alignment=64): return (math.ceil(x / alignment) * alignment) - x
        flops = 2 * (4*n*d*d + 2*n*n*d)
        mem = (n*d + 3*d*d + 3*n*d + n*n + n*d) * 2
        
        df = pd.DataFrame([{
            'n': n, 'd': d, 'is_int8': is_int8,
            'flops': flops, 'memory_bytes': mem,
            'arithmetic_intensity': flops/(mem+1e-9),
            'aspect_ratio': n/d, 'score_matrix_size': n*n,
            'pad_overhead_n': calc_pad(n), 'pad_overhead_d': calc_pad(d)
        }])
        
        # Predict QKV
        p_q = predict_model(self.brain_qkv, data=df, verbose=False)
        l_q = p_q.iloc[0]['prediction_label'] if 'prediction_label' in p_q else p_q.iloc[0]['Label']
        res_q = "NPU" if l_q == 1 else "GPU"

        # Predict Attn
        p_a = predict_model(self.brain_attn, data=df, verbose=False)
        l_a = p_a.iloc[0]['prediction_label'] if 'prediction_label' in p_a else p_a.iloc[0]['Label']
        res_a = "NPU" if l_a == 1 else "GPU"
        
        return res_q, res_a

    # ==========================================================
    # 2. Low-Level Execution Wrappers (No Loops!)
    # ==========================================================
    
    def _run_qkv_npu(self, X, W, n, d, is_int8):
        """RKNN MatMul via C-API (핸들 캐싱 적용으로 속도 최적화)"""
        M, K = X.shape
        _, N = W.shape # N = 3*d
        
        # 캐싱을 위한 고유 키 생성 (Shape + Precision)
        key = f"rknn_mm_{M}_{N}_{K}_{is_int8}"
        
        if is_int8:
            # 1. 핸들이 없으면 생성 및 가중치 설정 (최초 1회만 실행됨)
            if key not in self._handles:
                handle = self.lib_rknn_mm.matmul_rknn_i8_create(M, K, N, 0)
                if not handle:
                    raise RuntimeError("Failed to create RKNN INT8 MatMul handle")
                
                # 가중치(B) 설정은 비용이 크므로 한 번만 수행
                self.lib_rknn_mm.matmul_rknn_i8_set_B(handle, ctypes.c_void_p(W.ctypes.data))
                self._handles[key] = handle
            
            # 2. 캐싱된 핸들 재사용
            handle = self._handles[key]
            C = np.empty((M, N), dtype=np.int32)
            self.lib_rknn_mm.matmul_rknn_i8_run(handle, ctypes.c_void_p(X.ctypes.data), ctypes.c_void_p(C.ctypes.data))
            
            # 주의: 여기서 destroy를 호출하면 안 됩니다! (다음 실행을 위해 유지)
            
        else:
            # FP16 경우
            if key not in self._handles:
                handle = self.lib_rknn_mm.matmul_rknn_f16_create(M, K, N, 0)
                if not handle:
                    raise RuntimeError("Failed to create RKNN FP16 MatMul handle")
                
                self.lib_rknn_mm.matmul_rknn_f16_set_B(handle, ctypes.c_void_p(W.ctypes.data))
                self._handles[key] = handle
            
            handle = self._handles[key]
            C = np.empty((M, N), dtype=np.float32)
            self.lib_rknn_mm.matmul_rknn_f16_run(handle, ctypes.c_void_p(X.ctypes.data), ctypes.c_void_p(C.ctypes.data))
            
        return C

    def _run_qkv_gpu(self, X, W, n, d, is_int8):
        """ACL MatMul 1회 실행 (핸들 캐싱 사용)"""
        M, K = X.shape
        _, N = W.shape
        
        key = f"acl_mm_{M}_{N}_{K}_{is_int8}"
        
        if is_int8:
            if key not in self._handles:
                h = self.lib_acl_mm.matmul_acl_i8_create(M, N, K)
                # Set B (Weights) - 한 번만 하면 됨
                self.lib_acl_mm.matmul_acl_i8_set_B(h, W.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)))
                self._handles[key] = h
            
            C = np.empty((M, N), dtype=np.int32)
            self.lib_acl_mm.matmul_acl_i8_run(
                self._handles[key], 
                X.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), 
                C.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            )
        else:
            # FP16 Input -> FP32 Output (ACL Wrapper 로직 따름)
            if key not in self._handles:
                h = self.lib_acl_mm.matmul_acl_f16_create(M, N, K)
                # Uint16 view for FP16 pointer
                W_ptr = W.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
                self.lib_acl_mm.matmul_acl_f16_set_B(h, W_ptr)
                self._handles[key] = h
            
            C = np.empty((M, N), dtype=np.float32) # Wrapper가 내부에서 FP16->FP32 변환한다고 가정
            X_ptr = X.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
            
            # Run Once!
            self.lib_acl_mm.matmul_acl_f16_run(
                self._handles[key], 
                X_ptr, 
                C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
        return C


    def _run_attn_gpu(self, Q, K, V, n, d, is_int8):
        """ACL Attention 1회 실행"""
        key = f"acl_attn_{n}_{d}_{is_int8}"
        C = np.empty((n, d), dtype=np.float32)

        if is_int8:
            if key not in self._handles:
                self._handles[key] = self.lib_acl_attn.attention_acl_i8_create(n, d)
            
            self.lib_acl_attn.attention_acl_i8_run(
                self._handles[key],
                Q.ctypes.data_as(ctypes.c_void_p),
                K.ctypes.data_as(ctypes.c_void_p),
                V.ctypes.data_as(ctypes.c_void_p),
                C.ctypes.data_as(ctypes.c_void_p)
            )
        else:
            if key not in self._handles:
                self._handles[key] = self.lib_acl_attn.attention_acl_f16_create(n, d)
            
            # Data Casting needed for ctypes
            Q_ptr = Q.astype(np.float16).ctypes.data_as(ctypes.c_void_p)
            K_ptr = K.astype(np.float16).ctypes.data_as(ctypes.c_void_p)
            V_ptr = V.astype(np.float16).ctypes.data_as(ctypes.c_void_p)
            Out_ptr = C.astype(np.float16).ctypes.data_as(ctypes.c_void_p) # 내부에서 FP16으로 씀

            self.lib_acl_attn.attention_acl_f16_run(self._handles[key], Q_ptr, K_ptr, V_ptr, Out_ptr)
            
        return C

    '''
    def _run_attn_npu(self, Q, K, V, n, d, is_int8):
        """RKNN Attention 1회 실행"""
        key = f"acl_attn_{n}_{d}_{is_int8}"
        C = np.empty((n, d), dtype=np.float32)

        if is_int8:
            if key not in self._handles:
                self._handles[key] = self.lib_rknn_attn.attention_acl_i8_create(n, d)
            
            self.lib_rknn_attn.attention_rknn_i8_run(
                self._handles[key],
                Q.ctypes.data_as(ctypes.c_void_p),
                K.ctypes.data_as(ctypes.c_void_p),
                V.ctypes.data_as(ctypes.c_void_p),
                C.ctypes.data_as(ctypes.c_void_p)
            )
        else:
            if key not in self._handles:
                self._handles[key] = self.lib_rknn_attn.attention_rknn_f16_create(n, d)
            
            # Data Casting needed for ctypes
            Q_ptr = Q.astype(np.float16).ctypes.data_as(ctypes.c_void_p)
            K_ptr = K.astype(np.float16).ctypes.data_as(ctypes.c_void_p)
            V_ptr = V.astype(np.float16).ctypes.data_as(ctypes.c_void_p)
            Out_ptr = C.astype(np.float16).ctypes.data_as(ctypes.c_void_p) # 내부에서 FP16으로 씀

            self.lib_rknn_attn.attention_rknn_f16_run(self._handles[key], Q_ptr, K_ptr, V_ptr, Out_ptr)
            
        return C
    '''
    def _run_attn_npu(self, Q, K, V, n, d, is_int8):
        """
        [수정됨] RKNN Runtime을 캐싱하여 재사용하는 로직
        """
        # 1. 고유 키 생성 (입력 크기와 양자화 여부에 따라 모델이 다름)
        key = f"attn_rknn_{n}_{d}_{is_int8}"

        # 2. 런타임이 없으면 생성 (최초 1회만 실행됨)
        if key not in self._rknn_pool:
            
            # 모델 경로 확보 (빌드가 안 되어 있으면 빌드 수행)
            rknn_path = build_attention_rknn(n, d, quant=bool(is_int8))
            
            # RKNN 초기화
            rknn_instance = RKNN()
            rknn_instance.load_rknn(rknn_path)
            
            # 하드웨어 런타임 연결 (가장 시간이 많이 걸리는 부분)
            ret = rknn_instance.init_runtime(target="rk3588")
            if ret != 0:
                print(f"[Error] Init runtime failed for {key}")
                return None
            
            # 풀에 저장
            self._rknn_pool[key] = rknn_instance

        # 3. 캐싱된 런타임으로 추론만 수행 (매우 빠름)
        rknn_instance = self._rknn_pool[key]
        
        # RKNN inference는 리스트를 반환함
        # inputs=[Q, K, V] 순서는 ONNX 생성 순서와 일치해야 함
        outputs = rknn_instance.inference(inputs=[Q, K, V])
        
        return outputs[0]

    def release_resources(self):
        """종료 시 리소스 해제"""
        for key, rknn in self._rknn_pool.items():
            rknn.release()
        self._rknn_pool.clear()

    # ==========================================================
    # 3. Main Pipeline (Execute)
    # ==========================================================
    def execute(self, X, n, d, is_int8=0):
        print(f"\n[Request] N={n}, D={d}, INT8={is_int8}")
        
        # 1. Decide Backend
        q_dev, a_dev = self.predict_backend(n, d, is_int8)
        print(f"   -> Plan: QKV({q_dev}) + Attn({a_dev})")
        
        # 2. Prepare Weights (Dummy for test)
        # 실제론 모델에서 가져와야 함. 여기선 Shape만 맞춤 (D, 3D)
        W_type = np.int8 if is_int8 else np.float16
        W_QKV = np.random.randn(d, 3*d).astype(W_type)
        X_in = X.astype(W_type)
        
        # 3. Run QKV
        t0 = time.time()
        qkv_out = None
        if q_dev == 'NPU':
            qkv_out = self._run_qkv_npu(X_in, W_QKV, n, d, is_int8)
        else:
            qkv_out = self._run_qkv_gpu(X_in, W_QKV, n, d, is_int8)
        t_qkv = (time.time() - t0) * 1000

        # 4. Split & Convert (Bridge)
        t1 = time.time()
        # QKV Output Split: (N, 3D) -> Q, K, V
        # 주의: ACL/RKNN 결과 타입이 다를 수 있음 (INT32 vs FP32)
        Q, K, V = np.split(qkv_out, 3, axis=1)
        
        # 하이브리드 변환 로직
        if q_dev != a_dev:
            print(f"   -> Bridge: Converting {q_dev} output to {a_dev} input...")
            # 예: NPU INT32 -> GPU FP16
            if is_int8 and a_dev == 'GPU':
                 # Dequantize (Scale은 1.0 가정)
                 Q = Q.astype(np.float16)
                 K = K.astype(np.float16)
                 V = V.astype(np.float16)
            # 기타 변환 로직...
        t_bridge = (time.time() - t1) * 1000
        
        # 5. Run Attention
        t2 = time.time()
        final_out = None
        if a_dev == 'NPU':
            final_out = self._run_attn_npu(Q, K, V, n, d, is_int8)
        else:
            final_out = self._run_attn_gpu(Q, K, V, n, d, is_int8)
        t_attn = (time.time() - t2) * 1000
        
        print(f"   [Done] Total: {t_qkv + t_bridge + t_attn:.3f} ms (QKV: {t_qkv:.2f}, Bridge: {t_bridge:.2f}, Attn: {t_attn:.2f})")
        return final_out

if __name__ == "__main__":
    executor = SmartAttentionExecutor()
    
    # Test Loop
    scenarios = [(128, 64, 0), (1, 4096, 0), (512, 64, 1), (1, 1024, 0)]
    
    for n, d, i8 in scenarios:
        # Dummy Input
        x_dummy = np.random.randn(n, d).astype(np.float16)
        executor.execute(x_dummy, n, d, i8)
