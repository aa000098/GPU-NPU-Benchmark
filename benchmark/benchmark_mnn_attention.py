import pandas as pd
import numpy as np
import time
import os
import sys
import onnxruntime as ort
import MNN
from tqdm import tqdm

# 1. 경로 설정
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

# ONNX 모델이 있는 폴더 (../my_rknn)
MODEL_DIR = os.path.join(PROJECT_ROOT, 'my_rknn')

# 결과 저장 경로
SAVE_PATH = os.path.join(HERE, 'framework_benchmark_results.csv')

print(f"Searching for models in: {MODEL_DIR}")

# === 벤치마크 함수들 ===

def bench_mnn_gpu(onnx_path, iters=20):
    """MNN GPU (OpenCL)"""
    try:
        # Config: Backend 3 = OpenCL (Mali GPU), Precision Low = FP16
        config = {'backend': 3, 'precision': 'low', 'numThread': 4}
        
        # Session 생성
        interpreter = MNN.Interpreter(onnx_path)
        session = interpreter.createSession(config)
        
        # Warmup
        for _ in range(5):
            interpreter.runSession(session)
            
        # Measure
        start = time.time()
        for _ in range(iters):
            interpreter.runSession(session)
            interpreter.waitSession(session) # GPU Sync 필수
        end = time.time()
        
        return (end - start) * 1000.0 / iters
    except Exception as e:
        # print(f"[MNN Error] {e}")
        return None

def bench_ort_cpu(onnx_path, n, d, iters=10):
    """ONNX Runtime CPU"""
    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        sess = ort.InferenceSession(onnx_path, opts, providers=['CPUExecutionProvider'])
        
        # Dummy Input 생성
        # 입력 노드 이름을 동적으로 찾아서 데이터 주입
        inputs = {}
        for node in sess.get_inputs():
            shape = []
            for dim in node.shape:
                if isinstance(dim, str): shape.append(n if 'seq' in dim else d) # 동적 축 처리
                elif isinstance(dim, int): shape.append(dim)
                else: shape.append(1)
            
            # 만약 shape 추론이 어렵다면 고정값 [N, D] 사용
            if len(shape) != 2: shape = [n, d]
                
            inputs[node.name] = np.random.randn(*shape).astype(np.float32)

        # Warmup
        sess.run(None, inputs)
        
        start = time.time()
        for _ in range(iters):
            sess.run(None, inputs)
        end = time.time()
        
        return (end - start) * 1000.0 / iters
    except Exception as e:
        # print(f"[ORT Error] {e}")
        return None

# === 메인 실행 ===
def run_benchmark():
    # 탐색 공간 (collect_data.py와 동일하게 설정)
    seq_lens = [1, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 4096]
    seq_lens += [33, 65, 129, 257, 513, 769, 1025, 1537, 2049, 4097]
    
    head_dims = [32, 64, 96, 128, 256, 384, 512, 768, 1024, 1536, 2048, 4096]
    head_dims += [33, 65, 97, 129, 257, 513, 769, 1025, 1537, 2049, 4097]

    seq_lens = sorted(list(set(seq_lens)))
    head_dims = sorted(list(set(head_dims)))
    
    # 0 (FP16/32) 과 1 (INT8)
    quant_modes = [0, 1] 

    # Resume 로직
    file_exists = os.path.isfile(SAVE_PATH)
    processed = set()
    if file_exists:
        try:
            df_exist = pd.read_csv(SAVE_PATH)
            for _, r in df_exist.iterrows():
                processed.add((int(r['n']), int(r['d']), int(r['is_int8'])))
            print(f"Resuming... {len(processed)} combinations already done.")
        except: pass

    pbar = tqdm(total=len(seq_lens) * len(head_dims) * 2)

    for n in seq_lens:
        for d in head_dims:
            for is_int8 in quant_modes:
                
                if (n, d, is_int8) in processed:
                    pbar.update(1)
                    continue
                
                # 파일 이름 규칙 설정
                # FP32/16 -> attention_core_S{n}_D{d}.onnx
                # INT8    -> attention_core_S{n}_D{d}_int8.onnx (라고 가정)
                if is_int8 == 1:
                    filename = f"attention_core_S{n}_D{d}_int8.onnx"
                else:
                    filename = f"attention_core_S{n}_D{d}.onnx"
                
                model_path = os.path.join(MODEL_DIR, filename)
                
                # 파일이 없으면 스킵 (매우 중요)
                if not os.path.exists(model_path):
                    # print(f"Skip: {filename} not found")
                    pbar.update(1)
                    continue
                
                row = {'n': n, 'd': d, 'is_int8': is_int8}
                
                # 1. MNN GPU 측정 (OpenCL)
                row['latency_mnn'] = bench_mnn_gpu(model_path)
                
                # 2. ORT CPU 측정
                row['latency_ort'] = bench_ort_cpu(model_path, n, d)
                
                # 저장
                df_chunk = pd.DataFrame([row])
                if not os.path.isfile(SAVE_PATH):
                    df_chunk.to_csv(SAVE_PATH, index=False, mode='w', header=True)
                else:
                    df_chunk.to_csv(SAVE_PATH, index=False, mode='a', header=False)
                
                pbar.update(1)
    
    pbar.close()
    print(f"\n[Done] Results saved to {SAVE_PATH}")

if __name__ == "__main__":
    run_benchmark()
