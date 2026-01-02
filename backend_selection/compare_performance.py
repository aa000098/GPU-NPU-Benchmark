import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from smart_attention_executor import SmartAttentionExecutor

class BenchmarkExecutor(SmartAttentionExecutor):
    def execute_forced_bench(self, X, n, d, is_int8=0, force_mode='Auto', warmup_iters=2, run_iters=10):
        # 1. 전략 결정
        if force_mode == 'All_GPU': q_dev, a_dev = 'GPU', 'GPU'
        elif force_mode == 'All_NPU': q_dev, a_dev = 'NPU', 'NPU'
        else: q_dev, a_dev = self.predict_backend(n, d, is_int8)

        # 2. 가중치 준비
        W_type = np.int8 if is_int8 else np.float16
        # 메모리 절약을 위해 실제 가중치는 랜덤 생성 (Shape만 맞춤)
        try:
            W_QKV = np.random.randn(d, 3*d).astype(W_type)
            X_in = X.astype(W_type)
        except np.core._exceptions.MemoryError:
            print(f"[Warning] OOM during weight alloc for N={n}, D={d}")
            return None

        # [Warm-up]
        try:
            for _ in range(warmup_iters):
                q_out = None
                if q_dev == 'NPU': q_out = self._run_qkv_npu(X_in, W_QKV, n, d, is_int8)
                else: q_out = self._run_qkv_gpu(X_in, W_QKV, n, d, is_int8)
                
                Q, K, V = np.split(q_out, 3, axis=1)
                if q_dev != a_dev and is_int8 and a_dev == 'GPU':
                        Q, K, V = Q.astype(np.float16), K.astype(np.float16), V.astype(np.float16)

                if a_dev == 'NPU': self._run_attn_npu(Q, K, V, n, d, is_int8)
                else: self._run_attn_gpu(Q, K, V, n, d, is_int8)
        except Exception as e:
            print(f"[Error] Warmup failed: {e}")
            return None

        # [Measure]
        times_total = []
        for _ in range(run_iters):
            t0 = time.time()
            try:
                # 1. QKV
                qkv_out = None
                if q_dev == 'NPU': qkv_out = self._run_qkv_npu(X_in, W_QKV, n, d, is_int8)
                else: qkv_out = self._run_qkv_gpu(X_in, W_QKV, n, d, is_int8)
                t1 = time.time()
                
                # 2. Bridge
                Q, K, V = np.split(qkv_out, 3, axis=1)
                if q_dev != a_dev:
                    if is_int8 and a_dev == 'GPU':
                            Q, K, V = Q.astype(np.float16), K.astype(np.float16), V.astype(np.float16)
                
                # 3. Attn
                if a_dev == 'NPU': self._run_attn_npu(Q, K, V, n, d, is_int8)
                else: self._run_attn_gpu(Q, K, V, n, d, is_int8)
                t2 = time.time()
                
                times_total.append((t2 - t0) * 1000.0)
            except Exception as e:
                print(f"[Error] Execution failed at iter: {e}")
                return None

        if not times_total: return None

        return {
            'total_min': min(times_total), 
            'plan': (q_dev, a_dev)
        }

def run_multidim_benchmark():
    executor = BenchmarkExecutor()
    results = []

    print("Running Multi-Dimension Realistic Benchmarks...")

    # =========================================================
    # [Exp 1] Prefill Phase Sweep (N Sweep for Multiple Ds)
    # 목적: 모델 크기별로 GPU가 역전하는 N 지점이 다른지 확인
    # =========================================================
    
    # 1.5B(Qwen), 2B(Gemma), 8B(Llama3) 대표 차원
    TARGET_D_LIST = [1536, 2048, 4096] 
    seq_range = [32, 40, 64, 100, 128, 180, 256, 380, 512, 725, 1024, 1500] 
    
    print(f"\n[Exp 1] Prefill Phase - Multi-D Sweep {TARGET_D_LIST}")
    
    for d in TARGET_D_LIST:
        print(f"  -> Testing Model D={d}...")
        for n in tqdm(seq_range, leave=False):
            dummy_input = np.random.randn(n, d).astype(np.float16)
            
            # Run All Modes
            res_gpu = executor.execute_forced_bench(dummy_input, n, d, 0, 'All_GPU')
            res_npu = executor.execute_forced_bench(dummy_input, n, d, 0, 'All_NPU')
            res_smart = executor.execute_forced_bench(dummy_input, n, d, 0, 'Auto')
            
            if res_gpu and res_npu and res_smart:
                times = {'GPU': res_gpu['total_min'], 'NPU': res_npu['total_min'], 'Smart': res_smart['total_min']}
                winner = min(times, key=times.get)
                speedup = res_gpu['total_min'] / times[winner]

                results.append({
                    'exp_type': 'prefill_multidim',
                    'n': n, 'd': d, 'int8': 0,
                    'gpu_lat': res_gpu['total_min'], 'npu_lat': res_npu['total_min'], 'smart_lat': res_smart['total_min'],
                    'winner': winner, 'speedup': speedup, 'plan': str(res_smart['plan'])
                })

    # =========================================================
    # [Exp 2] Decode Phase (N=1) - Detailed D Sweep
    # 목적: 아주 작은 D부터 큰 D까지 추세를 부드럽게 그리기 위함
    # =========================================================
    fixed_n = 1
    # 촘촘한 D 리스트
    decode_dims = [768, 1024, 1536, 2048, 2560, 3072, 4096]
    
    print(f"\n[Exp 2] Decode Efficiency (N=1) - Detailed Sweep")
    
    for d in tqdm(decode_dims):
        dummy_input = np.random.randn(fixed_n, d).astype(np.float16)
        
        res_gpu = executor.execute_forced_bench(dummy_input, fixed_n, d, 0, 'All_GPU')
        res_npu = executor.execute_forced_bench(dummy_input, fixed_n, d, 0, 'All_NPU')
        res_smart = executor.execute_forced_bench(dummy_input, fixed_n, d, 0, 'Auto')

        if res_gpu and res_npu and res_smart:
            times = {'GPU': res_gpu['total_min'], 'NPU': res_npu['total_min'], 'Smart': res_smart['total_min']}
            winner = min(times, key=times.get)
            speedup = res_gpu['total_min'] / times[winner]

            results.append({
                'exp_type': 'decode_sweep',
                'n': fixed_n, 'd': d, 'int8': 0,
                'gpu_lat': res_gpu['total_min'], 'npu_lat': res_npu['total_min'], 'smart_lat': res_smart['total_min'],
                'winner': winner, 'speedup': speedup, 'plan': str(res_smart['plan'])
            })

    # =========================================================
    # [Exp 3] INT8 Effect on Multiple Sizes & Seq Lengths
    # =========================================================
    # INT8 효과가 모델 크기(D)와 시퀀스 길이(N)에 따라 어떻게 다른지 확인
    # N=1 (Decode)일 때의 대역폭 이득 vs N=512 (Prefill)일 때의 연산 이득 비교
    int8_targets = [1536, 2048, 4096] # 모델 크기 (D)
    int8_seqs = [1, 128, 512, 1024]   # 시퀀스 길이 (N) 리스트로 변경
    
    print(f"\n[Exp 3] INT8 Effect (N={int8_seqs}, D={int8_targets})")

    for d in tqdm(int8_targets, desc="Model Size"):
        for n in tqdm(int8_seqs, desc=f"Seq Len (D={d})", leave=False):
            
            dummy_input = np.random.randn(n, d).astype(np.float16)
            
            # 1. Baseline 측정: GPU FP16 (속도 향상의 기준점)
            # INT8이 얼마나 빠른지 비교하려면 '기존 방식(FP16 GPU)'이 필요함
            res_baseline = executor.execute_forced_bench(dummy_input, n, d, 0, 'All_GPU')
            
            # 2. INT8 모드별 측정 (GPU, NPU, Smart)
            # GPU INT8은 하드웨어/라이브러리 지원 여부에 따라 느리거나 실패할 수 있으나 기록을 위해 실행
            res_gpu_int8 = executor.execute_forced_bench(dummy_input, n, d, 1, 'All_GPU')
            res_npu_int8 = executor.execute_forced_bench(dummy_input, n, d, 1, 'All_NPU')
            res_smart_int8 = executor.execute_forced_bench(dummy_input, n, d, 1, 'Auto')
            
            # 데이터가 모두 유효한 경우에만 저장
            if res_baseline and res_gpu_int8 and res_npu_int8 and res_smart_int8:
                
                # Winner 결정 (INT8끼리 경쟁)
                times = {
                    'GPU_INT8': res_gpu_int8['total_min'], 
                    'NPU_INT8': res_npu_int8['total_min'], 
                    'Smart_INT8': res_smart_int8['total_min']
                }
                winner = min(times, key=times.get)
                
                # Speedup: 기준(GPU FP16) 대비 Smart INT8이 얼마나 빠른가?
                # 이것이 논문에 쓸 핵심 수치 (예: 4.5배)
                speedup = res_baseline['total_min'] / res_smart_int8['total_min']
                
                results.append({
                    'exp_type': 'int8_multidim_sweep',
                    'n': n, 'd': d, 'int8': 1,
                    # CSV에는 INT8 측정값들을 기록
                    'gpu_lat': res_gpu_int8['total_min'], 
                    'npu_lat': res_npu_int8['total_min'], 
                    'smart_lat': res_smart_int8['total_min'],
                    'winner': winner, 
                    'speedup': speedup, 
                    'plan': str(res_smart_int8['plan'])
                })

    # Save
    df = pd.DataFrame(results)
    df.to_csv("paper_results_multidim.csv", index=False)
    print("\n[Done] Results saved to 'paper_results_multidim.csv'")


if __name__ == "__main__":
    run_multidim_benchmark()
