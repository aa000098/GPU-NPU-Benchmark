# collect_data.py
import pandas as pd
import numpy as np
import time
import os
import math
from datetime import datetime
from tqdm import tqdm  # 진행률 표시 (없으면 pip install tqdm)

# 수정한 profile_attention 모듈 import
from profile_attention_performance import profile_attention, profile_attention_i8

def get_padding_overhead(val, align=32):
    """패딩으로 인해 낭비되는 비율 계산"""
    aligned_val = math.ceil(val / align) * align
    if val == 0: return 0
    return (aligned_val - val) / aligned_val

def collect_benchmark_data(filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attention_profile_data_{timestamp}.csv"
        #print(f"Output Filename: {filename}")

    # === 1. 탐색할 공간 설정 (Sweep Space) ===
    # 시퀀스 길이 (n)
    seq_lens = [1, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 4096]
    seq_lens += [33, 65, 129, 257, 385, 513, 769, 1025, 1537, 2049, 4097] # (선택) 정렬 안된 데이터도 테스트하려면 주석 해제
    
    # 히든 차원 (d = head_dim)
    # 32의 배수와 배수가 아닌 수를 섞어서 Alignment 효과 확인
    #head_dims = [32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 4096]
    #head_dims += [33, 65, 129, 257, 385, 513, 769, 1025, 1537, 2049, 4097] # (선택) 정렬 안된 데이터도 테스트하려면 주석 해제
    head_dims = [34, 66, 130, 258, 386, 514, 770, 1026, 1538, 2050, 2060, 2070, 2080, 2090, 2100, 2200, 2300, 2400, 2500, 2600, 2800, 3000, 3500, 8000] # (선택) 정렬 안된 데이터도 테스트하려면 주석 해제

    print(f"Total Combinations: {len(seq_lens) * len(head_dims)}")
    print(f"Total Data Rows (x2 for INT8/FP16): {len(seq_lens) * len(head_dims) * 2}")
   
    processed_combinations = set()
    file_exists = os.path.isfile(filename)

    if file_exists:
        try:
            # 기존 파일을 읽어서 이미 완료한 (n, d) 조합을 찾음
            df_exist = pd.read_csv(filename)
            # n과 d 컬럼을 묶어서 set에 저장
            for _, row in df_exist.iterrows():
                processed_combinations.add((int(row['n']), int(row['d'])))
            print(f"Found existing data: {len(processed_combinations)//2} combinations already done. Resuming...")
        except Exception as e:
            print(f"[Warning] Could not read existing file: {e}. Starting fresh.")
            file_exists = False # 읽기 실패하면 덮어쓰거나 새로 만듦

    # === 2. 벤치마크 루프 ===
    for n in tqdm(seq_lens, desc="Collecting Data"):
        for d in head_dims:

            if (n, d) in processed_combinations:
                continue

            try:
                # (1) 측정 실행
                gpu_fp16, npu_fp16 = profile_attention(seq=n, head_dim=d)
                gpu_int8, npu_int8 = profile_attention_i8(seq=n, head_dim=d)

                # 공통 피처 계산
                flops = 4*n*(d**2) + 4*(n**2)*d 
                total_elements = (n*d) + (3*d*d) + (3*n*d) + (n*n) + (n*d)
                
                aspect_ratio = n / d if d > 0 else 0
                score_matrix_size = n * n
                pad_overhead_n = get_padding_overhead(n, 32)
                pad_overhead_d = get_padding_overhead(d, 32)

                current_batch = []

                # ==========================================
                # Row 1: FP16 데이터 생성
                # ==========================================
                mem_fp16 = total_elements * 2 # 2 bytes
                intensity_fp16 = flops / mem_fp16 if mem_fp16 > 0 else 0
                
                row_fp16 = {
                    # Input Features
                    "n": n,
                    "d": d,
                    "is_int8": 0,  # <--- 구분자
                    "flops": flops,
                    "memory_bytes": mem_fp16, # FP16 기준
                    "arithmetic_intensity": intensity_fp16,
                    "aspect_ratio": aspect_ratio,
                    "score_matrix_size": score_matrix_size,
                    "pad_overhead_n": pad_overhead_n,
                    "pad_overhead_d": pad_overhead_d,

                    # Latency Targets
                    "latency_gpu": gpu_fp16["total"],
                    "latency_npu": npu_fp16["total"],
                    "latency_gpu_fused": gpu_fp16["fused_total"],
                    "latency_npu_fused": npu_fp16["fused_total"],
                    
                    # Component Breakdown (병목 분석용)
                    "gpu_qkv": gpu_fp16["qkv"],
                    "npu_qkv": npu_fp16["qkv"],
                    "gpu_qk": gpu_fp16["qk"],
                    "npu_qk": npu_fp16["qk"],
                    "gpu_sm": gpu_fp16["softmax"],
                    "npu_sm": npu_fp16["softmax"],
                    "gpu_av": gpu_fp16["av"],
                    "npu_av": npu_fp16["av"],
                    "gpu_attn_fused": gpu_fp16["fused_core"],
                    "npu_attn_fused": npu_fp16["fused_core"]
                }
                
                # Labeling: 누가 이겼나? (0: GPU, 1: NPU)
                row_fp16["best_device_class"] = 0 if row_fp16["latency_gpu_fused"] < row_fp16["latency_npu_fused"] else 1
                row_fp16["speedup_ratio"] = row_fp16["latency_gpu_fused"] / row_fp16["latency_npu_fused"] # >1 이면 NPU 승

                current_batch.append(row_fp16)

                # ==========================================
                # Row 2: INT8 데이터 생성
                # ==========================================
                mem_int8 = total_elements * 1 # 1 byte (approx)
                intensity_int8 = flops / mem_int8 if mem_int8 > 0 else 0 # FP16보다 2배 높음
                
                row_int8 = {
                    # Input Features
                    "n": n,
                    "d": d,
                    "is_int8": 1,  # <--- 구분자
                    "flops": flops,
                    "memory_bytes": mem_int8, # INT8 기준 (작아짐)
                    "arithmetic_intensity": intensity_int8, # 커짐 (Compute Bound 성향 강해짐)
                    "aspect_ratio": aspect_ratio,
                    "score_matrix_size": score_matrix_size,
                    "pad_overhead_n": pad_overhead_n,
                    "pad_overhead_d": pad_overhead_d,

                    # Latency Targets
                    "latency_gpu": gpu_int8["total"],
                    "latency_npu": npu_int8["total"],
                    "latency_gpu_fused": gpu_int8["fused_total"],
                    "latency_npu_fused": npu_int8["fused_total"],

                    # Component Breakdown
                    "gpu_qkv": gpu_int8["qkv"],
                    "npu_qkv": npu_int8["qkv"],
                    "gpu_qk": gpu_int8["qk"],
                    "npu_qk": npu_int8["qk"],
                    "gpu_sm": gpu_int8["softmax"],
                    "npu_sm": npu_int8["softmax"],
                    "gpu_av": gpu_int8["av"],
                    "npu_av": npu_int8["av"],
                    "gpu_attn_fused": gpu_int8["fused_core"],
                    "npu_attn_fused": npu_int8["fused_core"]
                }

                # Labeling
                row_int8["best_device_class"] = 0 if row_int8["latency_gpu_fused"] < row_int8["latency_npu_fused"] else 1
                row_int8["speedup_ratio"] = row_int8["latency_gpu_fused"] / row_int8["latency_npu_fused"]

                current_batch.append(row_int8)

                df_chunk = pd.DataFrame(current_batch)

                if not file_exists:
                    df_chunk.to_csv(filename, index=False, mode='w', header=True)
                    file_exists = True
                else:
                    df_chunk.to_csv(filename, index=False, mode='a', header=False)

            except Exception as e:
                print(f"Error at n={n}, d={d}: {e}")
                continue


if __name__ == "__main__":
    collect_benchmark_data()
