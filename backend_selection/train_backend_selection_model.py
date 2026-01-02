import pandas as pd
import os
import sys
import math
from pycaret.classification import *

# === 설정 ===
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
CSV_PATH = os.path.join(PROJECT_ROOT, 'benchmark', 'attention_profile_data.csv')



def get_data_with_targets():
    """데이터를 로드하고 QKV용, Attention용 타겟을 각각 생성"""
    if not os.path.exists(CSV_PATH):
        print(f"[Error] CSV not found: {CSV_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(CSV_PATH)
    print(f"Data Loaded: {df.shape}")
    
    # 1. QKV 승자 (0: GPU, 1: NPU)
    df['best_qkv_class'] = df.apply(lambda r: 1 if r['npu_qkv'] < r['gpu_qkv'] else 0, axis=1)
    
    # 2. Attention 승자 (0: GPU, 1: NPU)
    # (Fused Kernel 기준 비교)
    df['best_attn_class'] = df.apply(lambda r: 1 if r['npu_attn_fused'] < r['gpu_attn_fused'] else 0, axis=1)
    
    return df

def train_component_model(df, target_col, model_name):
    """
    특정 타겟(QKV 또는 Attn)을 위한 이진 분류 모델 학습
    """
    print(f"\n=== Training Model for: {target_col} ===")
    
    # 학습에 방해되는(정답 유출) 모든 레이턴시 관련 컬럼 제거
    ignore_cols = [
        'latency_gpu', 'latency_npu', 'latency_gpu_fused', 'latency_npu_fused',
        'gpu_qk', 'npu_qk', 'gpu_sm', 'npu_sm',
        'gpu_qkv', 'npu_qkv', 'gpu_av', 'npu_av',
        'speedup_ratio', 'min_latency', 'best_backend', 'best_backend_name',
        'best_device_class', 
        'gpu_attn_fused', 'npu_attn_fused',
        'best_qkv_class', 'best_attn_class' # 타겟 후보들도 일단 다 제거
    ]
    
    # 현재 타겟이 ignore 목록에 있다면 살려둠 (학습해야 하니까)
    cols_to_drop = [c for c in ignore_cols if c in df.columns and c != target_col]
    data = df.drop(columns=cols_to_drop)
    
    # PyCaret Setup
    exp = setup(
        data=data, 
        target=target_col, 
        session_id=42, 
        verbose=False, 
        html=False, 
        normalize=True
    )
    
    # 모델 비교 & 튜닝 (LightGBM 권장)
    best = compare_models(sort='Accuracy', n_select=1, turbo=True)
    tuned = tune_model(best, optimize='Accuracy')
    final = finalize_model(tuned)
    
    # 저장
    save_path = os.path.join(HERE, model_name)
    save_model(final, save_path)
    print(f"[Done] Saved {model_name}.pkl")
    return final

def predict_hybrid(input_data, model_qkv = None, model_attn = None):
    """
    두 모델을 각각 돌려서 결과를 조합하는 함수
    """
    if model_qkv is None:
        model_qkv = load_model(os.path.join(HERE, 'selector_qkv'), verbose=False)
    if model_attn is None:
        model_attn = load_model(os.path.join(HERE, 'selector_attn'), verbose=False)

    # 1. QKV 예측
    pred_q = predict_model(model_qkv, data=input_data, verbose=False)
    # PyCaret 버전에 따라 결과 컬럼명이 다를 수 있음 처리
    cls_q = pred_q.iloc[0]['prediction_label'] if 'prediction_label' in pred_q else pred_q.iloc[0]['Label']
    res_q = "NPU" if cls_q == 1 else "GPU"

    # 2. Attention 예측
    pred_a = predict_model(model_attn, data=input_data, verbose=False)
    cls_a = pred_a.iloc[0]['prediction_label'] if 'prediction_label' in pred_a else pred_a.iloc[0]['Label']
    res_a = "NPU" if cls_a == 1 else "GPU"
    
    return res_q, res_a

def prepare_inputs(n, d, is_int8=0):
    """입력 특성 계산 함수 (추론용)"""
    flops = 2 * (4*n*d*d + 2*n*n*d)
    mem = (n*d + 3*d*d + 3*n*d + n*n + n*d) * 2

    def calculate_overhead(x, alignment=64):
        return (math.ceil(x / alignment) * alignment) - x

    pad_n = calculate_overhead(n, alignment=64) # 혹은 32
    pad_d = calculate_overhead(d, alignment=64)

    input_data = pd.DataFrame([{
        'n': n,
        'd': d,
        'is_int8': is_int8,
        'flops': flops,
        'memory_bytes': mem,
        'arithmetic_intensity': flops / (mem + 1e-9),
        'aspect_ratio': n / d,
        'score_matrix_size': n * n,
        'pad_overhead_n': pad_n, 'pad_overhead_d': pad_d
    }])
    return input_data

if __name__ == "__main__":
    # 1. 데이터 준비
    df = get_data_with_targets()
    
    # 2. QKV 모델 학습 및 저장
    #model_qkv = train_component_model(df, target_col='best_qkv_class', model_name='selector_qkv')
    
    # 3. Attn 모델 학습 및 저장
    #model_attn = train_component_model(df, target_col='best_attn_class', model_name='selector_attn')
    
    # 4. 추론 테스트
    print("\n" + "="*40)
    print("      SEPARATED MODEL INFERENCE")
    print("="*40)
    
    test_cases = [
        (1, 70), (1, 300), (1, 500), (1, 800), (1, 1200), (1, 1500), (1, 2000), (1, 4000),
        (20, 70), (20, 300), (20, 1500), (20, 2000), (20, 4000),
        (100, 70), (100, 300), (100, 1500), (100, 2000), (100, 4000),
        (500, 70), (500, 300), (500, 1500), (500, 2000), (500, 4000),
        (1400, 70), (1400, 300), (1400, 1500), (1400, 2000), (1400, 4000),
        (2000, 70), (2000, 300), (2000, 1500), (2000, 2000), (2000, 4000),
    ]

    for n, d in test_cases:
        
        # Feature 자동 계산
        input_data_f16 = prepare_inputs(n, d, 0)
        input_data_i8 = prepare_inputs(n, d, 1)
        
        # 분리된 모델로 각각 예측
        #qkv_f16, attn_f16 = predict_hybrid(input_data_f16, model_qkv, model_attn)
        #qkv_i8, attn_i8 = predict_hybrid(input_data_i8, model_qkv, model_attn)
        qkv_f16, attn_f16 = predict_hybrid(input_data_f16)
        qkv_i8, attn_i8 = predict_hybrid(input_data_i8)
        
        print(f"[Input] N={n}, D={d}")
        print(f"   -> QKV(FP16) : {qkv_f16}")
        print(f"   -> Attn(FP16): {attn_f16}")
        print(f"   -> QKV(INT8) : {qkv_i8}")
        print(f"   -> Attn(INT8): {attn_i8}")
        
        # [중요] 여기서 하이브리드 오버헤드 로직을 추가할 수 있음
        if qkv_f16 != attn_f16:
            print(f"   => ★ Hybrid Mode Recommended(FP16): {qkv_f16} -> {attn_f16}")
            # 예시: "데이터가 너무 작으면 그냥 QKV를 따라가라" 같은 룰 추가 가능
            # if (n*d*2) < 100000: print("      (Warning: Small data, maybe force Single Mode)")
        else:
            print(f"   => Single Mode(FP16): All {qkv_f16}")
        if qkv_i8 != attn_i8:
            print(f"   => ★ Hybrid Mode Recommended(INT8): {qkv_i8} -> {attn_i8}")
            # 예시: "데이터가 너무 작으면 그냥 QKV를 따라가라" 같은 룰 추가 가능
            # if (n*d*2) < 100000: print("      (Warning: Small data, maybe force Single Mode)")
        else:
            print(f"   => Single Mode: All(INT8) {qkv_i8}")
        print("-" * 30)
