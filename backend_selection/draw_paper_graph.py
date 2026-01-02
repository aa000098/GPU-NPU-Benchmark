import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 설정: 저장할 폴더 이름 및 데이터 파일
# ==========================================
DATA_FILE = "paper_results_multidim.csv"
OUTPUT_DIR = "paper_images"  # 이미지를 저장할 폴더명

# 폴더가 없으면 생성 (있으면 무시)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} 파일을 찾을 수 없습니다. 같은 폴더에 위치시켜주세요.")
        return None
    return pd.read_csv(DATA_FILE)

# ==========================================
# 1. Prefill Phase Performance (Line Chart)
# ==========================================
def plot_prefill(df):
    # d=1536 (Qwen-1.5B급) 데이터만 필터링
    subset = df[(df['exp_type'] == 'prefill_multidim') & (df['d'] == 1536)].sort_values('n')
    
    plt.figure(figsize=(8, 5))
    
    plt.plot(subset['n'], subset['gpu_lat'], marker='o', label='GPU (FP16)', color='#1f77b4', linestyle='--')
    plt.plot(subset['n'], subset['npu_lat'], marker='s', label='NPU (FP16)', color='#ff7f0e', linestyle='--')
    plt.plot(subset['n'], subset['smart_lat'], marker='^', label='Hybrid (Proposed)', color='#2ca02c', linewidth=2.5)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Sequence Length ($n$)', fontsize=12, fontweight='bold')
    plt.ylabel('Latency (ms) - Log Scale', fontsize=12, fontweight='bold')
    plt.title('Fig 1. Prefill Phase Performance ($d=1536$)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="-", alpha=0.4)
    
    plt.xticks(subset['n'], subset['n'], rotation=45)
    
    # 폴더 경로와 결합하여 저장
    save_path = os.path.join(OUTPUT_DIR, 'fig_prefill_performance.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Saved] {save_path}")

# ==========================================
# 2. Decode Phase Performance (Bar Chart)
# ==========================================
def plot_decode(df):
    subset = df[df['exp_type'] == 'decode_sweep'].sort_values('d')
    
    plt.figure(figsize=(8, 5))
    
    bar_width = 0.25
    index = np.arange(len(subset))
    
    plt.bar(index, subset['gpu_lat'], bar_width, label='GPU Only', color='#1f77b4', alpha=0.7)
    plt.bar(index + bar_width, subset['npu_lat'], bar_width, label='NPU Only', color='#ff7f0e', alpha=0.7)
    plt.bar(index + 2*bar_width, subset['smart_lat'], bar_width, label='Smart (Hybrid)', color='#2ca02c', edgecolor='black')
    
    plt.xlabel('Hidden Dimension ($d$)', fontsize=12, fontweight='bold')
    plt.ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    plt.title('Fig 2. Decode Phase Efficiency ($n=1$)', fontsize=14, fontweight='bold')
    plt.xticks(index + bar_width, subset['d'].astype(int))
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    save_path = os.path.join(OUTPUT_DIR, 'fig_decode_performance.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Saved] {save_path}")

# ==========================================
# 1. FP16 Decode Analysis (Hybrid Focus)
# ==========================================
def plot_decode_fp16(df):
    subset = df[df['exp_type'] == 'decode_sweep'].sort_values('d')
    
    if subset.empty: return

    plt.figure(figsize=(9, 6))
    
    x = np.arange(len(subset))
    width = 0.25
    
    plt.bar(x - width, subset['gpu_lat'], width, label='GPU (FP16)', color='#1f77b4', alpha=0.7)
    plt.bar(x, subset['npu_lat'], width, label='NPU (FP16)', color='#ff7f0e', alpha=0.7)
    plt.bar(x + width, subset['smart_lat'], width, label='Smart (Hybrid)', color='#2ca02c', edgecolor='black', linewidth=1.5)
    
    plt.xlabel('Hidden Dimension ($d$)', fontsize=12, fontweight='bold')
    plt.ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    plt.title('Decode Phase (FP16): Hybrid Efficiency', fontsize=14, fontweight='bold')
    plt.xticks(x, subset['d'].astype(int))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Highlight Hybrid Winner
    for i in range(len(subset)):
        row = subset.iloc[i]
        # Smart가 GPU보다 5% 이상 빠르면 강조
        if row['smart_lat'] < row['gpu_lat'] * 0.95:
            plt.text(x[i] + width, row['smart_lat'], "★", ha='center', va='bottom', color='red', fontsize=12)

    save_path = os.path.join(OUTPUT_DIR, 'fig_decode_fp16.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Saved] {save_path}")

# ==========================================
# 2. INT8 Decode Analysis (Quantization Focus)
# ==========================================
def plot_decode_int8(df):
    # n=1인 INT8 데이터 필터링
    subset = df[(df['exp_type'] == 'int8_multidim_sweep') & (df['n'] == 1)].sort_values('d')
    
    if subset.empty: 
        print("[Warning] INT8 데이터가 없습니다.")
        return

    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(subset))
    width = 0.25  # 막대 너비 조정
    
    # 1. Baseline: GPU FP16 (비교 기준)
    # 2. NPU Only: INT8 NPU (주로 빠름)
    # 3. Smart: Hybrid INT8 (특정 구간 최적화 가능성)
    
    rects1 = plt.bar(x - width, subset['gpu_lat'], width, label='Baseline (GPU FP16)', color='gray', alpha=0.5)
    rects2 = plt.bar(x, subset['npu_lat'], width, label='NPU Only (INT8)', color='#ff7f0e', alpha=0.8) # 주황색
    rects3 = plt.bar(x + width, subset['smart_lat'], width, label='Smart (Hybrid INT8)', color='#9467bd', edgecolor='black', linewidth=1.5) # 보라색 강조
    
    plt.xlabel('Hidden Dimension ($d$)', fontsize=12, fontweight='bold')
    plt.ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    plt.title('Decode Phase (INT8): Quantization & Hybrid Efficiency', fontsize=14, fontweight='bold')
    plt.xticks(x, subset['d'].astype(int))
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Smart(Hybrid) 막대 위에 Speedup 표시 (Baseline 대비)
    for i in range(len(subset)):
        base = subset.iloc[i]['gpu_lat']     # FP16 GPU
        smart = subset.iloc[i]['smart_lat']  # Hybrid INT8
        
        # 0.0인 경우(에러 등) 제외하고 계산
        if smart > 0:
            speedup = base / smart
            # 막대 위에 텍스트 표시
            plt.text(x[i] + width, smart, f"{speedup:.1f}x", 
                     ha='center', va='bottom', fontweight='bold', fontsize=10, color='purple')

    save_path = os.path.join(OUTPUT_DIR, 'fig_decode_int8.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Saved] {save_path}")

def plot_decode_fp16_hybrid(df):
    # 1. Decode(n=1) & FP16 데이터 필터링
    subset = df[df['exp_type'] == 'decode_sweep'].copy()
    
    # 2. 하이브리드 플랜만 추출 (plan 문자열에 GPU, NPU 둘 다 있는 경우)
    # 예: "('NPU', 'GPU')" -> True
    subset['is_hybrid'] = subset['plan'].apply(lambda x: ('GPU' in str(x)) and ('NPU' in str(x)))
    hybrid_subset = subset[subset['is_hybrid']].sort_values('d')
    
    if hybrid_subset.empty:
        print("[Warning] FP16 Decode 단계에서 하이브리드(Mixed) 플랜이 선택된 케이스가 없습니다.")
        return

    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(hybrid_subset))
    width = 0.25
    
    # 막대 그리기
    rects1 = plt.bar(x - width, hybrid_subset['gpu_lat'], width, label='GPU Only', color='#1f77b4', alpha=0.6)
    rects2 = plt.bar(x, hybrid_subset['npu_lat'], width, label='NPU Only', color='#ff7f0e', alpha=0.6)
    # Smart(Hybrid) 강조
    rects3 = plt.bar(x + width, hybrid_subset['smart_lat'], width, label='Hybrid', color='#2ca02c', edgecolor='black', linewidth=2)
    
    plt.xlabel('Hidden Dimension ($d$)', fontsize=12, fontweight='bold')
    plt.ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    plt.title('Fig 2-a. Decode Phase (FP16): Selected Hybrid Cases', fontsize=14, fontweight='bold')
    
    # X축 라벨 (차원 d)
    plt.xticks(x, hybrid_subset['d'].astype(int))
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 상세 정보 표시 (Plan & Speedup)
    for i in range(len(hybrid_subset)):
        row = hybrid_subset.iloc[i]
        
        # 1. Plan 표시 (예: NPU->GPU)
        plan_str = row['plan'].replace("'", "").replace("(", "").replace(")", "").replace(", ", "→")
        plt.text(x[i] + width, row['smart_lat'], plan_str, 
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkgreen')
        
        # 2. Speedup 표시 (단일 기기 중 빠른 놈 대비)
        # Baseline = min(GPU, NPU)
        baseline = min(row['gpu_lat'], row['npu_lat'])
        speedup = baseline / row['smart_lat']
        
        # 막대 중간에 하얀색으로 표시
        plt.text(x[i] + width, row['smart_lat'] * 0.5, f"{speedup:.2f}x", 
                 ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    save_path = os.path.join(OUTPUT_DIR, 'fig_decode_fp16_hybrid.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Saved] {save_path}")

def plot_decode_int8_hybrid(df):
    # 1. INT8 Decode(n=1) 데이터 필터링
    subset = df[(df['exp_type'] == 'int8_multidim_sweep') & (df['n'] == 1)].copy()
    
    # 2. 하이브리드 플랜만 추출
    subset['is_hybrid'] = subset['plan'].apply(lambda x: ('GPU' in str(x)) and ('NPU' in str(x)))
    hybrid_subset = subset[subset['is_hybrid']].sort_values('d')
    
    if hybrid_subset.empty:
        print("[Warning] INT8 Decode 단계에서 하이브리드(Mixed) 플랜이 선택된 케이스가 없습니다.")
        return

    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(hybrid_subset))
    width = 0.25
    
    # FP16 GPU 데이터 매칭 (Baseline 비교용)
    # 주의: 이 부분은 전체 df에서 찾아와야 함
    fp16_subset = df[df['exp_type'] == 'decode_sweep'].set_index('d')['gpu_lat']
    
    # 막대 그리기
    # Baseline: 해당 차원(d)의 FP16 GPU 성능 (없으면 현재 행의 GPU INT8 사용)
    baselines = hybrid_subset['gpu_lat'].values
    #for d_val in hybrid_subset['d']:
    #    if d_val in fp16_subset.index:
    #        baselines.append(fp16_subset[d_val])
    #    else:
    #        baselines.append(0) # 데이터 없음
            
    #rects1 = plt.bar(x - width, baselines, width, label='GPU Only (Int8)', color='gray', alpha=0.5)
    rects1 = plt.bar(x - width, hybrid_subset['gpu_lat'], width, label='GPU Only (Int8)', color='gray', alpha=0.5)
    rects2 = plt.bar(x, hybrid_subset['npu_lat'], width, label='NPU Only (INT8)', color='#ff7f0e', alpha=0.6)
    rects3 = plt.bar(x + width, hybrid_subset['smart_lat'], width, label='Hybrid (INT8)', color='#9467bd', edgecolor='black', linewidth=2)
    
    plt.xlabel('Hidden Dimension ($d$)', fontsize=12, fontweight='bold')
    plt.ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    plt.title('Fig 2-b. Decode Phase (INT8): Selected Hybrid Cases', fontsize=14, fontweight='bold')
    
    plt.xticks(x, hybrid_subset['d'].astype(int))
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 상세 정보 표시
    for i in range(len(hybrid_subset)):
        row = hybrid_subset.iloc[i]
        base_val = baselines[i]
        
        # 1. Plan 표시
        plan_str = row['plan'].replace("'", "").replace("(", "").replace(")", "").replace(", ", "→")
        plt.text(x[i] + width, row['smart_lat'], plan_str, 
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='purple')
        
        # 2. Speedup 표시 (FP16 GPU 대비)
        if base_val > 0:
            speedup = base_val / row['smart_lat']
            plt.text(x[i] + width, row['smart_lat'] * 0.5, f"{speedup:.2f}x", 
                     ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    save_path = os.path.join(OUTPUT_DIR, 'fig_decode_int8_hybrid.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Saved] {save_path}")

def plot_hybrid_success_cases(df):
    # 1. 하이브리드 플랜 필터링 로직
    # plan 컬럼 문자열 안에 'GPU'와 'NPU'가 둘 다 포함되어 있으면 하이브리드로 간주
    # 예: "('NPU', 'GPU')" 또는 "('GPU', 'NPU')"
    df['is_hybrid'] = df['plan'].apply(lambda x: ('GPU' in str(x)) and ('NPU' in str(x)))
    
    # 하이브리드가 선택된 행만 추출
    subset = df[df['is_hybrid']].copy()
    
    # 정렬 (Exp Type -> d 순서)
    subset = subset.sort_values(by=['exp_type', 'd'])

    if subset.empty:
        print("[Warning] 하이브리드(Hybrid)가 선택된 케이스가 하나도 없습니다.")
        return

    # 2. 그래프 그리기
    plt.figure(figsize=(10, 6))
    
    # X축 라벨 생성 (실험 종류와 차원 d 표시)
    # 예: "Decode (d=1024)"
    labels = []
    for _, row in subset.iterrows():
        exp_name = "Decode" if "decode" in row['exp_type'] else "Prefill"
        if row['int8'] == 1:
            exp_name += " (INT8)"
        labels.append(f"{exp_name}\n$d={int(row['d'])}$")

    x = np.arange(len(subset))
    width = 0.25

    # 막대 3개 (GPU vs NPU vs Hybrid)
    rects1 = plt.bar(x - width, subset['gpu_lat'], width, label='GPU Only', color='gray', alpha=0.4)
    rects2 = plt.bar(x, subset['npu_lat'], width, label='NPU Only', color='orange', alpha=0.6)
    rects3 = plt.bar(x + width, subset['smart_lat'], width, label='Smart (Hybrid)', color='#2ca02c', edgecolor='black', linewidth=2)

    plt.xlabel('Selected Scenarios (Where Hybrid was Chosen)', fontsize=12, fontweight='bold')
    plt.ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    plt.title('Performance of Selected Hybrid Executions', fontsize=14, fontweight='bold')
    plt.xticks(x, labels, fontsize=11)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 3. 상세 정보 표시 (Plan & Speedup)
    for i in range(len(subset)):
        row = subset.iloc[i]
        
        # Smart Latency 위에 Plan 표시 (예: NPU->GPU)
        plan_str = row['plan'].replace("'", "").replace("(", "").replace(")", "").replace(", ", "→")
        plt.text(x[i] + width, row['smart_lat'], plan_str, 
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkgreen')
        
        # Baseline(보통 GPU) 대비 Speedup 표시
        # 만약 NPU가 더 빠르면 NPU 대비로 할 수도 있지만, 논문에서는 보통 GPU 기준 비교를 선호
        baseline = min(row['gpu_lat'], row['npu_lat']) # 단일 기기 중 가장 빠른 놈 기준
        speedup = baseline / row['smart_lat']
        
        # Hybrid 막대 중간에 Speedup 표시
        plt.text(x[i] + width, row['smart_lat']/2, f"{speedup:.2f}x", 
                 ha='center', va='center', color='white', fontweight='bold')

    save_path = os.path.join(OUTPUT_DIR, 'fig_hybrid_success_cases.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Saved] {save_path}")

# ==========================================
# 3. Precision Comparison (FP16 Best vs INT8 Best)
# ==========================================
def plot_decode_precision_compare(df):
    # FP16 최적값 (Smart FP16)
    fp16_data = df[df['exp_type'] == 'decode_sweep'][['d', 'smart_lat']].rename(columns={'smart_lat': 'fp16_lat'})
    # INT8 최적값 (Smart INT8)
    int8_data = df[(df['exp_type'] == 'int8_multidim_sweep') & (df['n'] == 1)][['d', 'smart_lat']].rename(columns={'smart_lat': 'int8_lat'})
    
    # Merge
    merged = pd.merge(fp16_data, int8_data, on='d').sort_values('d')
    
    if merged.empty: return

    plt.figure(figsize=(9, 6))
    
    plt.plot(merged['d'], merged['fp16_lat'], marker='o', label='Smart (FP16)', linewidth=2, linestyle='--', color='#2ca02c')
    plt.plot(merged['d'], merged['int8_lat'], marker='s', label='Smart (INT8)', linewidth=2, color='#d62728')
    
    plt.fill_between(merged['d'], merged['fp16_lat'], merged['int8_lat'], color='gray', alpha=0.1, label='Latency Reduction')
    
    plt.xlabel('Hidden Dimension ($d$)', fontsize=12, fontweight='bold')
    plt.ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    plt.title('Decode Performance: FP16 vs INT8', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.xticks(merged['d'], merged['d'].astype(int))
    
    save_path = os.path.join(OUTPUT_DIR, 'fig_decode_precision_compare.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Saved] {save_path}")

# ==========================================
# 3. INT8 Speedup (Bar Chart)
# ==========================================
def plot_int8(df):
    subset = df[(df['exp_type'] == 'int8_multidim_sweep') & (df['n'] == 1)].sort_values('d')
    speedups = subset['gpu_lat'] / subset['smart_lat']
    
    plt.figure(figsize=(8, 5))
    
    bars = plt.bar(subset['d'].astype(str), speedups, color='#9467bd', alpha=0.8, width=0.5)
    
    plt.xlabel('Hidden Dimension ($d$)', fontsize=12, fontweight='bold')
    plt.ylabel('Speedup (vs GPU FP16)', fontsize=12, fontweight='bold')
    plt.title('Fig 3. INT8 Quantization Speedup ($n=1$)', fontsize=14, fontweight='bold')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline (1.0x)')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    save_path = os.path.join(OUTPUT_DIR, 'fig_int8_speedup.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Saved] {save_path}")

# ==========================================
# 4. Summary Table Generation
# ==========================================
def draw_table(df):
    targets = [
        ("Prefill (Small Batch)", (df['exp_type'] == 'prefill_multidim') & (df['n'] == 40) & (df['d'] == 1536)),
        ("Prefill (Large Batch)", (df['exp_type'] == 'prefill_multidim') & (df['n'] == 512) & (df['d'] == 1536)),
        ("Decode (Hybrid Sweetspot)", (df['exp_type'] == 'decode_sweep') & (df['d'] == 1024)),
        ("Decode (Large Model)", (df['exp_type'] == 'decode_sweep') & (df['d'] == 1536)),
        ("INT8 Quantization", (df['exp_type'] == 'int8_multidim_sweep') & (df['d'] == 1536) & (df['n'] == 1))
    ]

    table_rows = []
    
    # INT8 비교를 위해 FP16 GPU Latency 추출 (없으면 Fallback)
    try:
        fp16_row = df[(df['exp_type'] == 'decode_sweep') & (df['d'] == 1536)].iloc[0]
        fp16_latency = fp16_row['gpu_lat']
    except:
        fp16_latency = 2.78

    for name, condition in targets:
        if condition.any():
            row = df[condition].iloc[0]
            
            if row['exp_type'] == 'int8_multidim_sweep':
                baseline = fp16_latency
                method = "INT8 NPU"
                proposed = row['smart_lat']
            else:
                baseline = row['gpu_lat']
                if row['winner'] == 'GPU':
                    proposed = row['gpu_lat']
                    method = "GPU"
                elif row['winner'] == 'NPU':
                    proposed = row['npu_lat']
                    method = "NPU"
                else: 
                    proposed = row['smart_lat']
                    method = "Hybrid"
            
            speedup = baseline / proposed
            
            cond_str = f"n={int(row['n'])}, d={int(row['d'])}"
            base_str = f"{baseline:.2f}"
            prop_str = f"{proposed:.2f} ({method})"
            spd_str = f"{speedup:.2f}x"
            
            table_rows.append([name, cond_str, base_str, prop_str, spd_str])

    summary_df = pd.DataFrame(table_rows, columns=["Scenario", "Condition", "Baseline [ms]", "Proposed [ms]", "Speedup"])

    def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=12,
                         header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                         bbox=[0, 0, 1, 1], header_columns=0, ax=None, **kwargs):
        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')

        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        for k, cell in mpl_table.get_celld().items():
            cell.set_edgecolor(edge_color)
            if k[0] == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors)])
                if k[1] == 4:
                    cell.set_text_props(weight='bold', color='black')
        return ax

    plt.figure(figsize=(12, 4))
    render_mpl_table(summary_df, header_columns=0, col_width=3.0)
    plt.title("Table 1. Experimental Results Summary on RK3588", fontsize=16, weight='bold', y=1.05)
    
    save_path = os.path.join(OUTPUT_DIR, 'fig_summary_table.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Saved] {save_path}")



# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(f"Saving images to folder: {OUTPUT_DIR}/ ...")
        plot_prefill(df)
        plot_decode(df)
        plot_decode_fp16(df)
        plot_decode_int8(df)
        plot_decode_fp16_hybrid(df)
        plot_decode_int8_hybrid(df)
        plot_hybrid_success_cases(df)
        plot_decode_precision_compare(df)
        plot_int8(df)
        draw_table(df)
        print("\nAll visualizations created successfully!")
