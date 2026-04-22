# LLM Quantization Benchmark on RK3588

RK3588 SoC 위에서 양자화된 LLM의 추론 성능과 품질을 비교하는 벤치마크 프레임워크.
MNN (CPU/GPU) 과 RKNN-LLM (NPU) 두 가지 추론 백엔드를 동일 모델·동일 조건으로 평가한다.

## 타겟 모델

- **Llama-3.2-1B-Instruct** (primary)

## 디렉토리 구조

```
llm_quant_bench/
├── config/             # 실험 설정 (experiment_matrix.yaml)
├── models/             # HuggingFace 원본 모델
├── mnn_models/         # MNN 변환 모델 (M1-M8)
├── rkllm_models/       # RKLLM 변환 모델 (R1-R4)
├── calib_data/         # 양자화 캘리브레이션 데이터
├── export/             # 모델 변환 스크립트
├── benchmark/          # 벤치마크 실행 스크립트
├── eval/               # Perplexity 평가
├── analysis/           # 결과 분석 및 시각화
├── results/            # 실험 결과 (raw JSON, figures)
├── scripts/            # 자동화 셸 스크립트
└── colab_package/      # Google Colab용 RKLLM 변환 패키지
```

## 실험 설정

### MNN 설정 (M1-M8) — CPU + OpenCL GPU 백엔드

| Config | Bit | Block Size | Method | 설명 |
|--------|-----|------------|--------|------|
| M1 | 4 | 0 (channel) | direct | 4-bit channel-wise |
| M2 | 4 | 64 | direct | 4-bit block-64 |
| M3 | 4 | 128 | direct | 4-bit block-128 |
| M4 | 8 | 0 (channel) | direct | 8-bit channel-wise |
| M5 | 8 | 64 | direct | 8-bit block-64 |
| M6 | 4 | 64 | AWQ | Activation-aware Weight Quantization |
| M7 | 4 | 64 | SmoothQuant | SmoothQuant |
| M8 | 4 | 64 | HQQ | Half-Quadratic Quantization |

### RKNN-LLM 설정 (R1-R8) — NPU 3-core 백엔드

| Config | dtype | Algorithm | Group Size | 설명 | RK3588 지원 |
|--------|-------|-----------|------------|------|-------------|
| R1 | W8A8 | normal | None (channel) | 8-bit channel-wise | O |
| R2 | W8A8 | normal | 128 | 8-bit group-128 | O |
| R3 | W8A8 | normal | 256 | 8-bit group-256 | O |
| R4 | W8A8 | normal | 512 | 8-bit group-512 | O |
| R5 | W4A16 | GRQ | None (channel) | 4-bit GRQ channel-wise | X |
| R6 | W4A16 | GRQ | 32 | 4-bit GRQ group-32 | X |
| R7 | W4A16 | GRQ | 64 | 4-bit GRQ group-64 | X |
| R8 | W4A16 | GRQ | 128 | 4-bit GRQ group-128 | X |

### RK3588 양자화 지원 제한사항

rkllm-toolkit은 두 가지 양자화 스킴을 제공한다:

| Scheme | 설명 | 권장 Algorithm |
|--------|------|---------------|
| W8A8 | 8-bit weights, 8-bit activations | normal |
| W4A16 | 4-bit weights, 16-bit activations | GRQ |

단, **RK3588 NPU 하드웨어는 W8A8만 지원**한다. W4A16은 rkllm-toolkit 레벨에서는 지원되지만 RK3588 타겟으로 빌드 시 다음 에러가 발생한다:

```
ERROR: target_platform: rk3588 not support quantized_dtype: w4a16!
```

W4A16은 **RK3576** 등 신규 SoC에서만 사용 가능하다. 따라서 RK3588 환경에서 RKLLM 실험은 **R1-R4 (W8A8 계열)만 유효**하다.

## 벤치마크 파라미터

- Prompt 길이: 64, 256 tokens
- 생성 길이: 256 tokens
- Warmup: 5회
- 측정: 10회 (평균 + 표준편차)
- Max context: 4096 tokens

## 측정 지표

- **TTFT** (Time to First Token): 첫 토큰 생성 지연 시간 (ms)
- **Decode Speed**: 토큰 생성 속도 (tokens/sec)
- **Perplexity**: wikitext-2 기반 모델 품질 지표
- **Model Size**: 디스크 용량 (MB)
- **Peak Memory**: 추론 시 최대 메모리 사용량

## 실험 결과

### MNN CPU 벤치마크 (4 threads)

| Config | Size (MB) | Prefill 64 (tok/s) | Prefill 256 (tok/s) | Decode (tok/s) | PPL |
|--------|-----------|--------------------:|--------------------:|---------------:|----:|
| M1 (4b ch) | 595.8 | 130.13 | 115.68 | 29.89 | 35.76 |
| M2 (4b b64) | 739.2 | 94.00 | 87.19 | 24.59 | 22.07 |
| M3 (4b b128) | 665.6 | 112.22 | 99.03 | 25.61 | 23.62 |
| M4 (8b ch) | 1185.1 | 152.53 | 144.28 | 15.63 | 19.75 |
| M5 (8b b64) | 1328.5 | 109.53 | 104.37 | 14.01 | 19.76 |
| M6 (4b AWQ) | 739.2 | 96.96 | 87.32 | 24.23 | 21.78 |
| M7 (4b Smooth) | 739.8 | 91.39 | 78.07 | 23.83 | 22.62 |
| M8 (4b HQQ) | 739.2 | 93.56 | 85.87 | 26.95 | 23.09 |

### MNN OpenCL GPU 벤치마크

| Config | Size (MB) | Prefill 64 (tok/s) | Prefill 256 (tok/s) | Decode (tok/s) |
|--------|-----------|--------------------:|--------------------:|---------------:|
| M1 (4b ch) | 593.6 | 22.26 ± 1.49 | 21.29 ± 2.05 | 7.72 ± 0.36 |
| M2 (4b b64) | 737.1 | 16.31 ± 0.94 | 16.34 ± 1.10 | 6.83 ± 0.73 |
| M3 (4b b128) | 663.4 | 18.86 ± 1.06 | 18.22 ± 1.13 | 7.55 ± 0.37 |
| M4 (8b ch) | 1187.8 | 17.23 ± 1.08 | 20.85 ± 1.14 | 5.34 ± 0.10 |
| M5 (8b b64) | 1331.2 | 15.15 ± 2.34 | 16.65 ± 1.16 | 5.31 ± 0.30 |
| M6 (4b AWQ) | 737.1 | 15.98 ± 1.16 | 17.25 ± 1.18 | 7.02 ± 0.28 |
| M7 (4b Smooth) | 737.1 | 15.67 ± 1.41 | 15.03 ± 0.97 | 6.97 ± 0.40 |
| M8 (4b HQQ) | 737.1 | 19.44 ± 1.61 | 15.83 ± 0.64 | 7.84 ± 0.53 |

> CPU가 GPU(OpenCL)보다 전반적으로 빠르다. RK3588의 Mali GPU는 LLM 추론에서 CPU 대비 이점이 없음.

### RKNN-LLM NPU 벤치마크 (3-core)

| Config | Size (MB) | TTFT 64 (ms) | TTFT 256 (ms) | Decode p64 (tok/s) | Decode p256 (tok/s) |
|--------|-----------|-------------:|--------------:|-------------------:|--------------------:|
| R1 (W8A8 ch) | 1704 | 49.4 | 51.5 | 19.97 | 19.07 |
| R2 (W8A8 G128) | 1798 | 78.3 | 80.4 | 12.39 | 12.16 |
| R3 (W8A8 G256) | 1750 | 63.9 | 66.2 | 15.22 | 14.76 |
| R4 (W8A8 G512) | 1727 | 59.5 | 62.2 | 16.30 | 15.71 |
| R5-R8 (W4A16) | - | - | - | - | - |

> R5-R8은 RK3588 W4A16 미지원으로 변환 실패. RKLLM PPL 평가는 chunk 30 부근에서 에러 발생하여 미완료.

### 주요 관찰

- **CPU vs GPU**: MNN CPU가 OpenCL GPU보다 3~5배 빠름 (decode 기준 ~25 vs ~7 tok/s)
- **CPU vs NPU**: NPU R1(W8A8 ch)의 decode 속도(~20 tok/s)는 MNN 4-bit CPU(~25-30 tok/s)보다 느리나, 8-bit CPU(~15 tok/s)와 유사
- **NPU TTFT**: R1이 49ms로 가장 빠르고, group size가 커질수록 느려짐
- **양자화 품질**: 8-bit(PPL ~19.75)이 4-bit(PPL ~22-36)보다 품질이 좋음. 4-bit 중에서는 AWQ(21.78)가 가장 우수
- **NPU group size**: channel-wise(R1)가 속도·TTFT 모두 가장 우수. group size가 작을수록 모델은 커지고 느려짐

## 사용법

### 1. 캘리브레이션 데이터 생성

```bash
python export/generate_calib_data.py
```

### 2. 모델 변환

```bash
# MNN 변환 (로컬)
python export/export_mnn.py --model_path ./models/Llama-3.2-1B-Instruct --run_all

# RKLLM 변환 (Colab 권장 — GPU 필요)
python export/export_rkllm.py --model_path /path/to/model --run_all
```

### 3. 벤치마크 실행

```bash
# 전체 파이프라인
python benchmark/run_all.py \
    --mnn_models_dir ./mnn_models \
    --rkllm_models_dir ./rkllm_models \
    --tokenizer_path ./models/Llama-3.2-1B-Instruct

# 개별 실행
python benchmark/bench_mnn.py --all_models_dir ./mnn_models --backends cpu opencl
python benchmark/bench_rkllm.py --model_dir ./rkllm_models
```

### 4. Perplexity 평가

```bash
# MNN
python eval/ppl_mnn.py --all_models_dir ./mnn_models

# RKLLM
python eval/ppl_rkllm.py --model_path ./rkllm_models/model.rkllm \
    --tokenizer_path ./models/Llama-3.2-1B-Instruct
```

### 5. 결과 분석

```bash
python analysis/collect_results.py
python analysis/generate_plots.py
```

## 외부 의존성

- **MNN**: `~/install_files/MNN/` (빌드: `scripts/build_mnn_llm.sh`)
- **RKNN-LLM Runtime**: `~/install_files/rknn-llm/` (`librkllmrt.so`)
- **Python**: transformers, torch, datasets, matplotlib, pandas, numpy
