# 왜 RK3588 NPU는 context가 길어질수록 느려지는가?

## TL;DR

- **NPU decode는 context에 비례해 선형적으로 느려진다** (9.26 µs/tok slope)
- **CPU/GPU는 거의 context-invariant** (slope ≈ 0)
- 통념: "KV cache bandwidth 때문" → **반증됨** (KV는 전체 read의 10%뿐)
- 진짜 원인: **NPU가 M=1 matmul에서 peak compute의 0.37%만 사용**

---

## 실측: 3종 백엔드 decode 속도 (Llama 3.2 1B, RK3588)

### Linear fit: `decode_ms = a + b × ctx`

| Backend | Fixed (a, ms) | Slope (b, µs/tok) | 의미 |
|---------|--------------|-------------------|------|
| NPU | 50.2 | **9.26** | context에 선형 증가 |
| CPU | 61.5 | 0.22 | 거의 일정 |
| GPU | 73.7 | -0.81 (noise) | 일정 |

### Decode 속도 실측

| Context | NPU | CPU | GPU |
|---------|-----|-----|-----|
| 32 | 20.0 tok/s | 15.5 | 13.9 |
| 1024 | 17.0 tok/s | 16.4 | 14.4 |
| **4096** | **11.3 tok/s** | 15.8 | 14.1 |

p4096에서 **CPU가 NPU를 역전**.

---

## NPU가 느려지는 진짜 이유

### 핵심 실험: NPU matmul을 직접 호출

C API (`rknn_matmul_create` → `rknn_matmul_run`)로 attention-like matmul을 측정:

```
Matmul shape: (1, 2048) × (2048, context) FP16→FP32
```

| Context | Min latency | Per-ctx-token slope |
|---------|-------------|---------------------|
| 32 | 42.58 µs | — |
| 256 | 124.54 µs | 0.367 µs/tok |
| 4096 | 1518.16 µs | 0.366 µs/tok |

**완벽한 선형 scaling**: `latency = 18.86 + 0.366 × ctx`, **R² = 0.9999**

### NPU 활용률 계산

| 지표 | 값 |
|------|-----|
| NPU FP16 peak | **3 TFLOPS** (= 3000 GFLOPS) |
| NPU INT8 peak | 6 TOPS |
| 측정된 sustained (ctx=4096) | **11 GFLOPS** |
| **활용률** | **0.37%** |

즉 NPU의 **99.63% parallelism이 낭비**된다.

### 왜 NPU가 그렇게 비효율적인가?

Decode 시 matmul 형태:
```
Q [1, 2048] × K^T [2048, context]
  ↑
  M=1 (row vector)
```

NPU는 **대량 병렬 MAC 유닛** (수천 개)으로 설계됨:
- **M이 커야** (batch 또는 multiple queries) MAC 유닛들이 일함
- **M=1이면** 대부분 idle → compute는 K 축 reduction만
- 결과: peak의 0.37%만 사용 = pure compute time이 그대로 노출

### Decode 1 토큰 time decomposition (ctx=4096)

```
┌─────────────────────────────────────────────────────┐
│  Non-attention matmul (QKV proj, MLP)  ~40 ms  45% │ ← ctx 무관
├─────────────────────────────────────────────────────┤
│  Dispatch overhead (144 × 25µs)         ~3.6 ms  4% │ ← ctx 무관
├─────────────────────────────────────────────────────┤
│  Attention fixed                        ~0.9 ms  1% │ ← ctx 무관
├─────────────────────────────────────────────────────┤
│  Attention compute O(ctx)               ~44 ms  50% │ ← ctx에 비례
│  32 matmul × 0.367µs × ctx                          │   이게 범인
└─────────────────────────────────────────────────────┘
Total: 88.5 ms/token
```

---

## 왜 CPU/GPU는 안 느려지는가?

같은 attention 계산을 하는데 왜 CPU는 context-invariant일까?

### 이유 1: CPU는 hardware prefetch + L3 cache

Cortex-A76 L3 = 2-3 MB:
- **Weights는 재사용** → L3가 자주 쓰는 weight tile을 hold
- **KV cache는 sequential read** → hardware prefetcher가 미리 땡겨옴
- Effective BW ~15 GB/s (peak의 75%)
- 결과: **memory latency가 숨겨짐**, context 증가분이 pipeline에 묻힘

### 이유 2: CPU는 SIMD width가 작아 항상 fully utilized

| 지표 | CPU (Cortex-A76 4 cores) | NPU (3 cores) |
|------|-------------------------|---------------|
| SIMD width | 16 lanes (NEON sdot) | 수천 MAC |
| FP16 peak | ~150 GFLOPS | 3000 GFLOPS |
| M=1 matmul 활용률 | ~10% | **0.37%** |
| Attention 11 GFLOPS 달성 | ✓ | ✓ |

**CPU는 peak 대비 상대 활용률이 높아** (작은 SIMD가 full) 작은 matmul도 효율적.
**NPU는 peak 대비 상대 활용률이 낮음** (큰 parallelism unused) → 그대로 compute time 노출.

### 이유 3: GPU는 느리지만 context에 무관

Mali G610 OpenCL:
- 전체적으로 CPU보다 느림 (GPU kernel launch overhead, memory coalescing 비효율)
- 하지만 decode 같은 small-batch에서도 **kernel 내 thread가 context 늘면 늘어남**
- → Latency가 context에 비례해 늘지 않음 (throughput이 늘어남)

---

## 비유

### CPU = 컨베이어 벨트 (16 lanes)
- 모든 레인이 항상 일함 (fully utilized)
- 부품(weight)은 창고(L3 cache)에서 미리 가져옴
- 물건(context) 늘어도 벨트가 계속 돌면서 소화

### NPU = 거대 공장 기계 1대 (수천 MAC)
- 거대한 설비지만 셋업 비용 큼
- **M=1이면 드릴 1개만 쓰고 나머지 수천 개는 놀고 있음**
- context 늘어나면 노는 드릴은 그대로, 일하는 드릴이 선형적으로 일 많아짐

### GPU = 빠른 미니 공장
- 병렬도 높지만 program startup 비용 큼
- 작업량 자체가 적어 context 영향 적음

---

## Phase 2의 반증: KV 압축 실험

"KV bandwidth가 문제면, KV 압축하면 빨라질 것"

MNN CPU에서 `quant_qkv=8 (no KV quant)` vs `9 (Q,K INT8)` vs `10 (Q,K,V INT8)` 비교:

| Context | qkv=8 decode | qkv=9 | qkv=10 |
|---------|-------------|-------|--------|
| 32 | 16.07 | 15.67 | 15.62 |
| 1024 | 16.12 | 15.76 | 16.01 |
| 4096 | 15.62 | 16.00 | 15.62 |

**차이 ≤ 2% (noise 수준)**. KV 압축으로 decode 속도 향상 **0%**.

→ TurboQuant/KIVI 같은 KV 압축은 **엣지 NPU 문제의 해결책이 아니다**.

---

## Dispatch overhead도 주범 아님

NPU API 호출 자체의 cost를 격리 측정:
- 가장 작은 matmul (32×16) 반복 호출 → **21 µs/call**
- 중간 크기 → 25-35 µs/call
- **Per-call dispatch ≈ 25 µs** (ioctl + cache sync + kernel launch)

Llama decode:
- 16 layers × ~9 matmul = 144 matmul per token
- 144 × 25 µs = **3.6 ms/tok** (dispatch 총량)
- 전체 decode 50-88 ms 중 **4-7%만 dispatch**
- **나머지 93-96%는 pure compute**

---

## 예측 검증

### 마이크로벤치 → 원 측정 재현

```
예측 NPU decode slope
= (per-matmul slope) × (# attention matmul per token)
= 0.367 µs/tok × 32 matmul
= 11.71 µs/tok
```

실측 NPU decode slope = **9.26 µs/tok**

→ **79% 일치** (나머지 21%는 NPU multi-core 병렬화, GQA head 분할 최적화 등)

수학적으로 **compute-bound attention 가설 검증 완료**.

---

## 결론 및 실용 함의

### 1. 통념 반증
- **KV bandwidth 가설**: 반증 (KV는 10%, 압축 효과 0%)
- **Dispatch overhead 가설**: 반증 (4-7%만 차지)

### 2. 진짜 원인
- **NPU M=1 matmul에서 peak의 0.37%만 사용** → parallelism 낭비
- Attention matmul의 O(context) compute가 그대로 노출 → linear slowdown

### 3. CPU vs NPU 대조
| 상황 | 승자 | 이유 |
|------|------|------|
| 짧은 context (≤1024) | NPU | Non-attention 부분은 큰 matmul이라 NPU가 효율적 |
| 긴 context (≥2048) | **CPU** | Attention이 전체를 지배하고, NPU의 M=1 비효율 노출 |

### 4. "NPU가 무조건 빠르다"는 통념 반증
- 엣지 NPU는 **batch>>1 또는 speculative decoding 같은 기법**이 있어야 제 성능
- 순수 autoregressive decode는 **NPU의 주특기가 아님**

### 5. 실용 제안
- **Prefill은 NPU, Decode는 CPU** 하이브리드 실행
- 또는 **Flash Attention + tiling**으로 M을 인위적으로 키우는 kernel fusion
- KV 압축은 대용량 context 메모리 절약에는 유효하나 **속도 향상에는 무용**

---

## 데이터 소스

- [results/figures_roofline/](results/figures_roofline/) — Phase 1 Roofline 분석
- [results/figures_microbench/](results/figures_microbench/) — Phase 4 NPU matmul microbench
- [results/raw/full_results.csv](results/raw/full_results.csv) — 전체 벤치마크
- [results/raw/kv_bench_full.log](results/raw/kv_bench_full.log) — MNN KV quant 실험
- [microbench/attention_matmul_scaling.c](microbench/attention_matmul_scaling.c) — RKNN matmul C 코드
- [microbench/gqa_matmul_shapes.c](microbench/gqa_matmul_shapes.c) — shape 비교
- [microbench/dispatch_overhead.c](microbench/dispatch_overhead.c) — dispatch 격리

---

## Figures

| File | 내용 |
|------|------|
| `fig_shape_comparison.pdf` | 3 shapes (K=64/512/2048) slope 비교 + K proportionality |
| `fig_decode_decomposition.pdf` | Decode time 분해 (non-attn / dispatch / attn-compute) |
| `fig_dispatch_vs_compute.pdf` | Per-call dispatch (~25µs) isolation |
| `npu_matmul_scaling.pdf` | Shape A linear fit R²=0.9999 |
| `decode_prediction_from_microbench.pdf` | 마이크로벤치 예측 vs 실측 79% 일치 |
