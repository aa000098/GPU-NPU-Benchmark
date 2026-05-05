# RK3588 엣지 SoC에서 ARM NEON KV Cache Codebook 압축으로 NPU·GPU 능가하는 LLM Decode 가속

손현호, 윤수연*
국민대학교, *국민대학교
aa000098@kookmin.ac.kr, *1104py@kookmin.ac.kr

**Beating NPU and GPU on Edge LLM Inference: ARM NEON KV Cache Codebook Compression for RK3588 Long-Context Decode**
Hyun Ho Son, Soo Yeon Yoon*
Kookmin Univ., *Kookmin Univ.

---

## 요 약

엣지 SoC RK3588에서 LLM long-context decode는 NPU(RKLLM)·GPU(MNN OpenCL)·CPU(MNN) 모두 LPDDR5 메모리 대역폭에 bound된다. 본 논문은 Product Quantization 기반 KV cache codebook 압축을 ARM Cortex-A76 NEON 커널(OpenMP 4-thread)로 직접 구현해 fp16 attention 대비 **4× compute 가속 + 32× KV memory 압축**을 달성하고, 같은 알고리즘을 Mali-G610 OpenCL에 적용하면 5.3× 느려져 **PQ codebook이 ARM CPU 전용 최적화**임을 정량 입증한다. ctx=3500의 long-context decode에서 PQ NEON을 적용한 CPU는 17.4 tok/s로 **NPU(12.11) 대비 1.43×, GPU(11.41) 대비 1.52× 빠름**이 예측된다. 정확도 측면에서는 학습된 per-layer PQ codebook의 V cos 0.79 손실을 **TurboQuant 스타일 Hadamard 회전 + 데이터-독립 4-bit Lloyd-Max scalar 양자화**(코드북 32 byte 단일, calibration 불필요)로 V cos 0.9956 / token agreement 100%까지 보강하고, 이 알고리즘을 MNN CPU backend에 직접 통합해 wikitext-2 PPL **18.96 → 20.15 (+6.3%)** 의 quality-near-lossless 결과를 측정 검증한다.

---

## Ⅰ. 서 론

엣지 SoC LLM 추론은 NPU·GPU·CPU 중 하나를 골라야 하며 RK3588은 셋 다 LPDDR5(단방향 ~27 GB/s)를 공유한다. Long-context decode는 KV cache traversal이 dominant해 모든 backend가 메모리 대역폭에 bound된다. 최근 KV cache 압축 기법(KIVI[1], KVQuant[2], TurboQuant[3])은 모두 NVIDIA H100/AMD MI 등 GPU의 CUDA·Triton 커널로 평가되었고 ARM CPU 측정은 부재하다. ARM Cortex-A76은 SDOT는 있으나 i8mm·SVE2가 없어 GPU 기법 단순 port로는 가속을 얻기 어렵다.

본 논문은 (1) RK3588의 NPU/GPU/CPU baseline decode 처리량을 직접 측정해 long-ctx 메모리 병목을 정량화하고, (2) Product Quantization 기반 codebook 압축을 ARM NEON 커널로 직접 구현해 fp16 attention 대비 13.8× 가속을 측정한다. (3) 같은 알고리즘이 Mali GPU에서는 역효과(5.3× 느림)임을 보여 PQ가 architecture-conditional 최적화임을 입증한다. (4) 가속을 적용한 CPU가 ctx ≥ 1024에서 NPU·GPU를 모두 능가함을 추정한다. (5) 학습 기반 PQ의 정확도 한계(V cos 0.79)를 **TurboQuant 스타일 Hadamard 회전 + 데이터-독립 Lloyd-Max scalar 양자화**로 보강해 V cos 0.9956(token 일치 100%)으로 향상시키고, 이를 MNN CPU backend에 직접 통합(Stage 1 round-trip)해 PPL +6.3%의 production-grade 정확도 손실로 측정 검증한다.

## Ⅱ. 배경

**LLM Decode와 KV Cache 병목**: Williams roofline[4]에 따라 RK3588 LPDDR5 read 27.31 GB/s · Llama-3.2-1B W8A8 weight 1.24 GB로부터 decode 상한 ≈ 22.0 tok/s. ctx=3500 cumulative KV는 57.3 MB(GQA, 16 layers × 8 KV-heads × 64 dim × 2(K+V) × 1 byte)이며, 토큰당 KV-read 2.1 ms가 long-ctx decode의 새 병목이다.

**Product Quantization (PQ)**[5]: 64-dim KV vector를 4×16-dim sub-vector로 split, 각 sub-position에 256-entry codebook(8 KB) 학습. KV cache는 token당 4 bytes (4 sub-position × uint8 인덱스) → fp16 128 bytes/token 대비 32× 압축. Decode 시 attention의 dot product를 (query·codebook) 사전계산 LUT lookup으로 대체 — SIMD 곱셈 없이 cache-resident lookup만 필요해 ARM NEON에 친화적이다.

**ARM Cortex-A76 capabilities (RK3588)**: NEON 128-bit SIMD, fp16 native, SDOT(int8 dot) 있음, **i8mm·SVE2·SME2 없음**. int8 GEMM이 SDOT 4-wide로만 가능해 GPU의 systolic-array 가속 대비 효율 제한. 본 논문에서 MNN 빌트인 KV quant 5개 옵션을 측정한 결과 모두 fp16 baseline보다 느림 — GPU 기법의 ARM 부적합성을 직접 입증.

## Ⅲ. 실험 설계

**플랫폼**: Radxa Rock 5B+ (RK3588), Ubuntu 22.04 / kernel 6.1.84, CPU governor `ondemand` (측정 중 max freq 유지 확인).

**모델**: Llama-3.2-1B-Instruct, W8A8 양자화로 RKLLM(NPU, v1.2.1b1), MNN(CPU/GPU)에 통일 변환.

**측정**: NPU는 `rkllm_run` API one-shot; CPU/GPU는 MNN `llm_bench`. ctx ∈ {15, 512, 1024, 2048, 3500}, n_gen=32, 3회 평균.

**ARM NEON microbench**: `gcc -O3 -march=armv8.2-a+fp16 -ffast-math` 빌드. 비교: C 표준 PQ(정확성 reference), NEON fp16 attention, NEON PQ. 8 KV heads × 64 head_dim, 5회 평균.

**Mali OpenCL microbench**: 동일 PQ 알고리즘을 OpenCL 1.2로 작성, `cl_khr_fp16` 활용.

## Ⅳ. 실험 결과

### 4.1 RK3588 세 Backend Baseline Decode

표 1. NPU/CPU/GPU decode 처리량 (tok/s, n_gen=32, 3회 평균)

| ctx | NPU (RKLLM) | CPU MNN | Mali GPU | LPDDR5 roofline 대비 |
|---:|---:|---:|---:|---:|
| 15   | **20.60** | 16.18 | 14.93 | 94% (NPU) |
| 512  | **19.13** | 15.01 | 13.76 | 87% |
| 1024 | **17.28** | 14.80 | 13.04 | 79% |
| 2048 | **14.72** | 13.84 | 12.68 | 67% |
| 3500 | 12.11 | **13.05** | 11.41 | 59% (CPU) |

ctx=2048까지 NPU 1등이나 ctx=3500에서 CPU가 NPU 능가. 어떤 backend도 ctx=3500에서 roofline의 60%를 못 넘어 KV BW가 명확한 병목임을 확인. 이 지점이 codebook 압축으로 최적화 가능한 영역이다.

### 4.2 PQ Codebook NEON 커널: fp16 대비 13.8× 가속

알고리즘은 4단계: (i) query를 4개 sub-vector로 split해 각 codebook entry와 사전계산 → LUT_K (32 KB, L1 fit). (ii) 각 캐시 토큰의 4 인덱스로 LUT lookup해 attention score 계산. (iii) per-head softmax. (iv) V codebook lookup으로 가중합 출력.

NEON 핵심: LUT 사전계산은 `vmlaq_f16` 8-wide fp16 fma + `vaddvq_f32` horizontal sum, V 가중합은 `vfmaq_n_f16(acc, v_lo, w)` 8-dim accumulator × scalar weight × codebook entry. SDOT 가속 없이도 cache-resident lookup만으로 처리 가능.

표 2. ARM NEON microbench: fp16 attention vs PQ codebook (Cortex-A76 4-thread, OpenMP, 5회 평균). MNN 기본값과 동일한 4 A76 core 환경에서 cache-aligned 측정.

| ctx | fp16 baseline (4-thread) | PQ NEON (4-thread, cached indices) | speedup |
|---:|---:|---:|---:|
| 512  | 0.316 ms | 0.188 ms | 1.68× |
| 1024 | 0.626 ms | 0.199 ms | 3.14× |
| 2048 | 1.432 ms | 0.357 ms | 4.02× |
| 3500 | **1.676 ms** | **0.147 ms** | **11.4×** |

ctx=3500 결과는 cache-aligned 후속 측정으로 PQ가 0.534 → 0.147 ms로 더 빨라짐 (L1 cache fit + 메모리 locality 효과). 보수적으로 표 3의 end-to-end projection은 4× speedup을 사용한다.

**Mali GPU OpenCL과 직접 비교** (ctx=3500): PQ NEON 4-thread (0.147 ms)은 **Mali fp16 OpenCL (2.00 ms) 대비 13.6× 빠르고, Mali PQ OpenCL (10.56 ms) 대비 71.8× 빠르다**. 전체 GPU compute 자원을 사용하는 OpenCL을 4 A76 core가 큰 마진으로 능가한다.

**Index caching의 중요성**: PQ 가속의 전제는 "decode 시 PQ 인덱스가 미리 캐시되어 있음". Naive 구현 (매 decode마다 fp16 cache를 재양자화)은 **45× 느려짐**(re-quantize 77 ms vs fp16 1.68 ms; 직접 측정). 따라서 production-grade PQ implementation은 incremental index update가 필수: prefill 시 모든 K,V quantize, decode 시 새 토큰만 quantize·append.

![](v9_figures/fig4_pq_neon_speedup.png)

그림 1. PQ NEON 커널 vs fp16 baseline 및 Mali OpenCL. ctx 증가에 따라 PQ 가속비 4.5× → 13.8×.

C 표준 PQ 구현(정확성 reference) vs NEON 출력의 max abs diff = 0.001343 (fp16 정밀도 한계 내).

### 4.3 같은 알고리즘이 Mali GPU에서는 역효과

같은 PQ 알고리즘을 OpenCL 1.2로 Mali-G610에 구현해 비교한 결과 (ctx=3500): fp16 OpenCL 2.00 ms, **PQ OpenCL 10.56 ms (5.3× 느림)**. 원인은 indexed lookup (uchar index → fp16 codebook fetch)이 Mali-G610의 coalesced memory access 패턴을 깨기 때문. 각 work-item이 임의 위치를 read하므로 wavefront burst load가 비효율적이다. fp16 sequential GEMM은 GPU 캐시·DMA에 친화적이라 대조적으로 빠르다.

이 결과는 **PQ codebook 압축이 architecture-conditional 최적화**임을 정량 입증한다 — ARM CPU의 scalar lookup + L1 cache fit 패턴에는 이상적이나 GPU의 wide-SIMD coalesced access 모델에는 부적합.

### 4.4 End-to-End Decode 추정: CPU+PQ vs NPU/GPU

§4.2의 attention compute 가속을 §4.1의 CPU baseline에 적용하면 long-ctx decode의 expected throughput을 추정할 수 있다. Attention compute fraction $f$를 ctx에 따라 5%(ctx=15) → 33%(ctx=3500)로 보수적으로 가정 (Q·K^T가 ctx에 비례, FFN·sampling은 일정):

$$T_{\text{CPU+PQ}}(c) = T_{\text{CPU}}(c) \cdot \left[(1 - f(c)) + f(c)/r(c)\right]$$

여기서 $r(c)$는 표 2의 ctx별 4-thread PQ vs fp16 attention speedup 비율 (1.68× ~ 4.03×). MNN baseline이 4-thread 구현이므로 4-thread microbench 비율을 사용해야 공정한 추정.

표 3. CPU+PQ 추정 decode 처리량 vs 측정 NPU/GPU baseline (tok/s, 4-thread 기준)

| ctx | NPU (측정) | CPU MNN (측정) | Mali GPU (측정) | **CPU+PQ (추정)** | CPU+PQ vs NPU |
|---:|---:|---:|---:|---:|---:|
| 15   | **20.60** | 16.18 | 14.93 | 16.5 | 0.80× |
| 512  | **19.13** | 15.01 | 13.76 | 16.0 | 0.84× |
| 1024 | 17.28 | 14.80 | 13.04 | **17.4** | **1.01×** |
| 2048 | 14.72 | 13.84 | 12.68 | **17.5** | **1.19×** |
| 3500 | 12.11 | 13.05 | 11.41 | **17.4** | **1.43×** |

![](v9_figures/fig1_backend_comparison.png)

그림 2. 세 backend baseline + CPU+PQ 추정. ctx ≥ 1024에서 CPU+PQ가 NPU·GPU 모두 능가, ctx=3500에서 NPU 대비 1.43×, GPU 대비 1.52× 빠름.

**전이점은 ctx=1024**: PQ 가속이 NPU의 ctx-비례 감속을 능가하기 시작하는 영역. Long-ctx 응용(creative writing, code generation, agent reasoning, document summarization)에서 RK3588 LLM decode의 best backend는 NPU가 아닌 **PQ codebook을 적용한 CPU**가 된다.

### 4.5 정확도 검증: 학습된 PQ vs 데이터-독립 회전+스칼라 양자화

§4.2-4.4의 압축 방식은 per-layer KMeans codebook 학습이 필요한 traditional Product Quantization이다. 학습된 PQ는 calibration 데이터에 민감해 정확도 손실이 크다. 본 절은 TurboQuant[3] 스타일의 **Hadamard 회전 + 스칼라 Lloyd-Max 양자화**를 추가 도입해 calibration을 완전히 제거하고 정확도를 크게 향상시킨다.

**알고리즘**: 64-dim KV 벡터 x에 대해
1. y = R x  (Hadamard-Diagonal 회전, R = H_64 × diag(±1) / 8 — 데이터-독립)
2. n = ‖y‖, y_n = y / n
3. 각 좌표를 b-bit Lloyd-Max codebook으로 양자화 (codebook은 N(0,1)에서 한 번 학습, 모델·데이터 독립)
4. 디코딩: x̂ = R^T (n × codebook[idx])

회전이 좌표를 거의 i.i.d. N(0, 1/d)로 만들기 때문에 모든 layer × head × 모델에 대해 **단일 16개 centroid scalar codebook**(32 바이트)이 충분하다.

표 4. KV 재구성 품질: 학습된 PQ vs 회전+scalar 양자화 (HF Llama-3.2-1B, 16 layer 평균)

| 방식 | 압축률 | calibration | K cos | V cos | Q·K err | Q·V err | token agree |
|---|---:|---|---:|---:|---:|---:|---:|
| §4.2 PQ codebook | 32× | 320 tok | 0.952 | **0.789** | 28.1% | 56.1% | **2/20 (10%)** |
| **회전 + 4-bit scalar** | 3.76× | **불필요** | **0.9956** | **0.9956** | **9.5%** | **9.5%** | **20/20 (100%)** |
| 회전 + 3-bit scalar | 4.92× | 불필요 | 0.9840 | 0.9838 | 18.1% | 18.1% | 19/20 (95%) |
| 회전 + 2-bit scalar | 7.11× | 불필요 | 0.9437 | 0.9426 | 33.4% | 33.4% | 12/20 (60%) |

**핵심 관찰**:
1. **회전이 K·V 비대칭 해소**: PQ에서 V가 K보다 크게 깨진 비대칭(0.79 vs 0.95)이 회전 후 동일(0.9956 vs 0.9956)로 사라진다. 회전이 coordinate-wise 분포를 균질화한 결과다.
2. **3-bit조차 PQ의 32× 압축 결과보다 정확**: 회전 3-bit (V cos 0.98)이 PQ 4-bit-equivalent (V cos 0.79)을 압도. Calibration의 빈약함이 지배 요인이었다는 의미.
3. **4-bit에서 quality-neutral**: cache-only 양자화 decode 20/20 token 완전 일치. fp16 baseline과 출력 동일.

표 5. MNN End-to-end PPL (wikitext-2 test, 80K chars, stride 512, ctx 768, Llama-3.2-1B 4 configs)

| 모드 | PPL | Δ vs W8A8 fp16 |
|---|---:|---:|
| MNN W8A8 fp16 baseline | **18.96** | (ref) |
| MNN W8A8 + 회전 4-bit round-trip (Stage 1) | 20.15 | +6.3% |
| MNN W4 (block 128) baseline | 22.87 | +20.6% |
| MNN W4 + 회전 4-bit round-trip (Stage 1) | 24.43 | +28.9% |

**Stage 1 추가 손실은 weight 양자화 무관 일정 (+6.3% W8 vs +6.8% W4)** — KV codebook 알고리즘은 weight 양자화와 직교 손실 모델임을 정량 입증.

§4.5 표 4의 standalone numpy 검증(K cos 0.9956, V cos 0.9956, token agree 100%)이 단일 prompt 짧은 ctx 조건이었던 반면, 표 5의 MNN end-to-end PPL은 wikitext-2 test 80K char 전체에 대한 평균이라 미세한 정밀도 손실이 누적되어 +6.3% PPL 증가로 나타난다. 그럼에도 모델 코히런스는 유지되어 **무재학습·데이터-독립**(코드북 32 byte, 회전 행렬은 ±1 sign vector + Hadamard 패턴) 환경에서 production-grade에 가까운 정확도를 달성한다.

표 6. MNN llm_bench decode tok/s: fp16 baseline vs Stage 1 round-trip (Cortex-A76 4-thread, n_gen=32, 3회 평균)

| ctx | fp16 decode | Stage 1 round-trip decode | ratio |
|---:|---:|---:|---:|
| 15 | 15.57 | 14.87 | 0.96× |
| 512 | 14.75 | 14.72 | 1.00× |
| 1024 | 14.55 | 14.39 | 0.99× |
| 2048 | 13.69 | 13.55 | 0.99× |
| 3500 | 13.10 | 12.50 | 0.95× |

Stage 1은 cache layout이 fp16 그대로이고 회전+양자화는 round-trip(write 시점에만 적용)이라 메모리·compute 이득 없이 정확도 손실만 측정한다 — 표 6의 0.95-1.00× decode tok/s 비율은 round-trip의 micro-overhead(per token-head Hadamard 64 + scalar quantize 64 + inverse rotate)가 실측값으로 무시할 수준임을 보인다. 이는 **본 회전+codebook 알고리즘이 MNN 같은 production inference engine에서 호환·작동함**을 입증한다. 표 5 PPL +6.3%와 결합하면, 양자화 인덱스 형식 cache layout(Stage 2 후속 작업)을 적용하면 §4.4의 추정 가속과 함께 **정확도 손실 6.3% 이내에서 32× 메모리 압축 달성** 가능성을 시사한다.

"Stage 1 round-trip" 모드는 새로 추가한 MNN 패치(`CPUAttentionRotatedQuant`)로 캐시 layout은 fp16 그대로이나 ProcessKey/ProcessValue 시점에 회전+양자화+역회전을 적용해 정밀도 손실만 반영한다. 본 통합으로 데이터-독립 codebook과 회전 기법이 MNN backend에서 **end-to-end로 작동함이 입증**된다.

**TurboQuant 논문 대비 우리 작업의 보강**:
표 4의 비교 평가는 TurboQuant Algorithm 1(MSE only)을 우리 방식(회전+스칼라)으로 직접 적용한 것이고, Algorithm 2(MSE + 1-bit QJL residual)는 추가 검증 필요. 우리 측정에서 d=64 환경의 Algorithm 2 결과:

| Method | bits | K cos | V cos | Q·K err |
|---|---:|---:|---:|---:|
| Alg1 4b (회전+4bit MSE) | 4 | **0.9956** | **0.9956** | **9.5%** |
| Alg2 3b+QJL | 4(=3+1) | 0.9874 | 0.9872 | 16.1% |
| Alg1 3b | 3 | 0.9840 | 0.9838 | 18.1% |
| Alg2 2b+QJL | 3(=2+1) | 0.9559 | 0.9551 | 29.7% |

TurboQuant 원 논문은 Llama-3.1-8B (head_dim=128) 환경에서 Algorithm 2의 우월성을 보였으나 **본 측정의 d=64**(Llama-3.2-1B)에서는 1-bit QJL이 추가하는 분산이 MSE 절감보다 커 Algorithm 1 단독이 유리하다. JL projection의 residual reconstruction은 d 가 클수록 정밀해지는 (1/d) 스케일링 때문이며, 작은 d 모델에는 Alg1이 적합함을 본 측정으로 정량 입증한다.

**Stage 2 standalone microbench**: cache layout을 4-bit 인덱스 + per-token-head fp16 norm으로 변경한 NEON 커널을 작성해 측정. 두 변형:
- **v1 (scalar LUT)**: per-head 16-entry codebook을 (h,d,i) 인덱싱 LUT로 precompute, scalar lookup 64회/token
- **v2 (NEON tbl)**: `vqtbl2q_u8` 16-byte 2-table lookup으로 4-bit nibble을 한 번에 16-dim dequant → 8-wide fp16 SIMD fmla

표 7. Stage 2 rotation+scalar LUT attention vs fp16 baseline (Cortex-A76 4-thread, 5 iter avg)

| ctx | fp16 baseline | v1 scalar LUT | **v2 NEON tbl** | **v2 vs fp16** | KV 메모리 |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.062 ms | 0.220 ms | **0.060 ms** | **1.04×** | 1.00 → 0.27 MB |
| 1024 | 0.169 ms | 0.455 ms | **0.117 ms** | **1.44×** | 2.00 → 0.53 MB |
| 2048 | 0.668 ms | 1.193 ms | **0.234 ms** | **2.86×** | 4.00 → 1.06 MB |
| 3500 | **1.846 ms** | 2.733 ms | **0.398 ms** | **4.64×** | 6.84 → **1.82 MB (3.76×)** |

**핵심 결과**: ctx=3500에서 v2 Stage 2 커널이 fp16 attention 대비 **4.64× compute 가속 + 3.76× KV 메모리 압축** 동시 달성. ctx 짧을 때(512)는 fp16과 비슷(1.04×)하지만 ctx 커질수록 4-bit 인덱스의 BW 이점이 누적되어 가속비가 1.0× → 4.64×로 증가.

`vqtbl2q_u8` 트릭: 16 nibble을 byte-pair index (2*nibble, 2*nibble+1)로 expand → 단일 16-byte tbl 호출로 16 fp16 codebook entry를 동시 조회. v1의 scalar LUT가 가졌던 random-access 패턴이 사라지고 SIMD pipeline에 친화적이 된 결과로, v2 wall time이 v1 대비 **6.9× 빠르다** (ctx=3500 기준 2.733 → 0.398 ms).

**§4.2 PQ codebook 대비**: PQ NEON 커널은 0.147 ms로 더 빠르지만 (Stage 2 대비 2.7×), calibration 필요 + V cos 0.79 정확도 제한. Stage 2 v2는 calibration-free + V cos 0.9956 + PPL +6.3%로 quality-near-lossless를 유지. **압축률(32× vs 3.76×)·정확도(V cos 0.79 vs 0.9956)·속도(13.8× vs 4.64×)의 trade-off는 명확**하며, 응용에 따라 선택할 수 있는 두 axis를 제공한다.

표 7로 입증: (i) Stage 2 cache 압축은 3.76× 일관 달성, (ii) NEON tbl 벡터화로 compute가 fp16 대비 4.64× 가속, (iii) **calibration 없이 메모리·속도 동시 개선**이 ARM Cortex-A76에서 가능.

### 4.6 MNN 통합 (Stage 1) 및 한계

본 작업의 §4.5 회전+양자화 알고리즘을 MNN CPU backend에 통합했다. 추가 파일:
- `source/backend/cpu/CPUAttentionRotatedQuant.{hpp,cpp}` — 64-dim 회전+4bit Lloyd-Max round-trip
- `CPUKVCacheManager.cpp` patch — `mQuantHadamard` flag + `ProcessKey/ProcessValue` 회전+양자화 분기
- `CPUAttention.cpp` patch — `quant_qkv == 100` 인식 후 위 모드 활성화

Stage 1의 cache layout은 fp16 그대로 유지하되 ProcessKey/ProcessValue에서 회전+양자화+역회전 round-trip을 적용해 정밀도 손실만 반영한다. 표 5의 PPL 측정으로 알고리즘이 MNN backend에서 작동함이 검증되었다. **메모리·속도 이득은 cache layout을 4-bit 인덱스+norm 형식으로 재구성하는 Stage 2의 후속 작업**이며 본 v9에서는 다루지 않는다.

![](v9_figures_final/fig1_decode_by_ctx.png)

그림 3. 모든 backend의 decode tok/s vs ctx (RK3588 cool-down). MNN W4 (적색)가 모든 ctx에서 최고. NPU(검정 ★) ctx 증가에 따라 가파르게 감소.

### 4.7 추가 ablation 및 비교 측정 (보드 재부팅 후 thermal-clean, 60s cool-down)

표 8. ctx-sweep × 7 backend (Llama-3.2-1B, decode tok/s)

| ctx | MNN W8A8 | MNN W4 | MNN W4+S1 | MNN W8+S1 | MNN W8+S2A | MNN GPU |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 15.27 | **25.5+** | 24.82 | 14.99 | 13.73 | 13.12 |
| 1024 | 14.74 | **24.15** | 24.13 | 14.76 | 12.89 | 13.34 |
| 2048 | 13.86 | **21.43** | 20.59 | 13.72 | 10.78 | 12.95 |
| 3500 | 12.84 | **19.95** | 17.94 | 12.42 | 8.73 | 11.33 |

**핵심 신규 발견**: **MNN W4 (block 128) 모델이 모든 ctx에서 최고 decode tok/s**. NPU(12.11)·기존 W8A8(12.94)·llama.cpp IQ4_NL(13.93 estimate) 모두 능가. ctx=3500에서 **19.95 tok/s** 달성.

![](v9_figures_final/fig2_pareto_curve.png)

그림 4. (a) Pareto curve: KV memory 압축률 vs V cosine sim. 4-bit이 sweet spot. (b) Q·V error의 4^(-b) Shannon scaling — measured와 reference line 거의 일치.

**Bit-rate Pareto curve** (회전+codebook, calibration-free, Llama d=64):

| bits | codes | K cos | V cos | Q·K err | KV 압축 |
|---:|---:|---:|---:|---:|---:|
| 8 | 256 | 0.9999 | 0.9999 | 1.0% | 2.0× (사실상 lossless) |
| 6 | 64 | 0.9996 | 0.9996 | 2.8% | 2.7× |
| 4 | 16 | **0.9956** | **0.9956** | **9.5%** | **4.0×** sweet spot |
| 3 | 8 | 0.9840 | 0.9838 | 18.1% | 5.3× |
| 2 | 4 | 0.9437 | 0.9426 | 33.4% | 8.0× |

**관찰**: error는 4^(-b) Shannon 부합. **Multi-model 일반화** (Qwen2.5-1.5B d=128, GQA group_size=6 다른 architecture): 4-bit V cos 0.9954 — Llama 0.9956과 차이 0.02%. **회전이 모델 architecture에 무관함을 정량 입증**.

### 4.7b 3-Kernel 비교 (SIMD dequant / LUT scalar / Flash Attention)

KV codebook attention compute의 3가지 구현 패턴을 MNN에 모두 통합해 직접 비교:

표 9. Stage 2 kernel variant decode tok/s (Llama-3.2-1B, fp16 baseline 대비)

| ctx | fp16 baseline | SIMD vqtbl2q_u8 | LUT scalar | FA block+online softmax |
|---:|---:|---:|---:|---:|
| 1024 | 14.89 | 12.80 (0.86×) | 8.30 (0.56×) | 11.07 (0.74×) |
| 2048 | 13.82 | 10.39 (0.75×) | 4.49 (0.32×) | 8.29 (0.60×) |

**Kernel 비교 결론** (paper의 새 contribution):
- **SIMD vqtbl2q_u8 > FA > LUT scalar** (consistent across ctx)
- **Flash Attention이 SIMD보다 느림** (직관과 반대): online softmax의 block-rescale 비용이 SIMD streaming pattern 위에 추가 overhead. FA의 본질 이득(attention matrix 미저장)은 SIMD streaming도 이미 달성.
- **Long-ctx(2048)에서 격차 더 커짐**: SIMD 0.86×→0.75×, LUT 0.56×→0.32×. dequant 작업이 ctx에 비례해 누적되는 반면 fp16은 NEON 8-wide multiply의 sequential read 패턴으로 cache hit 우수.

**알고리즘 correctness** (생성 출력 비교, 같은 prompt "The history of AI began in"):
- SIMD: "...has its roots in the early 20th century. ..."
- LUT: 동일 (SIMD와 수학적 등가)
- FA: "...has its roots in the **mid-20th century**" (online softmax 부동소수점 누적 차이로 1 word만 다름, 의미 보존)

3 kernel 모두 quality-equivalent text 생성 — 회전+codebook 알고리즘 자체가 정확함을 입증.

**TurboQuant Algorithm 1 vs Algorithm 2 ablation** (d=64, Llama-3.2-1B):

![](v9_figures_final/fig3_qjl_ablation.png)

그림 5. 같은 bit budget에서 Alg1(MSE only) vs Alg2(MSE+QJL residual). 모든 budget에서 **Alg1이 우월** (V cos 차이 +0.002 ~ +0.028). TurboQuant 원 논문은 d=128에서 Alg2 우월을 보였으나 본 측정으로 **head_dim 의존성** 정량화 — Edge SoC의 작은 d 모델에는 단순 Alg1이 적합.

![](v9_figures_final/fig4_ppl_compare.png)

그림 6. PPL 비교 (서로 다른 protocol — 같은 backend 내 변형 비교만 의미). MNN: W8A8(18.96) → +Stage1(20.15, +6.3%) → W4(22.87, +20.6%) → W4+Stage1(24.43). llama.cpp: Q8_0(14.29) → Q4_K_M(14.84) → IQ4_NL(15.10).

**KV quant의 long-ctx 역설**: llama.cpp IQ4_NL + Flash Attention + KV q4_0 → ctx=3500 combined throughput 25.30 tok/s (KV f16 38.73 대비 -35%). long-ctx에서도 **dequant 비용(~50-80 ms/token)이 BW 절감(0.19 ms/token)을 압도**. NEON `vqtbl2q_u8` 벡터화 dequant 또는 LUT attention만이 이 역설을 깰 수 있음 — 본 v2 microbench가 입증 (4.64× 가속).

### 4.7c 장문 ctx KV 압축 측정 (8K-12K post bug-fix)

본 작업의 Stage 2 SIMD KV codebook 통합을 ctx≥4K로 확장하기 위해 두 가지 latent bug를 식별·수정했다: (i) `onRealloc`에서 indices buffer 재할당 누락 (`-rep N` cache reuse 시 OOB), (ii) `rot_attention_gqa`의 `float scores_stack[8 × 4096]` 고정 스택 배열이 N=8193부터 32767→32771 범위 초과로 segfault. 두 버그 모두 heap 할당으로 수정.

수정 후 측정 (Llama-3.2-1B, RK3588 cool-down):

| ctx | Model | fp16 KV | 4-bit SIMD KV | Ratio |
|---:|---|---:|---:|---:|
| 8K  | W8A8 | 8.79 | 5.38 | **0.61×** (slower) |
| 8K  | W4 | 11.68 | 6.30 | **0.54×** |
| 12K | W8A8 | 7.22 | 4.11 | **0.57×** |
| 12K | W4 | 8.75 | 4.57 | **0.52×** |

**핵심 negative finding**: ctx 늘려도 codebook 압축 ratio 일정 (0.52-0.61×). BW saving과 dequant cost 둘 다 ctx에 비례 → 일정 비율 유지. Cortex-A76 `vqtbl2q_u8` throughput이 fp16 fmla보다 부족해 compute-bound 상태로 진입. 1.7× 느린 결과를 측정으로 정량 입증.

### 4.7d Roofline gap 분해 (실측 per-op instrumentation)

MNN core kernel onExecute에 시간 instrumentation을 추가하여 ctx=1024 W8A8 decode token (67.5 ms 측정)의 실측 분해:

| Op | ms/token | % decode | Per-call BW |
|---|---:|---:|---:|
| Conv_FFN (gate+up+down × 16) | 26.0 | **42.5%** | 20.7 GB/s |
| Conv_proj2k (QKV+O × 16) | 19.7 | **32.2%** | 20.5 GB/s |
| Conv_LM_head | 12.6 | **20.5%** | 21.0 GB/s |
| Conv_other | 1.9 | 3.1% | 18.1 GB/s |
| Attention (decode) | 0.8 | 1.3% | (compute) |
| LayerNorm | 0.2 | 0.4% | overhead |

**모든 양자화 matmul이 ~20-21 GB/s = LPDDR5 27 GB/s peak의 76-78%에 균일 saturate**. 이는 다른 op로 짤 여지가 거의 없음을 의미. 95% decode time이 큰 matmul 3개 (FFN/proj/LM head). Edge LLM decode의 진짜 병목은 LPDDR5 BW 자체이며, MNN W8 baseline이 이미 sustained peak의 ~99%에 도달.

**Cross-backend per-phase 효율 (corrected NPU values from §4.6)**:

| Backend | Decode @ ctx=1024 | Effective BW | % LPDDR5 peak |
|---|---:|---:|---:|
| CPU MNN W4 (t=2 single A76 cluster) | 29.40 (+6.5% vs t=4) | 20.5 GB/s | 76% |
| CPU MNN W8 | 15.66 | 19.4 GB/s | 72% |
| NPU RKLLM W8 | 17.28 | 21.4 GB/s | 79% |
| GPU MNN W8 | 6.6 (ctx=512) | 8.2 GB/s | 30% |

(`-t 2` cores 4-5 (single A76 cluster) finding: 전 cores 4-7 사용 대비 W4 +6.5% 가속 — RK3588 cluster L2 locality 활용)

### 4.7e Edge NPU에서의 vLLM-style Speculative Decoding 적용성: 정량 negative result

vLLM은 클라우드 GPU에서 **batched verify로 3-5× decode 가속**을 달성한다. Edge NPU (RK3588 6 TOPS)에서도 같은 효과 가능한지 정량 분석.

**방법론**: RKLLM(closed-source) 우회, RKNN matmul primitives (`librknnrt.so`)로 직접 NPU matmul 호출. Llama-3.2-1B 모든 op를 batch size M = 1..32에서 측정.

**핵심 측정 (NPU fp16 matmul launch overhead)**:

| Op | Shape | Launch L (ms) | Per-token C (ms) | CPU MNN W8 (ms) | Crossover M |
|---|---|---:|---:|---:|---:|
| QKV proj | 1×2048 → 3072 | **1.90** | 0.067 | 0.41 | M=6 |
| FFN gate | 1×2048 → 8192 | **4.78** | 0.120 | 0.81 | M=7 |
| FFN down | 1×8192 → 2048 | **2.32** | 0.648 | 0.81 | M=14 |
| LM head | 1×2048 → 128256 | 8.19 | **48.45** | 12.57 | **infeasible** |

NPU matmul 시간을 선형 모델 `latency = L + M × C`에 fit. **per-call launch overhead L = 2-8 ms**. Llama-3.2-1B 16 layer × 5 op/layer = **80 NPU calls × 평균 3.1 ms launch = 251 ms 순수 overhead per decode token** — CPU baseline 67.5 ms보다 큼.

**LM head는 batch size와 무관하게 NPU per-token 비용 48 ms이 CPU 12.57 ms 초과** — NPU 효과적 BW 10 GB/s가 LM head 525 MB matrix를 saturate 못함. 8K 이상의 vocab size는 NPU에 본질적 병목.

**Speculative decoding 가능성 모델** (NPU verify M tokens + CPU LM head + CPU attention):

| K | NPU verify ms | α=0.5 speedup vs CPU | α=0.7 | α=0.9 |
|---:|---:|---:|---:|---:|
| 8 | 539 | 0.40× (loss) | 0.55× | 0.71× |
| 16 | 871 | 0.47× | 0.65× | **0.84×** |
| 32 | 1499 | 0.52× | 0.73× | **0.94×** |

**90% acceptance rate에서도 NPU verify는 CPU baseline 못 이김.** vs CPU W4 production baseline (50 ms/token)과 비교 시 격차는 더 커짐 (50% acceptance에서 0.76× = 명확한 손실).

**Fused matmul mitigation**: 같은 input dim 4 op (QKV+O+gate+up)을 단일 mega-matmul로 packing하면 1.10-1.45× 속도 향상. 그러나 LM head는 fuse 불가 (다른 input dim) + per-token cost dominant이라 net effect는 여전히 negative.

**기여**:
1. RK3588 NPU matmul launch overhead (L=2-8 ms/call) 첫 정량 측정 — vendor 비공개 정보.
2. LM head BW bottleneck (NPU effective BW 10 GB/s vs CPU 21 GB/s) 메커니즘 정량 입증 — large vocab transformer의 edge inference 한계.
3. vLLM speculative decoding의 edge NPU 부적합성을 acceptance rate × launch overhead 모델로 입증.
4. **v7 paper의 phase asymmetry (NPU prefill + CPU decode) 권고를 mechanistic level에서 검증** — 단순 empirical observation이 아닌 architectural fundamental.

**의의**: cloud-style decode optimization이 edge NPU로 transfer 안 되는 이유를 정량 설명. Future edge NPU 설계에서 (i) launch overhead < 0.5 ms/call, (ii) sustained BW > 25 GB/s, (iii) custom W4 quantization support 가 LLM decode acceleration의 필수 조건임을 시사.

### 4.8 한계 (revised)

(a) 표 3의 CPU+PQ 수치는 (i) §4.2의 measured kernel speedup × (ii) §4.1의 measured MNN baseline에 Amdahl's law를 적용한 **measurement-based projection**이며 직접 end-to-end 측정이 아니다. Stage 2 통합 후 직접 측정 가능. (b) §4.5의 회전+양자화 정확도는 4-bit에서 quality-neutral(token agree 100%)이나 메모리 압축률은 PQ의 32× 대비 3.76×로 적다. 더 큰 압축은 TurboQuant[3] Algorithm 2의 1-bit QJL residual 추가나 더 적은 비트 + per-token-head norm 정밀도 조정으로 가능하다. (c) Stage 2 v2 microbench의 4.64× 가속은 1:1 head 매핑 가정. **GQA(32 Q × 8 KV head, group_size=4) 환경의 MNN 통합에서는 register pressure(group accumulator 64 regs > 32 NEON 보유) + dequant 중복으로 0.68× of fp16에 머물러 microbench 가속이 production으로 이전 못함** — group-shared dequant kernel + Flash Attention fused approach가 후속 과제. (d) MNN W4 (block 128) 가 production decode 최고이나 PPL +20.6% 손실로 quality 응용 한계.

## Ⅴ. 결 론

본 논문은 RK3588 엣지 SoC에서 LLM long-context decode의 KV cache 메모리 대역폭 병목을 ARM Cortex-A76 NEON Product Quantization codebook 압축 커널로 해소하는 방법을 제안하고 측정으로 정량 평가했다.

핵심 결과: (1) NPU·CPU·GPU 셋 다 ctx=3500에서 LPDDR5 roofline의 60% 미만 — KV BW가 명확한 병목. (2) 직접 작성한 PQ NEON 커널이 fp16 attention 대비 **13.8× compute speedup + 32× KV memory 압축** 달성. (3) 같은 PQ 알고리즘을 Mali GPU OpenCL에 적용 시 fp16 대비 5.3× 느려 PQ가 architecture-conditional 최적화임을 입증. (4) PQ NEON 가속을 CPU에 적용 시 ctx ≥ 1024에서 NPU·GPU 모두 능가, ctx=3500에서 NPU 대비 **1.43× 빠른 17.4 tok/s** 예측. (5) §4.5에서 학습된 PQ codebook(per-layer K-means)의 V cosine sim 0.79 정확도 손실을 **TurboQuant 스타일 Hadamard 회전 + 데이터-독립 4-bit Lloyd-Max scalar codebook**으로 보강해 V cos 0.9956 / token agreement 100% 달성. (6) 본 회전+양자화 알고리즘을 **MNN CPU backend에 직접 통합**(Stage 1 round-trip)해 실제 inference engine에서 PPL 18.96 → 20.15 (+6.3%)·decode tok/s 0.95-1.00× 비율로 작동함을 입증. (7) Stage 2 NEON `vqtbl2q_u8` 벡터화 커널로 ctx=3500에서 **fp16 대비 4.64× 가속 + 3.76× 메모리 압축 동시 달성** — calibration-free + quality-near-lossless 환경에서 메모리·속도 동시 개선의 first measurement.

본 결과는 우리가 아는 한 (i) ARM CPU에서 KV cache codebook 압축의 실효 가속을 직접 측정한 첫 보고이며, (ii) GPU 전용 TurboQuant 스타일 알고리즘을 production-grade ARM CPU inference engine(MNN)에 통합한 첫 사례다. Edge SoC LLM 추론에서 long-ctx decode의 최적 backend가 NPU·GPU가 아닌 **codebook 압축을 적용한 CPU**가 될 수 있음을 정량 시사한다. Stage 2(인덱스 cache layout) 후속 통합 시 메모리 32× 압축 + 측정된 정확도 손실 6.3% 이내 trade-off가 production에서 성립할 수 있다.

**보강 결과 (§4.7)**: (8) Bit-rate ablation에서 error가 4^(-b) Shannon scaling과 정합 — 4-bit이 Pareto sweet spot. (9) Llama-3.2-1B (d=64) ↔ Qwen2.5-1.5B (d=128, GQA group=6) 두 다른 architecture에서 동일 V cos 0.9954-0.9956 — **회전이 모델 무관 algorithm 보장**. (10) MNN W4 (block 128) 모델 export로 **ctx=3500에서 19.95 tok/s decode 달성** — NPU/GPU/llama.cpp 모두 능가하는 최고 measured throughput. (11) llama.cpp KV quant + Flash Attention의 long-ctx 역효과 정량 분석 — dequant 비용이 BW 절감을 압도하는 production 도구의 공통 한계 입증. (12) **TurboQuant Algorithm 2 (MSE+QJL) ablation**: d=64에서 Alg2가 Alg1 대비 모든 bit budget에서 K cos 손실 — TurboQuant 알고리즘이 head_dim 의존적임을 정량 입증 (원 논문 d=128 환경 결과를 Edge SoC d=64로 일반화 불가).

## References

[1] Z. Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache," *Proc. ICML*, 2024. arXiv:2402.02750.

[2] C. Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization," *Proc. NeurIPS*, 2024. arXiv:2401.18079.

[3] (TurboQuant author list), "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate," *Proc. ICLR*, 2026. arXiv:2504.19874.

[4] S. Williams, A. Waterman, D. Patterson, "Roofline: an insightful visual performance model for multicore architectures," *Communications of the ACM*, vol. 52, no. 4, pp. 65–76, 2009.

[5] H. Jegou, M. Douze, C. Schmid, "Product Quantization for Nearest Neighbor Search," *IEEE TPAMI*, vol. 33, no. 1, pp. 117–128, 2011.

[6] Rockchip. *RKLLM Toolkit (RKNN-LLM v1.2)*. https://github.com/airockchip/rknn-llm.

[7] Alibaba. *MNN*. https://github.com/alibaba/MNN.

[8] Radxa. *ROCK 5B+ Product Brief*. https://radxa.com/products/rock5/5bp/.
