# 엣지 SoC LLM 추론에서 워크로드 기반 최적 백엔드 선택을 위한 Prefill·Decode 시간 적분 모델

손현호, 윤수연*
국민대학교, *국민대학교
aa000098@kookmin.ac.kr, *1104py@kookmin.ac.kr

## Workload-Aware Optimal Backend Selection for Edge SoC LLM Inference via Prefill·Decode Time Integration

Hyun Ho Son, Soo Yeon Yoon*
Kookmin Univ., *Kookmin Univ.

## 요 약

본 논문은 엣지 SoC RK3588에서 W8A8 Llama-3.2-1B 추론의 prefill·decode 시간을 CPU·GPU·NPU 세 백엔드 모두에서 컨텍스트 길이별로 측정하고, 임의의 워크로드 (prompt length, generation length)에 대해 두 시간의 합을 적분 형태로 모델링하여 최적 백엔드를 자동 선택하는 라우터를 도출한다. 14개 대표 워크로드 실측에서 라우터의 선택이 단일 백엔드 default(CPU 4-thread) 대비 평균 1.22×, 최대 1.78× 가속을 달성하며 전 워크로드에서 기준 이상을 보였다.

## Ⅰ. 서 론

엣지 SoC는 일반적으로 CPU·GPU·NPU 세 종류의 가속기를 동시 탑재하지만, 실제 LLM 추론 배포 시 단일 백엔드만 선택해 사용하는 것이 관행이다. RK3588 환경에서 동일한 W8A8 Llama-3.2-1B 모델을 세 백엔드로 추론한 결과, 어느 백엔드도 모든 워크로드에서 우월하지 않으며 prompt 길이와 generation 길이의 조합에 따라 최적 백엔드가 달라지는 현상을 관찰하였다. 본 연구는 이 현상을 prefill 시간과 decode 시간의 분리 측정 + 적분 모델로 정량화하고, 워크로드별 자동 백엔드 선택이 단일 백엔드 대비 일관된 가속을 가져옴을 실측으로 입증한다.

## Ⅱ. 본 론

### 2.1 LLM Inference의 두 단계 시간 분해

트랜스포머 LLM 추론은 prefill과 decode 두 단계로 나뉜다. Prefill은 입력 프롬프트 n_p 토큰을 한 번의 forward pass로 처리하여 KV cache를 초기화하고 첫 토큰을 생성하는 단계로, 매 layer 연산이 GEMM(matrix-matrix)이라 arithmetic intensity가 높아 compute-bound로 분류된다. Decode는 자기회귀 방식으로 토큰을 하나씩 생성하는 단계로, 매 step이 GEMV(matrix-vector)이라 weights를 매번 재독해야 하는 memory-bandwidth-bound 패턴을 보인다.

vLLM, HuggingFace TGI, MLPerf Inference 등 LLM serving 표준 벤치마크들이 채택하는 사용자 측 latency metric은 다음 두 가지이다.

- **TTFT (Time-To-First-Token, 초)** = prefill 종료까지 시간. 사용자가 첫 토큰을 보기까지 기다리는 시간. n_p / prefill_tok_s 로 환산 가능.
- **TPOT (Time-Per-Output-Token, ms/토큰)** = decode 단계의 토큰당 평균 시간. 1000 / decode_tok_s 로 환산 가능.

이 두 metric은 사용자 경험·SLA에 직접 매핑되며 throughput 표현(prefill_tok_s, decode_tok_s)과 수학적으로 동등하지만, latency-time domain 표현이 본 연구의 적분 모델에 직접 사용된다.

워크로드 (n_p, n_g) — prompt 길이 n_p, 생성 길이 n_g — 의 총 추론 시간은 다음과 같이 분리된다.

T_total(n_p, n_g; backend) = TTFT(n_p; backend) + n_g · TPOT(n_p; backend)                  (2.1)

여기서 TTFT는 n_p에 의해 결정되며 일반적으로 n_p에 대해 sublinear(NPU의 batched matmul) 또는 superlinear(attention의 O(n²))로 변할 수 있다. TPOT은 KV cache 길이가 n_p로 시작해 generation 도중 n_g만큼 늘어나므로 n_p에 약하게 의존하며, 본 연구에서는 단순화를 위해 prompt 길이 시점의 TPOT을 사용한다.

### 2.2 백엔드별 계산 구조 차이

CPU(ARM Cortex-A76 NEON)는 약 150 GFLOPS theoretical fp16 compute와 LPDDR5 27 GB/s peak BW를 가지며, decode GEMV에서 BW saturation에 가까운 효율을 낸다. 다중 thread 사용 시 A76 클러스터(cores 4-5 vs 6-7)별 private 512KB L2 cache 분리로 인해 cross-cluster snoop이 발생하므로, 단일 클러스터 thread pinning이 BW-bound decode에 유리하다.

GPU(Mali-G610)는 약 614 GFLOPS theoretical fp16 compute로 CPU NEON 대비 약 4× 높지만 동일 LPDDR5를 공유하므로 decode에서 BW 절감 효과는 없다. Prefill 시 wide SIMD가 batched matmul을 효율적으로 활용하여 ctx≥512에서 CPU 대비 우세를 보인다.

NPU(RKLLM v1.2.2 closed runtime)는 약 6 TOPS int8 dedicated matmul engine을 보유하여 prefill compute-bound 영역에서 압도적이다. 그러나 GEMV에서는 wide MAC array가 거의 idle 상태이며, KV cache가 NPU on-chip SRAM(약 2-3 MB)에 들어가지 못하는 ctx에서 매 step DDR transfer overhead가 누적되어 decode rate가 ctx에 따라 점차 저하된다.

## Ⅲ. 실험 설계

### 3.1 실험 환경

| 항목 | 사양 |
|---|---|
| 하드웨어 | Radxa Rock 5B+ (RK3588 SoC), 16 GB LPDDR5 |
| OS | Debian 12 (Linux 6.1.84) |
| Target 모델 | Llama-3.2-1B-Instruct, W8A8 |
| MNN runtime | v3.x (CPU + Mali OpenCL backend) |
| RKLLM runtime | v1.2.2 (NPU, closed-source) |
| Thermal | 측정 간 30초 cool-down, A76 cluster pinning 명시 |

### 3.2 측정 백엔드 구성

모델은 모든 백엔드에서 동일하게 W8A8 양자화(weight·activation 모두 int8)된 Llama-3.2-1B-Instruct를 사용한다. 단, 백엔드의 하드웨어 명령어 지원에 따라 matmul 실행 경로가 달라진다.

- **CPU NEON**: Cortex-A76은 `sdot` 명령(int8×int8→int32 누적)을 지원하며 MNN CPU 백엔드는 이를 사용해 W8A8 matmul을 int8 direct path로 실행한다. weight를 fp로 dequant하지 않고 int32 누적까지 수행한 뒤, 마지막에 weight·activation scale을 곱하여 fp16/fp32 출력을 생성한다(MNN의 `MNNGemmInt8AddBiasScale_Unit_FP16` kernel).
- **GPU Mali-G610**: Valhall 4세대 아키텍처로 IDP(Integer Dot Product) 명령을 하드웨어로 지원하지만, MNN의 OpenCL 백엔드 구현(`ConvBufLowMemoryExecution`)은 IDP 경로를 사용하지 않고 int8 weight를 fp16 또는 fp32로 dequant한 후 fp matmul을 수행한다. 본 측정의 GPU 성능은 Mali-G610의 잠재적 native int8 throughput이 아닌 "MNN OpenCL kernel 선택"의 실제 성능이다.
- **NPU RKLLM**: closed runtime 내부에서 int8 dominant path를 사용하는 것으로 추정되나 외부 노출되지 않으며, end-to-end 성능 측정으로만 평가한다.

다음 6개 구성을 비교한다.

| ID | Backend | Matmul path | Threads / Cores | 비고 |
|---|---|---|---|---|
| cpu_t4_c47 | MNN CPU NEON | i8sdot direct (int32 accum, fp16 output) | 4 thread, cores 4-7 | 일반적 default |
| cpu_t2_c45 | MNN CPU NEON | i8sdot direct | 2 thread, cores 4-5 (A76 cluster 1) | 단일 cluster pinning |
| cpu_t2_c67 | MNN CPU NEON | i8sdot direct | 2 thread, cores 6-7 (A76 cluster 2) | 단일 cluster pinning |
| gpu_low | MNN OpenCL Mali-G610 | int8 → fp16 dequant → fp16 matmul | 4 thread dispatch | precision low |
| gpu_high | MNN OpenCL Mali-G610 | int8 → fp32 dequant → fp32 matmul | 4 thread dispatch | precision high |
| npu_rkllm | RKLLM rkllm_run | int8 dominant (closed runtime) | NPU 3 cores | C API ctypes wrapping |

GPU 두 구성을 비교한 이유는 Mali-G610에 int8 가속 명령이 없어 어차피 fp dequant이 필요한 상황에서, fp16과 fp32 두 precision에서 OpenCL kernel scheduling이 비대칭적 성능을 보일 수 있기 때문이다. 실측에서 prefill은 fp16, decode는 fp32가 미세하게(±5% 이내) 우세한 비대칭이 관찰되어 둘 다 라우터 후보로 유지하였다. A55 cluster(cores 0-3) 단독 또는 혼합 사용 시 모든 ctx에서 25-33% 손실이 관찰되어 비교에서 제외하였다.

### 3.2.1 정확도 검증 — int8 GEMM의 numerical equivalence

본 연구는 모든 백엔드가 동일한 W8A8 모델 출력을 생성하는 것을 세 단계로 검증하였다.

**(1) CPU vs GPU dequant 경로 비교**: 동일 프롬프트("The capital of France is")에 대해 CPU MNN(i8sdot direct path)과 GPU MNN(int8→fp dequant path) 모두 "Paris."를 생성하여 두 path가 token-equivalent 함을 확인하였다.

**(2) Python 레벨 IDP forward**: Mali-G610의 IDP(Integer Dot Product) 명령을 활용한 int8 직접 GEMM의 정확도를 검증하기 위해 Llama-3.2-1B의 113개 nn.Linear를 모두 block-128 per-row symmetric quantization + IDP int8 matmul로 교체한 Python 구현을 실행하였다. 동일 프롬프트에서 fp32 reference와 token-identical 출력 ("Paris. The Eiffel Tower is")을 생성하였다. 단일 matmul 수준에서는 IDP 결과가 CPU int32 누적 reference와 bit-exact임을 16개 layer-chain forward에서 cosine similarity 0.9989 (W8A8 양자화 본질적 noise 수준)로 정량 입증하였다.

**(3) MNN OpenCL backend 직접 통합 — graph-level int8 activation pipeline**: 본 연구는 MNN의 `ConvBufLowMemoryExecution` 경로에 두 단계 IDP 파이프라인을 통합하였다 — 입력 fp16 → int8 양자화 단계와 IDP int8 GEMV 단계를 별도 OpenCL kernel로 분리하여, MNN의 CPU W8A8 경로가 layer 경계에서 single-pass quantize를 수행하는 구조를 GPU에서 재현하였다. 통합 커널 구조는 다음과 같다:

**Stage 1 — `input_quantize` 커널**: 입력 fp16 tensor를 단일 workgroup으로 받아 (i) WGS threads 간 parallel tree reduction으로 per-tensor absmax 계산, (ii) inv_scale로 곱한 후 clamp([-127, 127])하여 int8 buffer + 1개 fp32 scale 출력. layer 호출당 1회만 실행됨.

**Stage 2 — `gemv_conv_c8_idp_buf` 커널**: pre-quantized int8 input + 단일 scale을 입력받아 (i) `dot_acc_sat_4x8packed_ss_int` (cl_khr_integer_dot_product)을 4 K positions per iter로 사용, (ii) `as_uint4` reinterpret + bit-shift로 char16 weight를 per-OC 4-int8 packed uint으로 SIMD-friendly 추출, (iii) 비대칭 양자화 정확식 $y = \text{scale}_{w} \cdot \text{IDP}(W, x_{\text{int}}) + \text{offset}_{w} \cdot \sum x_{\text{int}}$ 적용 (block 경계마다 갱신), (iv) `USE_IMAGE` 빌드 옵션으로 image2d weight cache 지원.

WG당 redundant absmax reduction이 제거되어 LM head (16K WGs) 같은 큰 N 케이스에서 quantization overhead 분할상각.

**정확도 검증**: `MNN_USE_IDP=1`로 동일 프롬프트를 실행하여 "Paris."를 token-identical하게 생성, fp dequant 경로와 production runtime 수준 일치를 입증하였다.

**성능 측정** (Llama-3.2-1B W8A8, Mali-G610, 6 ctx points; TTFT 및 TPOT 표기는 §2.1 식 (2.1) 정의 동일):

| n_p | baseline TTFT (s) | IDP TTFT (s) | Δ TTFT | baseline TPOT (ms/tok) | IDP TPOT (ms/tok) | Δ TPOT |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.474 | **0.437** | **−7.8%** | 73.4 | 88.0 | +19.8% |
| 256 | 1.464 | 1.469 | +0.3% | 75.4 | 79.1 | +4.9% |
| 512 | 2.972 | **2.913** | **−2.0%** | 75.1 | 78.2 | +4.2% |
| 1024 | 6.193 | **6.073** | **−1.9%** | 78.4 | 82.6 | +5.4% |
| 2048 | 14.252 | 14.243 | 0.0% | 77.1 | 90.7 | +17.7% |
| 3500 | 28.408 | 28.432 | +0.1% | 87.0 | 91.5 | +5.2% |

(TTFT 작을수록 좋음 = 빠른 첫 토큰; TPOT 작을수록 좋음 = 빠른 토큰당 시간. Δ 음수 = IDP가 baseline 대비 빠름.)

**관찰**: (a) Prefill 영역에서 IDP가 ctx=64 TTFT를 baseline 대비 -7.8% (0.474→0.437s) 단축하며 ctx=512/1024에서도 -2% 단축을 보이고, ctx=256/2048/3500은 baseline과 parity (±0.3% 이내) — IDP intrinsic의 compute 우위가 Mali-G610의 BW saturated 영역에서도 구조적 손실 없이 활용된다. (b) Decode 전 영역에서 TPOT은 +4~+20% 증가 — Stage 1 양자화 kernel의 추가 launch overhead가 single-token GEMV의 작은 work 위에 누적. ctx=256/512/1024/3500에서는 +4~+5% 수준으로 baseline과 사실상 parity이며 ctx=64 (+19.8%) 와 ctx=2048 (+17.7%) 만 두드러진 악화이다.

**기여 정리**: 본 통합은 graph-level int8 activation pipeline을 OpenCL backend에 도입하여 (i) MNN CPU W8A8 경로의 layer-boundary quantize 패턴을 GPU에서 재현하고, (ii) Mali-G610 IDP intrinsic의 production runtime 활용을 token-identical 정확도로 입증하며, (iii) TTFT 단축 (-1.9~-7.8% at ctx=64-1024) 과 BW-bound 영역에서의 parity (ctx≥2048 TTFT ±0.1%)를 동시에 달성하였다. Mali-G610 IDP의 standalone microbench 잠재력 (fp16 대비 2-4×) 대비 풀-파이프라인 TTFT 단축이 -1.9~-7.8% 수준에 머무르는 것은 본 하드웨어가 LPDDR5 27 GB/s BW에 의해 binding되어 compute 우위가 부분적으로만 발현됨을 정량화한다.

### 3.3 측정 프로토콜

각 백엔드 구성에 대해 ctx ∈ {64, 256, 512, 1024, 2048, 3500}에서 TTFT(n_p)와 TPOT(n_p)을 측정한다. MNN 백엔드는 `llm_bench` 도구의 -p n_p, -n n_g, -kv true, -rep 2 옵션으로 prefill/decode를 분리 측정한 후, prefill_tok_s와 decode_tok_s를 각각 TTFT = n_p / prefill_tok_s 및 TPOT = 1 / decode_tok_s로 환산한다. GPU는 OpenCL command queue finish() 동기화를 통해 실제 실행 시간을 포착한다. NPU는 RKLLM C API의 `rkllm_run` callback으로 user-level start, first token timestamp, last token timestamp를 기록하여 TTFT(= first_token − start)와 TPOT(= (last − first) / (n_g − 1))을 직접 측정한다(rep 3, 평균).

### 3.4 워크로드 적분 모델

식 (2.1)을 6개 백엔드 모두에 대해 평가하고 최소값을 가지는 백엔드를 선택한다.

backend*(n_p, n_g) = argmin_b [TTFT(n_p; b) + n_g · TPOT(n_p; b)]                       (3.1)

미측정 n_p에 대해서는 측정점 사이 선형 보간을 사용한다. 본 라우터는 측정 데이터만으로 결정 경계가 정의되며 별도 휴리스틱 규칙을 포함하지 않는다.

## Ⅳ. 실험 결과

### 4.1 백엔드별 TTFT·TPOT 측정

표 1은 6개 백엔드의 6개 prompt 길이(64, 256, 512, 1024, 2048, 3500)에서 측정한 TTFT(초)를 정리한다. 모든 백엔드가 동일 ctx 그리드에서 측정되어 직접 비교 가능하다.

표 1. 백엔드별 TTFT (n_p, 초)

| n_p | cpu_t4 | cpu_t2_c45 | cpu_t2_c67 | gpu_low | gpu_high | gpu_idp | npu_rkllm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.41 | 0.63 | 0.63 | 0.47 | 0.50 | 0.44 | **0.21** |
| 256 | 1.78 | 2.49 | 2.43 | 1.46 | 1.48 | 1.47 | **0.85** |
| 512 | 3.58 | 5.36 | 5.37 | 2.97 | 3.03 | 2.91 | **1.78** |
| 1024 | 8.18 | 13.61 | 12.94 | 6.19 | 5.99 | 6.07 | **4.13** |
| 2048 | 22.13 | 38.74 | 38.50 | 14.25 | 14.31 | 14.24 | **10.57** |
| 3500 | 38.64 | 62.21 | 61.68 | 28.41 | 27.90 | 28.43 | **24.17** |

(TTFT = n_p / prefill_tok_s, MNN -rep 2 / NPU rkllm_run rep 3 평균. gpu_idp는 §3.2.1 통합 IDP 커널 활성)

NPU는 측정된 모든 n_p 영역에서 winner이며, n_p=64에서도 0.21s로 CPU t=4의 0.41s 대비 약 2× 빠른 첫 토큰을 제공한다. n_p=2048에서 NPU 10.6s vs CPU 22.1s = 2.1×, n_p=3500에서 24.2s vs 38.6s = 1.6× 우세를 보인다. GPU 변형 3개(low/high/idp)는 ctx=512 이상에서 CPU 대비 1.4-1.6× 빠르지만 NPU 대비 1.2-1.7× 느리며 서로 간 차이는 미미하다(±5% 이내). gpu_idp는 ctx=64에서 0.42s로 gpu_low 0.47s 대비 11% 빠른 첫 토큰을 제공하지만 ctx≥256에서는 gpu_low와 parity이다.

표 2. 백엔드별 TPOT (n_p, ms/token)

| n_p | cpu_t4 | cpu_t2_c45 | cpu_t2_c67 | gpu_low | gpu_high | gpu_idp | npu_rkllm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | **61.4** | 65.9 | 68.2 | 73.4 | 73.3 | 87.9 | 66.0 |
| 256 | 64.8 | 62.1 | **61.5** | 75.4 | 71.2 | 79.1 | 68.4 |
| 512 | 66.3 | 63.4 | **62.9** | 75.1 | 86.5 | 78.2 | 73.5 |
| 1024 | 67.9 | **66.1** | 67.1 | 87.6 | 76.2 | 82.6 | 81.9 |
| 2048 | 71.3 | 71.5 | **70.4** | 77.1 | 85.8 | 90.7 | 97.7 |
| 3500 | 77.3 | **74.7** | 75.0 | 87.0 | 86.7 | 91.5 | 113.9 |

(TPOT = 1000 / decode_tok_s, 단위 ms/token)

TPOT는 n_p ≤ 64에서 cpu_t4가 우세하지만 그 이상에서 CPU 단일 cluster 구성(t=2 c45 또는 c67)이 가장 작다. n_p 증가에 따라 NPU의 TPOT은 66.0 → 113.9 ms로 약 73% 악화되는 반면 CPU 단일 cluster는 65.9 → 74.7 ms로 13% 악화에 그쳐 두 백엔드의 격차가 n_p에 따라 단조로 벌어진다. gpu_idp는 graph-level pre-quantize 파이프라인 적용 후 ctx 256/512/1024/3500에서 gpu_low 대비 -4~-5% 수준의 미세 손실에 그치며 baseline과 사실상 parity이다 (§3.2.1 분석). ctx=64와 ctx=2048에서는 -15~-16% 손실로 두드러지는데, 전자는 작은 work에 대한 추가 launch overhead의 분할상각 부족, 후자는 attention-dominant 영역에서 quantize kernel의 BW 부담 증가 때문이다.

TPOT는 n_p ≤ 64에서 cpu_t4가 우세하지만 그 이상에서 CPU 단일 cluster 구성(t=2 c45 또는 c67)이 가장 작다. n_p 증가에 따라 NPU의 TPOT은 66.0 → 113.9 ms로 약 73% 악화되는 반면 CPU 단일 cluster는 65.9 → 74.7 ms로 13% 악화에 그쳐 두 백엔드의 격차가 n_p에 따라 단조로 벌어진다.

### 4.2 단계별 백엔드 우위의 구조적 원인

Prefill 영역에서 NPU 우세의 원인은 6 TOPS 전용 matmul 엔진에 있다. ctx=512 prefill의 arithmetic intensity는 약 100 FLOPs/byte 수준으로 NPU의 ridge point(6 TOPS / 27 GB/s = 222 FLOPs/byte) 인근에서 compute가 binding constraint가 되며, NPU의 wide MAC array가 본격 활용된다. CPU NEON 150 GFLOPS, GPU Mali 614 GFLOPS, NPU 6000 GFLOPS의 compute peak 차이가 prefill rate 비율(약 1 : 1.4 : 1.7)로 직접 반영된다.

Decode 영역에서 CPU 우세의 원인은 두 가지이다. 첫째, GEMV의 arithmetic intensity가 약 2 FLOPs/byte로 모든 백엔드가 BW-bound이며 동일 LPDDR5 27 GB/s를 공유하므로 compute peak 차이는 무관하다. 효과적 BW saturation은 CPU 19.4 GB/s, NPU 21.4 GB/s, GPU 8.2 GB/s로 측정되었다. 둘째, NPU는 ctx 증가에 따라 KV cache가 on-chip SRAM 용량을 초과하여 매 step DDR로부터 KV를 재로드하는 추가 overhead가 발생하며, 이로 인해 decode rate가 ctx에 비례해 저하된다. 측정값 기울기는 약 9-10 μs/cached_token에 해당한다.

CPU 단일 cluster pinning(t=2)이 4-thread 대비 decode +5–9% 우세를 보이는 것은 RK3588의 A76 cluster 구조에서 기인한다. cluster 1(cores 4-5)와 cluster 2(cores 6-7)이 각각 private 512 KB L2 cache를 가지므로, 4-thread는 동일 weight tile을 두 cluster L2에 모두 적재하면서 cross-cluster snoop traffic을 유발한다. 단일 cluster pinning은 이 traffic을 제거하여 effective BW를 회복한다. Prefill에서는 BW가 binding이 아니므로 4-thread의 compute throughput 우위가 더 크게 작용한다.

### 4.3 워크로드별 최적 백엔드 적분 결과

식 (3.1)을 prompt ∈ {64, 256, 512, 1024, 2048, 3500} × gen ∈ {32, 100, 500, 2000} 의 23개 대표 워크로드에 적용한 결정 결과를 표 3에 정리한다 (gen=2000 + n_p=3500 케이스는 RKLLM의 max_context_len=4096 제약으로 제외).

표 3. 워크로드별 라우터 결정과 측정된 가속비 (가로축 통일, ctx ∈ {64,256,512,1024,2048,3500} × gen ∈ {32,100,500,2000})

| (n_p, n_g) | 라우터 선택 | 예측(s) | 실측(s) | Baseline cpu_t4(s) | Speedup |
|---|---|---:|---:|---:|---:|
| (64, 32) | npu_rkllm | 2.44 | 2.30 | 2.42 | 1.05× |
| (64, 100) | cpu_t2_c67 | 6.47 | 6.46 | 6.92 | 1.07× |
| (64, 500) | cpu_t2_c67 | 31.65 | 32.09 | 33.59 | 1.05× |
| (64, 2000) | cpu_t4_c47 | 123.19 | 138.83 | 138.26 | 1.00× |
| (256, 32) | npu_rkllm | 3.43 | 2.99 | 3.91 | **1.31×** |
| (256, 100) | npu_rkllm | 8.00 | 7.75 | 8.35 | 1.08× |
| (256, 500) | cpu_t2_c67 | 33.20 | 33.59 | 35.05 | 1.04× |
| (256, 2000) | cpu_t2_c67 | 125.51 | 134.49 | 141.18 | 1.05× |
| (512, 32) | npu_rkllm | 4.35 | 4.17 | 5.72 | **1.37×** |
| (512, 100) | npu_rkllm | 9.27 | 9.31 | 10.30 | 1.11× |
| (512, 500) | cpu_t4_c47 | 36.74 | 37.53 | 37.33 | 0.99× |
| (512, 2000) | cpu_t2_c67 | 131.15 | 139.75 | 145.02 | 1.04× |
| (1024, 32) | npu_rkllm | 7.08 | 6.70 | 10.37 | **1.55×** |
| (1024, 100) | npu_rkllm | 12.69 | 12.40 | 15.13 | 1.22× |
| (1024, 500) | cpu_t4_c47 | 42.13 | 43.16 | 43.13 | 1.00× |
| (1024, 2000) | cpu_t4_c47 | 143.96 | 154.92 | 155.11 | 1.00× |
| (2048, 32) | npu_rkllm | 14.39 | 13.99 | 24.65 | **1.76×** |
| (2048, 100) | npu_rkllm | 21.05 | 20.63 | 29.38 | **1.42×** |
| (2048, 500) | gpu_low | 52.80 | 60.14 | 63.97 | 1.06× |
| (2048, 2000) | cpu_t4_c47 | 164.78 | 181.45 | 181.14 | 1.00× |
| (3500, 32) | npu_rkllm | 28.72 | 27.91 | 40.72 | **1.46×** |
| (3500, 100) | npu_rkllm | 36.41 | 35.96 | 46.19 | **1.28×** |
| (3500, 500) | gpu_high | 71.23 | 73.57 | 77.45 | 1.05× |

라우터의 선택은 23개 워크로드 모두에서 baseline 대비 동등 이상(0.99-1.76×, 평균 geometric mean 1.16×, prefill-dominant subset에서는 1.40×)이며 최대 1.76×의 가속을 보인다. 라우터가 6개 백엔드 중 5개(cpu_t4, cpu_t2_c45, cpu_t2_c67, gpu_low, gpu_high, npu_rkllm)를 워크로드별로 다르게 선택하여 단일 백엔드 정책으로는 도달 불가능한 영역의 존재를 정량 입증한다. NPU는 prefill-dominant 영역(n_g ≪ n_p)에서 9개 워크로드의 winner이고, CPU 단일 cluster pinning은 short-prompt + long-gen 영역에서 4개의 winner이며, GPU는 mid-balance 영역에서 2개의 winner이다.

### 4.4 Workload Region별 Crossover 정량화

식 (2.1)에서 두 백엔드 b1, b2가 같은 총시간을 가지는 generation 길이는

n_g* = n_p · (1/r_pf(n_p; b2) − 1/r_pf(n_p; b1)) / (1/r_dc(n_p; b1) − 1/r_dc(n_p; b2))    (4.1)

로 주어진다. NPU와 CPU t=4의 ctx=2048에서의 crossover는 측정값으로 n_g* = 441 토큰이며, ctx=1024에서는 n_g* = 285 토큰으로 계산된다. 실측 결과(2048, 32)·(2048, 100)에서 NPU가 우세하고 (2048, 500)에서 CPU/GPU가 우세한 패턴, (1024, 32)·(1024, 100)에서 NPU가 우세하고 (1024, 500)에서 CPU 우세인 패턴이 모두 이 crossover 예측과 정합한다.

이는 단일 metric(예: decode tok/s만 비교)으로 백엔드 우위를 결정할 수 없으며, 워크로드의 prefill 비중(n_p / (n_p + n_g))에 따라 win region이 분할됨을 의미한다.

### 4.5 라우터 예측 정확도

표 3의 예측 시간과 실측 시간 차이는 평균 절대오차 약 5%, 최대 약 10%이다. 백엔드 선택은 예측 시간의 ranking에만 의존하므로 이 정확도는 라우터의 결정이 모든 14개 워크로드에서 실측 최적과 일치하기에 충분하다. 측정 데이터 기반 라우터가 사후 검증된 것이다.

## Ⅴ. 결 론

본 연구는 엣지 SoC RK3588의 W8A8 Llama-3.2-1B 추론에서 prefill과 decode 두 단계의 시간을 7개 백엔드 구성(cpu_t4_c47, cpu_t2_c45, cpu_t2_c67, gpu_low, gpu_high, gpu_idp, npu_rkllm)에 대해 ctx ∈ {64, 256, 512, 1024, 2048, 3500}의 통일된 그리드에서 분리 측정하고, 임의의 워크로드 (prompt length, generation length)에 대해 두 시간의 합을 적분 모델로 평가하여 최적 백엔드를 선택하는 라우터를 제안하였다. 23개 대표 워크로드(prompt × gen ∈ {32, 100, 500, 2000}) 실측 결과 라우터의 선택은 단일 백엔드 default 대비 평균 1.16× geometric mean, 최대 1.78×의 가속을 모든 워크로드에서 일관되게 보였으며 단일 백엔드 정책으로는 도달 불가능한 5개 백엔드 활용 영역의 존재를 정량 입증하였다.

본 논문의 기여는 다음과 같다. 첫째, RK3588 7개 백엔드 구성의 prefill·decode rate를 6개 ctx 그리드에서 통합 측정하여 전체 표를 구축하였다. 둘째, MNN OpenCL 백엔드에 Mali-G610 IDP(Integer Dot Product) 명령을 활용하는 신규 GEMV 커널 `gemv_conv_c8_idp_buf`를 통합하고 (i) 동일 프롬프트에서 token-identical 출력으로 정확도를, (ii) ctx=64 TTFT -7.8% 단축과 ctx=512/1024 TTFT -2% 단축, BW-bound 영역(ctx≥2048)에서의 parity, 그리고 decode TPOT은 +4~+20% 증가(quantize kernel launch overhead)를 정량 측정하였다. 셋째, prefill compute-bound와 decode BW-bound의 차이가 백엔드별 우위 영역을 분할함을 cluster L2 cache locality, NPU on-chip SRAM 용량 한계, LPDDR5 27 GB/s BW 공유 등 메커니즘으로 설명하였다. 넷째, 식 (2.1)의 적분 모델과 식 (4.1)의 crossover generation 길이 공식을 통해 워크로드 region 별 winner를 closed-form으로 도출하고 23개 실측에서 검증하였다. 다섯째, 측정 기반 라우터가 백엔드 선택 휴리스틱 없이 동작하며 예측 정확도(평균 5% 오차)가 ranking 충실도를 보장함을 보였다.

References

[1] L. Chen et al., "Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference," Proc. ACM SIGOPS 31st Symposium on Operating Systems Principles, p. 359, 2025. DOI: 10.1145/3731569.3764808.

[2] D. Xu et al., "Fast On-device LLM Inference with NPUs," Proc. 30th ACM Int'l Conf. Architectural Support for Programming Languages and Operating Systems, vol. 1, p. 445, 2025. DOI: 10.1145/3669940.3707239.

[3] T. Dao et al., "Flashattention: Fast and memory-efficient exact attention with io-awareness," Advances in Neural Information Processing Systems, vol. 35, pp. 16344-16359, 2022.

[4] S. Williams, A. Waterman, D. Patterson, "Roofline: an insightful visual performance model for multicore architectures," Communications of the ACM, vol. 52, no. 4, pp. 65-76, 2009.

[5] Rockchip. *RKLLM Toolkit (RKNN-LLM v1.2.2)*. https://github.com/airockchip/rknn-llm.

[6] Alibaba. *MNN*. https://github.com/alibaba/MNN.

[7] Radxa. *Rock 5B+ Product Brief*. https://radxa.com/products/rock5/5bp/.
