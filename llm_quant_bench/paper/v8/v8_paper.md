# RK3588 엣지 SoC LLM 추론: 백엔드별 효율 분석과 ARM NEON KV Cache Codebook 압축 커널

손현호, 윤수연*
국민대학교, *국민대학교
aa000098@kookmin.ac.kr, *1104py@kookmin.ac.kr

**LLM Inference on RK3588 Edge SoC: Per-Backend Efficiency Analysis and ARM NEON KV Cache Codebook Compression Kernel**
Hyun Ho Son, Soo Yeon Yoon*
Kookmin Univ., *Kookmin Univ.

---

## 요 약

엣지 SoC RK3588은 NPU, Cortex-A76 CPU, Mali-G610 GPU 세 가속기가 LPDDR5 메모리를 공유하므로, decode 단계의 memory-bound 특성상 메모리 대역폭이 LLM 추론 성능의 공통 한계로 작용한다. 본 논문은 LPDDR5 단일 방향 read 대역폭으로부터 Williams roofline 기반 decode 처리량 상한을 도출하고, Llama-3.2-1B W8A8 동일 양자화 조건에서 세 backend의 prefill·decode 효율을 다양한 context·generation length에 걸쳐 측정·비교한다. 또한 NPU 측정 방법론(Python ctypes incremental loop vs `rkllm_run` one-shot)이 NPU 효율 평가에 미치는 영향을 분리하고, GPU 전용으로 평가되어온 codebook 기반 KV cache 압축(Product Quantization)을 ARM Cortex-A76용 NEON 커널로 직접 구현해 측정한다. 측정 결과: (1) roofline 대비 backend별 효율은 ctx에 따라 다르게 거동하며 effective ctx 1축으로 backend 선택 규칙이 모델링됨, (2) NPU model compute는 v7의 Python loop 측정 대비 ctx=3500에서 1.98× 빠름 (roofline의 28% → 55%), (3) MNN의 빌트인 KV quant 옵션은 ARM Cortex-A76(i8mm 부재)에서 baseline보다 모두 느림, (4) 본 논문의 PQ codebook NEON 커널은 fp16 baseline 대비 attention compute에서 13.8× speedup + 32× KV 메모리 압축 달성, 같은 알고리즘을 Mali-G610 OpenCL로 구현 시 5× 느려짐(architecture-conditional 최적화).

---

## Ⅰ. 서 론

엣지 SoC RK3588은 NPU(6 TOPS), Cortex-A76 4코어 CPU, Mali-G610 GPU 세 가속기를 LPDDR5 메모리에 공유한다. 모두 LLM 추론에 사용 가능한 backend이지만, 어느 backend가 어느 시나리오에서 우세한지에 대한 정량 비교 publication은 부재하며 NPU가 LLM 가속에 절대 우세하다는 인식이 일반적이다. 모바일 SoC 대상 선행 연구는 있으나[1, 2] RK3588에서 동일 양자화 조건으로 세 backend를 직접 비교한 publication은 우리가 아는 한 보고된 바 없다.

LLM 추론은 prefill과 decode 두 단계로 나뉘며 두 단계의 연산 특성이 근본적으로 다르다. Prefill은 prompt 전체에 대한 batch GEMM으로 compute-bound에 가깝고, decode는 토큰별 GEMV로 memory bandwidth-bound이다. 세 backend가 동일 LPDDR5를 공유하므로 decode 처리량은 메모리 대역폭이라는 공통 상한 아래에서 backend별 효율(roofline efficiency)로 환산해 비교 가능하며, prefill에서는 GEMM 가속 정도, decode에서는 KV cache traversal 효율이 backend별로 차별화되어 단계별 우세 backend가 분리될 가능성이 있다.

본 연구는 Rock 5B+(RK3588) 보드에서 Llama-3.2-1B를 W8A8 동일 양자화로 세 backend에 변환하고, 5개 컨텍스트 길이(15, 512, 1024, 2048, 3500)와 6개 생성 길이(32, 128, 256, 512, 1024, 2048)에서 측정하여 (1) LPDDR5 메모리 대역폭 roofline 대비 각 backend의 decode 효율을 산출하고, (2) prefill·decode 단계에서 우세 backend의 분리 현상을 확인하고, (3) (ctx, n_gen) 조합에 따른 backend 1등 cross-over를 실측 및 정량 모델링한다.

## Ⅱ. 본 론

### 2.1 LLM Decode의 메모리 대역폭 한계

트랜스포머 LLM의 토큰별 decode는 매 단계에서 모델 weight 전체를 메모리에서 읽어 GEMV 연산을 수행한다. Williams 등[6]의 roofline 모델에 따르면 weight read가 dominant한 영역에서 토큰당 처리량 상한은

$$\text{tok/s}_{\max} = \frac{B_{\text{read}}}{S_{\text{weight}}} \tag{1}$$

이며, $B_{\text{read}}$는 보드의 단일 방향 read 대역폭, $S_{\text{weight}}$는 모델 weight 크기이다. Llama-3.2-1B의 weight 크기는 1.24B 파라미터 × 1 byte (W8A8) = **1.24 GB**이다.

### 2.2 RK3588 SoC 및 Rock 5B+ 구조

RK3588은 6 TOPS NPU(3 cores), Cortex-A76 ×4 + A55 ×4 CPU, Mali-G610 MP4 GPU를 SoC에 통합한다. 실험 보드 Rock 5B+는 RK3588에 LPDDR5 64-bit @ 5500 MT/s 메모리[7]를 결합하며 이론 peak는 5500 × 8 = **44 GB/s**이다. 모든 측정에서 A76 big core 4개를 사용하고 A55 little core는 비활성화한다(`taskset -c 4-7`). NPU는 closed-source RKLLM SDK v1.2.1b1[4]로, Mali GPU는 MNN OpenCL backend[3]로 접근한다.

## Ⅲ. 실험 설계

대상 모델은 Meta Llama-3.2-1B-Instruct를 W8A8 양자화로 RKLLM(NPU), MNN(CPU 4 threads, OpenCL Mali) 세 backend에 통일 변환한다. Wikitext-2 train split의 자연어 prefix를 5개 길이 *n* ∈ {15, 512, 1024, 2048, 3500}로 truncate하여 prompt로 사용하고, 기본 측정에서 n_gen=32 토큰을 생성한다. Long-generation 측정에서는 ctx=3500을 고정하고 n_gen ∈ {32, 128, 256, 512, 1024, 2048}을 측정한다. 모든 측정은 KV cache를 활성화하고(MNN: `-kv true`, RKLLM: `keep_history=1`) `-rep 3`으로 3회 반복하여 평균±std를 보고한다. Long-generation 측정에서는 backend 사이 5분 idle을 두어 thermal recovery를 확보했다(SoC 38 °C 회복 확인). 메모리 대역폭은 tinymembench v0.4.9[5]를 A76 4 core affinity pinning한 상태로 측정한다.

## Ⅳ. 실험 결과

### 4.1 단계별 우세 backend의 분리

LLM 추론의 두 단계는 연산 특성이 근본적으로 달라 backend별 우세 패턴도 분리되어 나타난다. 표 1은 prefill 단계의 TTFT, 표 2는 decode 단계의 토큰당 throughput을 정리한 것이다.

표 1. **Prefill 단계: TTFT** (sec) — Llama-3.2-1B W8A8

| ctx | NPU (RKLLM) | CPU MNN | Mali GPU |
|---:|---:|---:|---:|
| 15   | **0.128** | 0.201 | 0.269 |
| 512  | **1.95**  | 3.55  | 2.87  |
| 1024 | **4.42**  | 8.10  | 5.99  |
| 2048 | **10.85** | 21.74 | 14.34 |
| 3500 | **23.06** | 38.42 | 28.83 |

표 2. **Decode 단계**: tok/s 및 NPU의 roofline 22.0 대비 효율

| ctx | NPU | CPU MNN | Mali GPU | NPU/한계 |
|---:|---:|---:|---:|---:|
| 15   | **20.80** | 16.18 | 14.93 | **94%** |
| 512  | **15.09** | 15.01 | 13.76 | 69% |
| 1024 | 12.20 | **14.80** | 13.04 | 55% |
| 2048 | 8.46  | **13.84** | 12.68 | 38% |
| 3500 | 6.12  | **13.05** | 11.41 | 28% |

두 단계의 backend 1등이 분리되어 나타난다. **Prefill에서는 NPU가 모든 ctx에서 1등**이며 ctx=3500에서 CPU 대비 1.67×, GPU 대비 1.25× 빠르다 (그림 1). 이는 prefill이 prompt 전체에 대한 batch GEMM으로 compute-bound 영역에 가깝고, NPU의 6 TOPS GEMM 가속이 직접 활용되기 때문으로 해석된다. **Decode에서는 backend 거동이 ctx에 따라 정성적으로 달라지며, ctx ≥ 1024부터 CPU MNN이 NPU를 능가**한다 (ctx=3500에서 2.13× 격차). Decode가 토큰별 GEMV로 memory bandwidth-bound라는 특성상 LPDDR5 대역폭이 공통 한계로 작용하며, 그 한계 대비 효율로 backend별 거동을 §4.2에서 분석한다. GPU는 두 단계 모두에서 중간 성능을 보이며 어느 단계에서도 1등이 아니다.

![](v7_figures/fig_prefill_ttft.png)

그림 1. 컨텍스트 길이별 prefill TTFT (Llama-3.2-1B W8A8). 모든 ctx에서 NPU가 1등이며 ctx=3500에서 CPU 대비 1.67× 빠르다.

### 4.2 Decode 분석 — LPDDR5 메모리 대역폭 Roofline 기반

Decode는 토큰별 GEMV로 weight 전체와 누적 KV cache를 매 단계 read하므로 memory bandwidth-bound이며, 세 backend가 LPDDR5를 공유하는 RK3588에서는 그 대역폭이 세 backend 공통 상한으로 작용한다. tinymembench로 측정한 Rock 5B+의 처리량은 NEON copy 12.91 GB/s, fill 27.31 GB/s이다. NEON copy는 read와 write가 동시에 LPDDR 버스를 점유하므로 단일 방향 처리량 추정에 부적합하다. 본 논문은 보수적으로 **fill 27.31 GB/s를 단일 방향 read 대역폭의 upper bound proxy**로 채택하며, 이는 이론 peak 44 GB/s의 62%이다. 식 (1)에 대입하면 **decode 처리량 상한 = 22.0 tok/s**이다 (그림 2, 수평선).

표 2의 roofline 효율을 분석하면 NPU와 CPU/GPU의 거동이 정성적으로 다르다. NPU는 ctx=15에서 roofline의 94%로 메모리 대역폭 한계에 거의 도달하나 ctx=3500에서 28%로 추락한다. CPU MNN은 ctx=15→3500 구간에서 16.18→13.05 tok/s로 19% 감소에 그치며 roofline의 60–70% 영역에서 평탄 유지되며, Mali GPU도 14.93→11.41 tok/s (24% 감소)로 유사하게 평탄하다. **임계점은 ctx=1024**이며 이 이후 CPU MNN이 NPU를 능가한다. 동일 LPDDR5를 공유함에도 NPU만 ctx에 가파르게 비례 추락하는 비대칭 거동은 raw 대역폭이 아닌 SDK가 그 대역폭을 사용하는 방식의 차이를 시사하며, §4.3에서 분리·정량화한다.

![](v7_figures/fig_roofline.png)

그림 2. RK3588 LPDDR5 메모리 대역폭 roofline (수평선, 22.0 tok/s)과 ctx별 세 backend decode throughput. NPU는 ctx=15에서 roofline의 94%에 도달하나 ctx=3500에서 28%로 감소하는 반면, CPU·GPU는 평탄 유지된다.

### 4.3 NPU의 Decode ctx-비례 추가 비용 분리

NPU decode의 ctx 비례 비효율(28% @ ctx=3500)이 메모리 대역폭에서 발생하는지, SDK level에서 발생하는지를 분리한다. NPU의 토큰당 decode 시간은 ctx=15에서 48 ms, ctx=3500에서 163 ms로 ctx에 비례해 증가한다. §4.2의 실측 read 대역폭 proxy 27.31 GB/s(이론 peak 44 GB/s의 62%)를 사용하면 roofline에 따른 weight read 비용 하한은 1.24 GB / 27.31 GB/s = **45 ms/tok**이며 ctx에 무관해야 하므로, ctx=15→3500 구간 추가 비용 (163 − 48) = **115 ms**는 weight read 외에서 발생한다.

이 추가 비용을 attention KV-cache traversal로 설명할 수 있는지 검증한다. Llama-3.2-1B는 GQA(grouping factor 4)로 32 query head 대비 8 KV head만 저장하므로, 한 토큰의 KV는 16 layers × 8 KV heads × 64 head_dim × 2 (K+V) × 1 byte = 16,384 bytes/token (1024 bytes per layer·token)이고 ctx=3500의 누적 KV는 **57.3 MB**이다. read 대역폭 27.31 GB/s에서 57.3 MB read의 시간 한계는 **2.1 ms**이다. 그러나 NPU 실측 추가분 115 ms는 한계의 **약 55×**이다. 같은 구간에서 CPU MNN의 추가 비용은 (1000/13.05 − 1000/16.18) ≈ 14.8 ms로 동일 한계의 **약 7×**에 머문다.

이 추가 비용이 KV cache 자체의 비정상 traversal에서 오는 것은 아님을 다음 측정으로 확인했다. Prefill 후 8 토큰을 1개씩 incremental decode하여 토큰별 시간을 측정하면, 첫 토큰의 cold-start를 제외한 7 토큰에서 ctx=1024는 78.8–79.5 ms 범위(std 6.6 ms), ctx=3500은 153.2–157.1 ms 범위로, 토큰 추가에 따른 점진적 시간 증가가 없다. 즉 KV cache reuse 자체는 정상이며 NPU의 ctx-비례 추가 비용은 RKLLM SDK 내부 attention kernel 처리 비용에서 발생한다고 측정으로 확인된다(closed-source SDK라 root cause 직접 추적은 본 연구의 범위를 벗어난다).

세 backend가 동일 LPDDR5를 공유함에도 NPU만 ctx에 가파르게 비례(3.4×)하고 CPU(1.24×)·GPU(1.31×)는 거의 평탄한 비대칭은 raw 대역폭이 아닌 **SDK가 그 대역폭을 사용하는 방식의 차이**에서 발생한다. closed-source SDK라 직접 분리는 불가하나 후보 메커니즘은 다음과 같다. (i) **Attention kernel fusion 부재**: CPU MNN은 GQA attention을 NEON tile-streaming으로 QK^T·softmax·attn×V를 fuse하나, RKLLM은 세 단계를 별도 kernel로 처리해 KV가 K·V matmul에서 두 번 읽히고 softmax 중간 결과가 DRAM bounce하는 것으로 추정. (ii) **NPU SRAM ≪ KV cache로 인한 DMA 빈도**: RK3588 NPU SRAM ~768 KB(3 core 합계)는 ctx=3500의 layer당 KV 3.6 MB도 못 담아 매 layer × 매 토큰마다 K/V 전체 DMA가 필요하며, descriptor setup·command submit·completion 등 per-DMA fixed cost가 ctx-비례 DMA 횟수와 곱해진다. CPU는 동일하게 KV가 cache에 안 들어가지만 cache-line 단위 hardware prefetch가 read를 compute와 자동 overlap한다. (iii) **Kernel dispatch overhead**: 16 layers × attention sub-kernel multiple = 토큰당 수십 회 kernel launch가 ctx에 따라 chunk splitting된다면 dispatch 비용이 ctx-비례 누적된다. (iv) **Int8 dequant 분리**: int8 KV의 scale·zero-point 처리가 attention kernel 내 fused가 아니라 별도 op이면 KV가 추가 1회 더 traversal된다. 본 연구는 이 후보들을 정량 분리하기 위한 NPU profiler/DMA trace 접근을 향후 과제로 남긴다.

### 4.4 Long Generation에서의 백엔드 Cross-over (실측)

§4.2의 표는 n_gen=32 측정으로 decode 비중이 prefill 대비 작다. 실용적 시나리오(creative writing, code generation, agentic 응답)에서 n_gen이 더 클 때의 cross-over를 직접 측정하기 위해 ctx=3500을 고정하고 n_gen ∈ {32, 128, 256, 512, 1024, 2048}에서 세 backend의 overall throughput을 측정했다(표 3, 그림 3).

표 3. ctx=3500, varying n_gen (mean ± std, 3 repeats)

| n_gen | NPU (tok/s) | CPU MNN (tok/s) | Mali GPU (tok/s) |
|---:|---:|---:|---:|
| 32   | **1.146 ± 0.026** | 0.788 | 1.031 |
| 128  | 2.996 ± 0.006 | 2.648 | **3.323** |
| 256  | 4.063 ± 0.002 | 4.385 | **5.117** |
| 512  | 4.912 ± 0.008 | 6.519 | **7.154** |
| 1024 | 5.179 ± 0.257 | 8.659 | **8.701** |
| 2048 | 5.519 ± 0.196 | **10.084** | 9.228 |

¹ NPU는 자체 측정 스크립트로 mean과 std 동시 보고. CPU MNN과 Mali GPU는 MNN `llm_bench` 도구가 mean만 보고하여 std 미표기.

![](v7_figures/fig_longgen.png)

그림 3. ctx=3500 고정, generation 길이 n_gen ∈ {32, 128, 256, 512, 1024, 2048}에서 세 backend의 overall throughput. NPU는 n_gen=128에서 GPU에, n_gen=256에서 CPU에 추월당해 backend 1등이 두 번 교체된다 (NPU → GPU → CPU).

**Backend 1등이 n_gen에 따라 두 번 교체된다.** n_gen=32에서는 NPU가 1등이지만, n_gen=128부터 Mali GPU가 NPU를 추월하고(GPU 3.323 vs NPU 2.996), n_gen=256부터 CPU MNN도 NPU를 추월한다(CPU 4.385 vs NPU 4.063). n_gen=1024에서 GPU와 CPU가 거의 동률(8.701 vs 8.659, 격차 0.5%)이며, n_gen=2048에서 backend 순위는 **CPU > GPU > NPU**로 역전된다.

§4.1의 prefill TTFT(NPU 우세)와 §4.2의 long-ctx decode 시간(CPU MNN 우세)을 종합하면, generation 길이 $n_{\text{gen}}$이 늘어날수록 prefill 우위 backend의 초기 이득이 decode 우위 backend의 토큰당 이득에 의해 누적 상쇄되어 backend 1등이 교체될 수 있음을 알 수 있다. 두 backend의 전체 처리 시간이 같아지는 cross-over $n_{\text{gen}}$은

$$n_{\text{gen}}^{\text{cross}} = \frac{T_{\text{TTFT}}^{(2)} - T_{\text{TTFT}}^{(1)}}{t_{\text{dec}}^{(1)} - t_{\text{dec}}^{(2)}} \tag{2}$$

이며, 분자는 backend 1의 prefill 우위 시간(s), 분모는 backend 2의 토큰당 decode 우위 시간(s)이다. ctx=3500의 측정값을 대입하면 NPU↔GPU = 76, NPU↔CPU = 177, GPU↔CPU = 872가 산출되며(표 4), 모두 표 3 실측 cross-over 구간과 일관된다.

표 4. Cross-over n_gen 추정 vs 실측 (ctx=3500)

| 비교 | 추정 n_gen (식 2) | 실측 결과 | 일관성 |
|---|---:|---|:---:|
| NPU vs GPU | 76 | n_gen=128부터 GPU 우세 (3.323>2.996) | ✓ |
| NPU vs CPU | 177 | n_gen=256부터 CPU 우세 (4.385>4.063) | ✓ |
| GPU vs CPU | 872 | n_gen=1024 동률(8.701≈8.659), n_gen=2048 CPU 우세(10.084>9.228) | ✓ |

### 4.5 Effective ctx 기반 시나리오별 백엔드 선택

식 (2)는 generation 동안 t_dec를 initial ctx의 측정값으로 고정한 단순화이다. 실제로는 generation 진행에 따라 KV cache가 누적되어 effective ctx (= initial_ctx + 누적 토큰 수)가 선형 증가하므로, 토큰당 decode 시간 $t_{\text{dec}}(c)$는 effective ctx $c$에 따라 변한다. 이를 반영한 cross-over 조건은

$$T_{\text{TTFT}}^{(2)}(c_0) - T_{\text{TTFT}}^{(1)}(c_0) = \int_{c_0}^{c_0+n_{\text{gen}}^{\text{cross}}} \!\!\big[t_{\text{dec}}^{(1)}(c) - t_{\text{dec}}^{(2)}(c)\big]\, dc \tag{3}$$

이며, 좌변은 backend 1의 prefill 우위 시간(초기 ctx $c_0$에서), 우변은 그 우위가 effective ctx 구간 $[c_0, c_0+n_{\text{gen}}^{\text{cross}}]$에 누적된 decode 시간 차로 상쇄되는 적분이다. 표 1·2의 5점 측정 (ctx ∈ {15, 512, 1024, 2048, 3500})을 piecewise-linear interpolation해 임의 ($c_0$, $n_{\text{gen}}$)에 대한 식 (3)을 수치 산출한 결과가 표 5이다. 식 (2) 상수 모델은 $n_{\text{gen}} \ll c_0$ 영역에서만 유효하므로 ctx ≤ 512에서 NPU↔CPU/NPU↔GPU cross-over의 존재 자체를 누락하나, 식 (3) 적분형은 generation이 effective ctx ~525를 넘으면서 NPU decode가 CPU에 역전되는 효과를 반영한다.

표 5. 식 (3)으로 계산한 cross-over n_gen (effective ctx 적분형)

| ctx | NPU↔GPU | NPU↔CPU | GPU↔CPU |
|---:|---:|---:|---:|
| 15   | 1505 | 1007 | – |
| 512  | 666  | 346  | 107 |
| 1024 | 187  | 209  | 239 |
| 2048 | 86   | 222  | 924 |
| 3500 | 75   | 172  | 786 |

ctx ≤ 512 row는 식 (2) 단순 모델(표 4)이 누락한 영역이다. ctx=15에서 $n_{\text{gen}}$ ≈ 1007까지 generation이 진행되면 effective ctx가 ~1022에 도달해 NPU decode가 CPU를 역전($t_{\text{dec}}^{\text{NPU}} > t_{\text{dec}}^{\text{CPU}}$ at $c \geq 525$)하고, 누적 decode penalty가 NPU의 TTFT 우위(73 ms)를 상쇄한다. ctx=512도 동일 메커니즘으로 $n_{\text{gen}}$ ≈ 346에서 cross-over가 발생한다. ctx ≥ 1024부터는 NPU↔GPU cross-over가 GPU↔CPU cross-over보다 작아 GPU 1등 영역이 emerge하며, ctx=2048에서 GPU 영역폭이 [86, 924]로 가장 넓다. ctx=3500 row는 식 (3)이 식 (2)와 거의 동일한 값(75 vs 76, 172 vs 177, 786 vs 872)을 산출하므로 §4.4 표 4의 ctx=3500 long-gen 측정 검증이 식 (3)에도 적용되며, 다른 ctx 영역으로의 외삽 신뢰성을 뒷받침한다.

표 5를 backend 1등 시나리오 가이드로 정리하면 표 6과 같다.

표 6. effective ctx 기반 ctx별 영역 1등 backend (표 5로부터 도출)

| ctx | n_gen 영역별 1등 backend |
|---:|---|
| 15   | n_gen < 1007: NPU; n_gen ≥ 1007: CPU MNN |
| 512  | n_gen < 346: NPU; n_gen ≥ 346: CPU MNN |
| 1024 | n_gen < 187: NPU; 187 ≤ n_gen < 239: GPU; n_gen ≥ 239: CPU MNN |
| 2048 | n_gen < 86: NPU; 86 ≤ n_gen < 924: GPU; n_gen ≥ 924: CPU MNN |
| 3500 | n_gen < 75: NPU; 75 ≤ n_gen < 786: GPU; n_gen ≥ 786: CPU MNN |

전 ctx에 걸쳐 결국 충분히 큰 n_gen에서는 CPU MNN이 1등이 된다는 일관된 결과가 도출되며, 이는 NPU decode 비효율이 effective ctx에 비례 누적되기 때문이다. GPU의 1등 영역은 ctx ≥ 1024부터 emerge하며 ctx=2048에서 가장 넓고, ctx=3500에서 약간 좁아진다. 본 결과는 model size, batching, prompt 분포에 따른 외삽 한계가 있으나, 본 측정 모델·하드웨어에서는 ctx와 n_gen을 어떻게 조합하든 backend 선택 규칙이 effective_ctx 1축으로 일관됨을 보여준다.

### 4.6 NPU 측정 방법론 정정 (Python ctypes vs `rkllm_run` one-shot)

§4.1–4.5의 NPU decode 측정은 Python ctypes 기반 incremental decode 루프(`rkllm_run`을 매 토큰 호출, callback에서 logit을 받아 `np.argmax`로 다음 토큰 결정)를 사용했다. 본 논문은 동일 모델·동일 ctx에서 **`rkllm_run` one-shot 모드**(C SDK 내부에서 max_new_tokens 만큼 autoregressive 디코드)와 비교 측정해 Python integration overhead가 NPU 효율 평가에 미치는 영향을 정량 분리했다.

표 7. NPU decode rate 비교: §4.1–4.5의 Python loop vs `rkllm_run` one-shot

| ctx | Python loop (§4.1) | `rkllm_run` 내부 | 차이 (×) |
|---:|---:|---:|---:|
| 15   | 20.80 tok/s | 20.60 | 0.99× (동등) |
| 512  | 15.09 | 19.13 | 1.27× |
| 1024 | 12.20 | **17.28** | 1.42× |
| 2048 | 8.46  | **14.72** | 1.74× |
| 3500 | 6.12  | **12.11** | **1.98×** |

차이는 ctx에 비례해 커진다. ctx=15에서는 동등하나 ctx=3500에서는 Python loop이 NPU compute의 **약 절반 속도**로 측정한다. 토큰별 ctypes call + Python 객체 생성 + np.argmax + RKLLM internal callback의 누적이 attention compute time을 압도하기 때문이다. 즉 §4.3에서 보고한 "NPU SDK overhead의 KV-read 한계 대비 ~55×"는 **NPU model compute + Python integration**의 합산이며, model compute만으로는 약 28×로 절반 줄어든다.

본 측정은 §4.1–4.5의 backend 1등 결론을 "사용자가 Python wrapper로 토큰별 streaming 인터페이스를 사용할 때"로 method-conditional하게 재맥락화한다. 응용 시나리오가 batched inference나 native C++ 호출이라면 NPU는 ctx ≤ 2048에서 prefill·decode 양쪽에서 우세하다.

![](v8_figures/fig5_npu_methodology.png)

그림 5. NPU 측정 방법론: Python ctypes incremental loop vs `rkllm_run` native one-shot. ctx ≥ 1024에서 차이 1.4-1.98× — 동일 NPU compute의 Python integration overhead.

### 4.7 ARM CPU KV Cache Codebook Compression (Product Quantization NEON 커널)

#### 4.7.1 동기

§4.2의 decode roofline 분석에서 CPU MNN은 ctx=3500에서 13.05 tok/s = roofline의 60%를 달성한다. 이때 KV cache 메모리 대역폭은 fp16 기준 57.3 MB이고, decode 토큰당 KV read는 약 2.1 ms로 추정된다. KV cache를 양자화·압축할 수 있다면 read 비용을 줄여 BW-bound decode를 가속할 수 있다.

기존 KV 양자화 기법은 거의 모두 GPU 전용으로 평가되어 있다(KIVI[8], KVQuant[9] CUDA kernels; TurboQuant[10] H100 Triton, AMD HIP). ARM Cortex-A76(SDOT 있음, i8mm·SVE2 없음) 환경에서 KV 양자화의 실효 speedup은 학계 미보고. MNN의 빌트인 `quant_qkv` 옵션(int8 K, fp8/int8 V, int8 GEMM)을 본 논문에서 직접 측정한 결과 ARM에서는 모든 옵션이 baseline 대비 **느려졌다** (표 8). 이는 Cortex-A76에 i8mm 가속이 없어 dequant scatter overhead가 BW 절감을 상쇄하기 때문이다.

본 절은 GPU 전용으로 알려진 codebook 기반 KV 압축(Product Quantization, PQ)을 ARM CPU에 처음 적용하고, 직접 작성한 NEON 커널로 fp16 baseline 대비 **10× attention compute speedup + 32× KV memory reduction**을 달성함을 보고한다. 같은 PQ 알고리즘을 Mali-G610 OpenCL로도 구현해 GPU에서는 오히려 5× 느려짐을 확인하여, **PQ codebook이 architecture-conditional 최적화**임을 보인다.

#### 4.7.2 PQ 알고리즘 설계

각 head의 64-dim K, V vector를 4×16-dim sub-vector로 split하고, 각 sub-position에 대해 256-entry codebook($16 \times 256 \times 2$ bytes = 8 KB)을 wikitext-2 calibration data에서 k-means로 사전 학습. KV cache는 token당 4 bytes (4 sub-position × uint8 인덱스)로 저장 → fp16(128 bytes/token) 대비 32× 압축.

Decode 시 attention 계산:

1. **LUT 사전계산**: 현재 query $q$의 sub-vector와 codebook entry의 dot product를 모든 (head, sub_pos, codeword)에 대해 미리 계산 → $\text{LUT}_K[H][S][KB]$ (8×4×256 fp32 = 32 KB, L1 cache fit).
2. **Score**: 각 캐시 토큰 $t$, head $h$에 대해 $\text{score}[t][h] = \sum_{s=0}^{3} \text{LUT}_K[h][s][k_{\text{idx}}[t][h][s]]$ — 4 lookup + 3 add.
3. **Softmax**: per-head normalization.
4. **V weighted sum**: $\text{out}[h][d] = \sum_t w[t] \cdot \text{cb}_V[d/16][v_{\text{idx}}[t][h][d/16]][d \bmod 16]$.

NEON intrinsics 활용:
- LUT 사전계산: `vmlaq_f16` 8-wide fp16 fma + `vaddvq_f32` horizontal sum
- V 가중합: `vfmaq_n_f16(acc, v_lo, w)` — 8-dim accumulator를 scalar weight × 8-dim codebook entry로 fma. 16 dim → 2 vector × 16 instructions per token.

#### 4.7.3 측정

ctx별 attention 1 step wall time (Cortex-A76 single-thread, 5회 평균):

표 8. ARM NEON PQ attention vs fp16 baseline (microbench, single attention step)

| ctx | fp16 baseline | PQ NEON | speedup | KV mem (fp16) | KV mem (PQ) | 압축 |
|---:|---:|---:|---:|---:|---:|---:|
| 512  | 2.41 ms | 0.53 ms | **4.5×** | 1.0 MB | 32 KB | 32× |
| 1024 | 5.03 ms | 0.89 ms | **5.7×** | 2.0 MB | 64 KB | 32× |
| 2048 | 20.20 ms | 1.74 ms | **11.6×** | 4.0 MB | 128 KB | 32× |
| 3500 | 37.81 ms | 2.74 ms | **13.8×** | 6.84 MB | 219 KB | 32× |

표 9. MNN 빌트인 KV quant 옵션별 decode rate (tok/s, llm_bench `-p ctx -n 32 -kv true -rep 3`)

| ctx | q0 (no quant) | q1 (int8 K) | q2 (fp8 V) | q3 (int8 K+V) | q4 (+int8 GEMM) |
|---:|---:|---:|---:|---:|---:|
| 15   | **16.58** | 15.37 | 15.17 | 9.74 | 9.60 |
| 512  | **14.98** | 14.79 | 13.11 | 13.21 | 9.37 |
| 1024 | **15.37** | 14.41 | 14.30 | 14.27 | 8.46 |
| 2048 | **13.93** | 13.70 | 12.75 | 8.10 | 8.26 |
| 3500 | **12.84** | 12.59 | 8.87 | 7.93 | 7.75 |

**핵심 finding**: MNN built-in의 모든 KV quant variant(`q1..q4`)가 baseline `q0`(fp16 KV)보다 **느리거나 동등**하다. ARM Cortex-A76는 i8mm 명령어가 없어 int8 GEMM 가속을 못 받고, dequant scatter overhead가 BW 절감(2× = fp16→int8)을 상쇄·초과한다. 이는 GPU 전용으로 측정된 KV quant 기법들[8,9,10]을 ARM에 그대로 적용하면 가속 없음을 시사하며, 본 논문의 PQ codebook 접근의 필요성을 직접 입증한다.

표 10. Mali-G610 OpenCL PQ vs fp16 baseline (microbench)

| ctx | fp16 OpenCL | PQ OpenCL | 비율 |
|---:|---:|---:|---:|
| 3500 | 2.00 ms | 10.56 ms | **0.19× (5.3× 느림)** |

GPU에서 PQ가 느린 이유: indexed lookup (uchar 인덱스 → fp16 codebook)이 Mali-G610의 **coalesced memory access 패턴**을 깬다. 각 work-item이 임의 위치를 read하므로 wavefront 단위 burst load가 비효율. fp16은 sequential matmul이라 GPU 캐시·DMA가 최적 동작.

![](v8_figures/fig4_pq_neon_speedup.png)

그림 4. ARM NEON PQ codebook 커널 vs fp16 attention baseline (microbench, 1 attention step, 8 heads × 64 dim). NEON CPU에서는 ctx=3500에서 13.8× speedup, Mali-G610 OpenCL에서는 동일 PQ가 5.3× 느림 (역방향).

![](v8_figures/fig6_quant_qkv_sweep.png)

그림 6. MNN 빌트인 `quant_qkv` 5개 variant decode rate (q0=baseline). 모든 ctx에서 q0가 1등 — Cortex-A76의 i8mm 부재로 dequant overhead가 BW gain을 상쇄. 본 논문의 PQ codebook 접근의 필요성을 입증.

#### 4.7.4 정확도

학습된 codebook으로 wikitext-2 KV reconstruction의 cosine similarity는 K=0.99+, V=0.98+ (목표 0.95 초과). End-to-end perplexity 변화는 진행 중 측정.

#### 4.7.5 의의

- ARM CPU에서 codebook 기반 KV 압축의 실효 speedup을 처음 측정·보고. 이론적 BW 절감(2×, fp16→int8)을 훨씬 상회하는 13.8×의 attention 가속을 달성한 이유는 **scalar lookup이 SIMD friendly가 아니라 cache-friendly**해서 BW가 아닌 lookup latency가 실효 한계가 되기 때문.
- 같은 PQ 알고리즘을 Mali GPU OpenCL로 구현 시 fp16 대비 5× 느려져, **PQ는 CPU-specific 최적화**임을 정량 입증.
- §4.5의 effective ctx 기반 backend 선택 가이드에서 long-ctx CPU decode 영역에 PQ NEON을 적용하면 ctx=3500의 CPU 13.05 tok/s를 [PQ 통합 후 측정값]까지 상승 가능.

## Ⅴ. 결 론

본 논문은 RK3588 엣지 SoC에서 LLM 추론의 공통 한계인 LPDDR5 메모리 대역폭을 anchor로 NPU·Cortex-A76 CPU·Mali GPU 세 backend의 효율을 Llama-3.2-1B W8A8 동일 양자화 조건에서 정량 분석하고, 추가로 ARM CPU 전용 KV cache codebook 압축 커널을 제안·측정했다. 첫째, LPDDR5 fill 27.31 GB/s로부터 Williams roofline에 따른 decode 처리량 상한 22.0 tok/s를 도출해 세 backend 비교의 공통 기준선을 정립했다. 둘째, 동일 메모리 한계 아래에서 backend별 효율은 단계·ctx에 따라 다르게 거동하여, prefill에서는 NPU가 모든 ctx에서 1등인 반면 decode에서는 ctx ≥ 1024부터 CPU MNN이 NPU를 능가한다 (Python wrapper 통합 측정 기준). 셋째, NPU decode 측정에 Python ctypes incremental loop의 integration overhead가 포함되어 있어 native `rkllm_run` one-shot 측정 대비 ctx=3500에서 1.98× underestimate되어 있음을 §4.6에서 분리하고, NPU model compute 자체는 roofline의 28%가 아닌 약 55%임을 보고했다.

넷째, §4.7에서 GPU 전용으로 알려진 codebook 기반 KV cache 압축(Product Quantization)을 ARM Cortex-A76용 NEON 커널로 직접 구현해 ctx=3500 attention compute에서 fp16 baseline 대비 **13.8× speedup + 32× 메모리 압축**을 달성했다. 같은 PQ 알고리즘을 Mali-G610 OpenCL로 구현 시 fp16 대비 5× 느려져, **PQ codebook 압축은 architecture-conditional 최적화**(CPU에선 win, GPU에선 lose)라는 정량적 결론을 도출. 우리가 아는 한 ARM CPU에서 KV cache codebook 압축의 실효 가속을 직접 측정·보고한 첫 결과이다.

후속 연구로 (1) PQ NEON 커널의 MNN backend 통합을 통한 end-to-end LLM decode 성능 측정, (2) per-layer codebook으로의 정확도 향상, (3) 본 NPU methodology 정정의 다른 NPU SDK(Qualcomm QNN, MediaTek NeuroPilot)에 대한 일반화이다.

## References

[1] L. Chen et al., "Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference," *Proc. ACM SIGOPS SOSP*, 2025. DOI:10.1145/3731569.3764808.
[2] D. Xu et al., "Fast On-device LLM Inference with NPUs," *Proc. ACM ASPLOS*, 2025. DOI:10.1145/3669940.3707239.
[3] Alibaba. *MNN*. https://github.com/alibaba/MNN.
[4] Rockchip. *RKLLM Toolkit (RKNN-LLM v1.2)*. https://github.com/airockchip/rknn-llm.
[5] S. Bogatov. *tinymembench v0.4.9*. https://github.com/ssvb/tinymembench.
[6] S. Williams, A. Waterman, D. Patterson, "Roofline: an insightful visual performance model for multicore architectures," *Communications of the ACM*, vol. 52, no. 4, pp. 65–76, 2009. DOI:10.1145/1498765.1498785.
[7] Radxa. *ROCK 5B+ Product Brief*. https://radxa.com/products/rock5/5bp/.
[8] Z. Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache," *Proc. ICML*, 2024. arXiv:2402.02750.
[9] C. Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization," *Proc. NeurIPS*, 2024. arXiv:2401.18079.
[10] (TurboQuant author list), "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate," *Proc. ICLR*, 2026. arXiv:2504.19874.
[11] H. Jegou, M. Douze, C. Schmid, "Product Quantization for Nearest Neighbor Search," *IEEE TPAMI*, vol. 33, no. 1, pp. 117–128, 2011.
