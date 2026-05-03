# RK3588 엣지 SoC에서 LLM 추론의 메모리 대역폭 한계와 백엔드별 효율 분석

손현호, 윤수연*
국민대학교, *국민대학교
aa000098@kookmin.ac.kr, *1104py@kookmin.ac.kr

**Memory Bandwidth Limits and Per-Backend Efficiency Analysis of LLM Inference on RK3588 Edge SoC**
Hyun Ho Son, Soo Yeon Yoon*
Kookmin Univ., *Kookmin Univ.

---

## 요 약

엣지 SoC RK3588은 NPU, Cortex-A76 CPU, Mali-G610 GPU 세 가속기가 LPDDR5 메모리를 공유하므로, decode 단계의 memory-bound 특성상 메모리 대역폭이 LLM 추론 성능의 공통 한계로 작용한다. 본 논문은 LPDDR5 단일 방향 read 대역폭으로부터 Williams roofline 기반 decode 처리량 상한을 도출하고, Llama-3.2-1B W8A8 동일 양자화 조건에서 세 backend의 prefill·decode 효율을 다양한 context·generation length에 걸쳐 측정·비교한다. 분석 항목은 (1) roofline 대비 backend별 효율, (2) prefill·decode 두 단계에서 우세 backend의 분리 현상, (3) (ctx, n_gen) 조합에 따른 backend 1등 cross-over의 정량 모델링이다. 측정 결과는 동일 메모리를 공유함에도 backend별 효율이 ctx에 따라 다르게 거동함을 보이고, generation 동안 누적되는 effective ctx 1축으로 backend 선택 규칙이 정량 모델링됨을 보인다.

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

## Ⅴ. 결 론

본 논문은 RK3588 엣지 SoC에서 LLM 추론의 공통 한계인 LPDDR5 메모리 대역폭을 anchor로 NPU·Cortex-A76 CPU·Mali GPU 세 backend의 효율을 Llama-3.2-1B W8A8 동일 양자화 조건에서 정량 분석했다. 첫째, LPDDR5 fill 27.31 GB/s로부터 Williams roofline에 따른 decode 처리량 상한 22.0 tok/s를 도출해 세 backend 비교의 공통 기준선을 정립했다. 둘째, 동일 메모리 한계 아래에서 backend별 효율은 단계·ctx에 따라 다르게 거동하여, prefill에서는 NPU가 모든 ctx에서 1등인 반면 decode에서는 ctx ≥ 1024부터 CPU MNN이 NPU를 능가한다. NPU decode의 long-ctx 비효율(roofline의 28% @ ctx=3500)은 메모리 대역폭이 아닌 SDK level attention 처리 비용(KV-read 한계의 약 55×)에서 발생함을 incremental decode 측정으로 분리하고, 후보 메커니즘 4가지(attention kernel fusion 부재, NPU SRAM ≪ KV로 인한 DMA 빈도, kernel dispatch overhead, int8 dequant 분리)를 §4.3에 명시했다. 셋째, generation 동안 누적되는 effective ctx (= initial_ctx + 누적 토큰 수)를 1축으로 식 (3) 적분형 cross-over 모델을 도출해 (initial_ctx, n_gen) 모든 조합에서 backend 1등 region을 정량 예측했으며, §4.4의 ctx=3500 long-generation 실측 cross-over와 일관됨을 확인했다.

본 결과는 우리가 아는 한 RK3588에서 동일 양자화로 세 backend를 LPDDR5 roofline 위에서 직접 비교한 첫 측정이며, "NPU가 LLM 가속에 절대 우세"라는 일반 가정에 대한 정량적 반례를 제시한다. 실용적 함의로, KV-read 메모리 한계 대비 약 55×의 SDK level overhead는 RKLLM attention kernel 개선만으로 회복 가능한 잠재 이득이 매우 큼을 시사하며, 표 5·6의 effective ctx 기반 backend 선택 가이드는 SDK 개선과 무관하게 즉시 적용 가능하다. 후속 연구는 (1) §4.3 후보 메커니즘의 NPU profiler/DMA trace 기반 정량 분리, (2) 더 큰 모델(3B, 7B)과 더 긴 context(8K, 16K)에서의 일반화이다.

## References

[1] L. Chen et al., "Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference," *Proc. ACM SIGOPS SOSP*, 2025. DOI:10.1145/3731569.3764808.
[2] D. Xu et al., "Fast On-device LLM Inference with NPUs," *Proc. ACM ASPLOS*, 2025. DOI:10.1145/3669940.3707239.
[3] Alibaba. *MNN*. https://github.com/alibaba/MNN.
[4] Rockchip. *RKLLM Toolkit (RKNN-LLM v1.2)*. https://github.com/airockchip/rknn-llm.
[5] S. Bogatov. *tinymembench v0.4.9*. https://github.com/ssvb/tinymembench.
[6] S. Williams, A. Waterman, D. Patterson, "Roofline: an insightful visual performance model for multicore architectures," *Communications of the ACM*, vol. 52, no. 4, pp. 65–76, 2009. DOI:10.1145/1498765.1498785.
[7] Radxa. *ROCK 5B+ Product Brief*. https://radxa.com/products/rock5/5bp/.
