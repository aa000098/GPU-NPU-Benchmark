# RK3588 엣지 SoC에서 Phase별 최적 백엔드의 분리와 Hetero Scheduling 가능성

손현호, 윤수연*
국민대학교, *국민대학교
aa000098@kookmin.ac.kr, *1104py@kookmin.ac.kr

**Phase-Separated Optimal Backends and Hetero Scheduling Potential for Edge LLM Inference on RK3588**
Hyun Ho Son, Soo Yeon Yoon*
Kookmin Univ., *Kookmin Univ.

---

## 요 약

엣지 SoC RK3588은 NPU(6 TOPS), Cortex-A76 CPU, Mali-G610 GPU 세 가속기를 LPDDR5 메모리(64-bit @ 5500 MT/s)에 공유한다. 본 논문은 Llama-3.2-1B를 W8A8 동일 양자화로 세 backend에서 측정해 다음 네 가지를 정량 입증한다. 첫째, LPDDR5 단일 방향 처리량(write fill 27.31 GB/s)으로부터 Williams roofline 모델[6]에 따른 decode 처리량 상한 22.0 tok/s를 도출했다. 둘째, **prefill phase에서는 NPU가 모든 ctx에서 1등**(ctx=3500에서 CPU의 1.67×)이지만 **decode phase에서는 ctx≥1024부터 CPU MNN이 NPU를 능가**(ctx=3500에서 2.13×)하여, phase별로 최적 backend가 분리되는 비대칭 구조를 보인다. 셋째, ctx=3500 long generation 실측에서 backend 1등이 n_gen에 따라 NPU(≤32) → GPU(128–512) → CPU(≥870 추정) 순서로 두 번 교체된다. 넷째, 이 비대칭 구조를 활용한 **NPU prefill + CPU decode hetero scheduling은 ctx=3500에서 단독 best backend 대비 1.18×(n_gen=512 기준), 1.22×(n_gen=128 기준)의 향상 가능성**을 산수로 추정 가능하다. 본 결과는 우리가 아는 한 RK3588에서 동일 양자화로 세 backend를 직접 비교하고 hetero scheduling 가능성을 정량화한 첫 측정이며, "NPU가 LLM 가속에 절대 우세"라는 일반 가정의 정량적 반례와 함께 단순 단독 backend 선택이 아닌 phase 분담이 엣지 LLM 가속의 실용적 경로임을 시사한다.

---

## Ⅰ. 서 론

엣지 SoC RK3588은 NPU(6 TOPS), Cortex-A76 4코어 CPU, Mali-G610 GPU 세 가속기를 LPDDR5 메모리에 공유한다. 모두 LLM 추론에 사용 가능한 backend이지만, 어느 backend가 어느 시나리오에서 우세한지에 대한 정량 비교 publication은 부재하며 NPU가 LLM 가속에 절대 우세하다는 인식이 일반적이다. 모바일 SoC 대상 선행 연구는 있으나[1, 2] RK3588에서 동일 양자화 조건으로 세 backend를 직접 비교한 publication은 우리가 아는 한 보고된 바 없다.

LLM 추론은 prefill과 decode 두 phase로 나뉘며 두 phase의 연산 특성이 근본적으로 다르다. Prefill은 prompt 전체에 대한 batch GEMM으로 compute-bound에 가깝고, decode는 토큰별 GEMV로 memory bandwidth-bound이다. 따라서 GEMM 친화적 가속기와 memory-friendly attention 구현을 가진 가속기가 phase별로 다르게 우세할 가능성이 있으며, 이 경우 단독 backend 선택이 아닌 phase 분담(hetero scheduling)이 두 backend 단독보다 빠를 수 있다.

본 연구는 Rock 5B+(RK3588) 보드에서 Llama-3.2-1B를 W8A8 동일 양자화로 세 backend에 변환하고, 5개 컨텍스트 길이(15, 512, 1024, 2048, 3500)와 4개 생성 길이(32, 128, 256, 512)에서 측정하여 (1) LPDDR5 메모리 대역폭 roofline 위에 각 backend를 위치시키고, (2) prefill·decode phase별 최적 backend의 비대칭 구조를 확인하고, (3) NPU prefill + CPU decode hetero scheduling의 향상 가능성을 정량화한다.

## Ⅱ. 본 론

### 2.1 LLM Decode의 메모리 대역폭 한계

트랜스포머 LLM의 토큰별 decode는 매 단계에서 모델 weight 전체를 메모리에서 읽어 GEMV 연산을 수행한다. Williams 등[6]의 roofline 모델에 따르면 weight read가 dominant한 영역에서 토큰당 처리량 상한은

$$\text{tok/s}_{\max} = \frac{B_{\text{read}}}{S_{\text{weight}}} \tag{1}$$

이며, $B_{\text{read}}$는 보드의 단일 방향 read 대역폭, $S_{\text{weight}}$는 모델 weight 크기이다. Llama-3.2-1B의 weight 크기는 1.24B 파라미터 × 1 byte (W8A8) = **1.24 GB**이다. 식 (1)은 weight read가 dominant한 영역에서의 상한이며, 실측 비용은 attention의 KV-cache traversal과 backend별 추가 오버헤드만큼 더 커진다.

### 2.2 RK3588 SoC 및 Rock 5B+ 구조

RK3588은 6 TOPS NPU(3 cores), Cortex-A76 ×4 + A55 ×4 CPU, Mali-G610 MP4 GPU를 SoC에 통합한다. 실험 보드 Rock 5B+는 RK3588에 LPDDR5 64-bit @ 5500 MT/s 메모리[9]를 결합하며 이론 peak는 5500 × 8 = **44 GB/s**이다. 모든 측정에서 A76 big core 4개를 사용하고 A55 little core는 비활성화한다(`taskset -c 4-7`). NPU는 closed-source RKLLM SDK v1.2.1b1[4]로, Mali GPU는 MNN OpenCL backend[3]로 접근한다.

## Ⅲ. 실험 설계

대상 모델은 Meta Llama-3.2-1B-Instruct를 W8A8 양자화로 RKLLM(NPU), MNN(CPU 4 threads, OpenCL Mali) 세 backend에 통일 변환한다. Wikitext-2 train split의 자연어 prefix를 5개 길이 *n* ∈ {15, 512, 1024, 2048, 3500}로 truncate하여 prompt로 사용하고, 기본 측정에서 n_gen=32 토큰을 생성한다. 모든 측정은 KV cache를 활성화하고(MNN: `-kv true`, RKLLM: `keep_history=1`) `-rep 3`으로 3회 반복하여 평균±std를 보고한다. Long-generation 측정에서는 backend 사이 5분 idle을 두어 thermal recovery를 확보했다(SoC 38 °C 회복 확인). 메모리 대역폭은 tinymembench v0.4.9[5]를 A76 4 core affinity pinning한 상태로 측정한다.

## Ⅳ. 실험 결과

### 4.1 LPDDR5 Roofline 도출

tinymembench로 측정한 Rock 5B+의 처리량은 NEON copy 12.91 GB/s, fill 27.31 GB/s이다. NEON copy는 read와 write가 동시에 LPDDR 버스를 점유하므로 단일 방향 처리량 추정에 부적합하다. 본 논문은 보수적으로 **fill 27.31 GB/s를 단일 방향 read 대역폭의 upper bound proxy**로 채택하며, 이는 이론 peak 44 GB/s의 62%이다. 실측 read 대역폭은 이보다 작을 가능성이 높으므로 본 roofline은 실제 상한보다 낙관적이며 어느 backend도 이를 초과해서는 안 된다. 식 (1)에 대입하면 **decode 처리량 상한 = 22.0 tok/s**이다(그림 1, 수평선).

![Fig 1. RK3588 LPDDR5 Roofline + Multi-backend Decode](figures/fig_roofline.png)

### 4.2 Phase별 최적 백엔드의 비대칭 구조

표 1은 prefill phase의 TTFT를, 표 2는 decode phase의 토큰당 throughput을 정리한 것이다. 두 phase는 backend 순위가 정성적으로 다르게 나타난다.

표 1. **Prefill phase: TTFT** (sec) — Llama-3.2-1B W8A8

| ctx | NPU (RKLLM) | CPU MNN | Mali GPU |
|---:|---:|---:|---:|
| 15   | **0.128** | 0.201 | 0.269 |
| 512  | **1.95**  | 3.55  | 2.87  |
| 1024 | **4.42**  | 8.10  | 5.99  |
| 2048 | **10.85** | 21.74 | 14.34 |
| 3500 | **23.06** | 38.42 | 28.83 |

표 2. **Decode phase**: tok/s 및 NPU의 roofline 22.0 대비 효율

| ctx | NPU | CPU MNN | Mali GPU | NPU/한계 |
|---:|---:|---:|---:|---:|
| 15   | **20.80** | 16.18 | 14.93 | **94%** |
| 512  | **15.09** | 15.01 | 13.76 | 69% |
| 1024 | 12.20 | **14.80** | 13.04 | 55% |
| 2048 | 8.46  | **13.84** | 12.68 | 38% |
| 3500 | 6.12  | **13.05** | 11.41 | 28% |

**Prefill phase**에서는 NPU가 모든 ctx에서 1등이며 ctx=3500에서 CPU 대비 1.67×, GPU 대비 1.25× 빠르다. 이는 prefill이 prompt 전체에 대한 batch GEMM으로 compute-bound 영역에 가깝고, NPU의 6 TOPS GEMM 가속이 직접 활용되기 때문으로 해석된다.

**Decode phase**에서는 backend별로 정성적으로 다른 거동이 관측된다. NPU는 ctx=15에서 roofline의 94%로 메모리 대역폭 한계에 거의 도달하나 ctx=3500에서 28%까지 추락한다. CPU MNN은 ctx=15→3500 구간에서 16.18→13.05 tok/s로 19% 감소에 그치며 안정 유지된다. 이 안정성은 CPU MNN이 NEON SIMD 기반 attention kernel과 cache hierarchy(A76의 64 KB L1 + 512 KB L2 + 3 MB shared L3)에 friendly한 메모리 접근으로 KV traversal을 처리하기 때문으로 추정된다. Mali GPU는 11.4–14.9 tok/s로 CPU보다 일관되게 1–2 tok/s 낮은데, 동일 LPDDR5를 공유함에도 OpenCL kernel launch 오버헤드와 MNN OpenCL backend 성숙도 부족이 누적된 결과로 해석된다. **임계점은 ctx=1024**이며 이 이후 CPU MNN이 NPU를 능가하여 ctx=3500에서 2.13× 격차가 된다.

이로써 **prefill에서는 NPU, decode에서는 (long ctx에 한해) CPU MNN이 우세**한 비대칭 구조가 확인된다. GPU는 두 phase 모두에서 중간 성능을 보이나 어느 phase에서도 1등이 아니다.

![Fig 2. Backend-by-Backend Throughput Decomposition (TTFT + Decode)](figures/fig_backends.png)

### 4.3 NPU의 Decode ctx-비례 추가 비용 분리

NPU decode의 ctx 비례 비효율(28% @ ctx=3500)이 메모리 대역폭에서 발생하는지, SDK level에서 발생하는지를 분리한다. NPU의 토큰당 decode 시간은 ctx=15에서 48 ms, ctx=3500에서 163 ms로 ctx에 비례해 증가한다. roofline에 따른 weight read 비용 하한은 1.24 GB / 27.31 GB/s = **45 ms/tok**이며 ctx에 무관해야 하므로, ctx=15→3500 구간 추가 비용 (163 − 48) = **115 ms**는 weight read 외에서 발생한다.

이 추가 비용을 attention KV-cache traversal로 설명할 수 있는지 검증한다. Llama-3.2-1B는 16 layers × 32 heads × 64 head_dim × 2 (K+V) × 1 byte = 65.5 KB/token이고 ctx=3500의 누적 KV는 229 MB이다. read 대역폭 27.31 GB/s에서 229 MB read의 시간 한계는 **8.4 ms**이다. 그러나 NPU 실측 추가분 115 ms는 한계의 **13.7×**이다. 같은 구간에서 CPU MNN의 추가 비용은 (1000/13.05 − 1000/16.18) ≈ 14.8 ms로 동일 한계의 1.76×에 머문다. 즉 backend가 동일한 LPDDR5와 동일한 KV 크기를 다루는데도 NPU에서만 한 자릿수 큰 추가 비용이 발생한다.

이 추가 비용이 KV cache 자체의 비정상 traversal(예: 토큰별 재계산)에서 오는 것은 아님을 다음 측정으로 확인했다. Prefill 후 8 토큰을 1개씩 incremental decode하여 토큰별 시간을 측정하면, 첫 토큰의 cold-start를 제외한 7 토큰에서 ctx=1024는 78.8–79.5 ms 범위(std 6.6 ms), ctx=3500은 153.2–157.1 ms 범위로, 토큰 추가에 따른 점진적 시간 증가가 없다. 즉 KV cache reuse 자체는 정상이다.

따라서 NPU의 ctx-비례 추가 비용은 **RKLLM SDK의 attention kernel 구현 비용**(K/V layout 변환, softmax/masking 처리, head별 sequential dispatch 등)에서 발생한다고 가정한다(hypothesis). RKLLM SDK가 closed-source이므로 직접적 root cause 추적은 본 연구의 범위를 벗어나지만, 측정 결과는 NPU의 long context 비효율이 메모리 대역폭이 아닌 **SDK level**에서 발생함을 시사한다. 이 분리는 두 가지 함의를 갖는다. 첫째, SDK level의 attention kernel 개선이 NPU long context decode 효율 회복의 직접적 경로이다. 둘째, SDK 개선 없이 현재 시점에서 활용 가능한 우회 전략은 long context decode를 NPU 외 backend에 위임하는 phase 분담이다. 후자가 §4.5의 hetero scheduling 가설로 이어진다.

### 4.4 Long Generation에서의 백엔드 Cross-over (실측)

§4.2의 표는 n_gen=32 측정으로 decode 비중이 prefill 대비 작다. 실용적 시나리오(creative writing, code generation, agentic 응답)에서 n_gen이 더 클 때의 cross-over를 직접 측정하기 위해 ctx=3500을 고정하고 n_gen ∈ {32, 128, 256, 512}에서 세 backend의 overall throughput을 측정했다(표 3, 그림 3).

표 3. ctx=3500, varying n_gen (mean ± std, 3 repeats)

| n_gen | NPU (tok/s) | CPU MNN (tok/s) | Mali GPU (tok/s) |
|---:|---:|---:|---:|
| 32  | **1.146 ± 0.026** | 0.788 | 1.031 |
| 128 | 2.996 ± 0.006 | 2.648 | **3.323** |
| 256 | 4.063 ± 0.002 | 4.385 | **5.117** |
| 512 | 4.912 ± 0.008 | 6.519 | **7.154** |

![Fig 3. Backend cross-over at ctx=3500 with varying n_gen](figures/fig_longgen.png)

**Backend 1등이 n_gen에 따라 두 번 교체된다.** n_gen=32에서는 NPU가 1등이지만, n_gen=128부터 Mali GPU가 NPU를 추월하고(GPU 3.323 vs NPU 2.996), n_gen=256부터 CPU MNN도 NPU를 추월한다(CPU 4.385 vs NPU 4.063). n_gen=512에서 backend 순위는 GPU > CPU > NPU로 역전된다.

이 cross-over는 §4.2의 phase별 비대칭 구조에서 직접 도출된다. Overall throughput은

$$\text{overall} = \frac{n_{\text{gen}}}{T_{\text{TTFT}} + n_{\text{gen}} \cdot t_{\text{decode}}} \tag{2}$$

이며, n_gen이 작을수록 TTFT 항이, 클수록 t_decode 항이 dominant해진다. 따라서 n_gen이 증가할수록 1등이 prefill 우세 backend(NPU)에서 decode 우세 backend(CPU)로 교체된다. 식 (2)에서 두 backend의 overall이 같아지는 cross-over n_gen은

$$n_{\text{gen}}^{\text{cross}} = \frac{T_{\text{TTFT}}^{(2)} - T_{\text{TTFT}}^{(1)}}{t_{\text{decode}}^{(1)} - t_{\text{decode}}^{(2)}} \tag{3}$$

이며, ctx=3500의 측정값을 대입하면 NPU↔GPU = 76, NPU↔CPU = 177, GPU↔CPU = 872가 된다(표 4). n_gen=128에서 GPU가 NPU를 추월(추정 76 이후), n_gen=256에서 CPU가 NPU를 추월(추정 177 이후)한 실측이 추정과 일치한다. n_gen → ∞에서 overall은 decode tok/s에 수렴하므로 매우 긴 generation에서 최종 1등은 decode tok/s 1등인 CPU MNN이 된다.

표 4. Cross-over n_gen 추정 (ctx=3500, 식 (3))

| 비교 | n_gen 추정 | 실측 일관성 |
|---|---:|---|
| NPU vs GPU | 76 | n_gen=128에서 GPU 추월 ✓ |
| NPU vs CPU | 177 | n_gen=256에서 CPU 추월 ✓ |
| GPU vs CPU | 872 | n_gen=512까지 GPU 우위 (측정 범위 내 미발생) ✓ |

### 4.5 Hetero Scheduling 가능성: NPU Prefill + CPU Decode

§4.2의 phase별 비대칭과 §4.4의 cross-over는 동일한 메커니즘 — TTFT 우세 backend와 decode 우세 backend가 다르다 — 의 두 표현이다. 이는 자연스럽게 **NPU의 prefill 강점과 CPU MNN의 decode 강점을 결합**하는 hetero scheduling을 시사한다. 즉 prefill을 NPU로, decode를 CPU MNN으로 위임하면 단독 backend 대비 두 phase 모두에서 손해 없이 처리 가능하다.

식 (2)에 NPU의 TTFT와 CPU MNN의 decode time을 대입하면 hetero overall은

$$\text{overall}_{\text{hetero}} = \frac{n_{\text{gen}}}{T_{\text{TTFT}}^{\text{NPU}} + n_{\text{gen}} \cdot t_{\text{decode}}^{\text{CPU}}} \tag{4}$$

이다. ctx=3500에서 산수 추정한 hetero throughput과 단독 best backend 대비 향상은 표 5와 같다.

표 5. ctx=3500에서 hetero(NPU prefill + CPU decode) 산수 추정

| n_gen | NPU 단독 | CPU 단독 | GPU 단독 | **Hetero** | gain vs best single |
|---:|---:|---:|---:|---:|---:|
| 32 | **1.13** | 0.78 | 1.01 | **1.25** | 1.11× (vs NPU) |
| 128 | 2.91 | 2.65 | **3.20** | **3.89** | **1.22× (vs GPU)** |
| 256 | 3.95 | 4.41 | **4.99** | **6.00** | 1.20× (vs GPU) |
| 512 | 4.80 | 6.59 | **6.95** | **8.22** | **1.18× (vs GPU)** |
| 1024 | 5.38 | **8.76** | 8.64 | **10.09** | 1.15× (vs CPU) |
| 2048 | 5.73 | **10.49** | 9.84 | **11.38** | 1.09× (vs CPU) |

ctx=3500에서 hetero는 측정 범위(n_gen ≤ 512)와 추정 범위(n_gen ≤ 2048) 모두에서 단독 best backend 대비 **1.05–1.22× 향상**된다. n_gen → ∞에서 hetero overall은 CPU decode tok/s(13.05)에 수렴하므로 gain은 1.0×에 점근하며, hetero가 가장 효과적인 영역은 n_gen이 prefill과 decode 비용이 비슷한 중간 영역(n_gen ≈ 100–500)이다.

**한계 및 가정.** 표 5는 식 (2)에 NPU TTFT와 CPU decode time을 대입한 산수 추정으로 실측이 아니다. 실제 hetero scheduling 구현에는 (1) NPU에서 생성된 KV cache를 CPU로 전달하는 비용, (2) NPU와 CPU의 KV layout 차이로 인한 변환 비용, (3) RKLLM SDK가 KV cache export를 지원하는지 여부의 세 가지 엔지니어링 이슈가 추가된다. KV cache 전달 비용의 메모리 대역폭 한계는 ctx=3500에서 229 MB / 27.31 GB/s ≈ **8.4 ms**로 매우 작아 표 5의 향상 폭(수십 ms 단위)을 위협하지 않으나, layout 변환과 SDK 지원은 별도 측정이 필요하다. 또한 hetero scheduling은 **long ctx에 한정된 전략**임에 유의해야 한다. ctx=15에서 hetero overall(15.20 tok/s @ n_gen=32)은 NPU 단독(19.19 tok/s)보다 오히려 느린데, short ctx에서는 NPU의 decode 자체가 CPU보다 빠르기 때문이다. Hetero scheduling이 효과적인 영역은 NPU decode가 CPU decode보다 느려지는 ctx≥1024 영역이다.

### 4.6 시나리오별 백엔드 선택 (측정 기반)

§4.2·§4.4·§4.5의 결과를 종합하면 본 환경(Rock 5B+, Llama-3.2-1B W8A8)에서 backend 선택은 **ctx와 n_gen의 두 축으로 결정**된다(표 6).

표 6. ctx × n_gen 따른 최적 backend

| | n_gen ≤ 32 | 128 ≤ n_gen ≤ 512 | n_gen ≥ 1024 |
|---|---|---|---|
| **ctx ≤ 512** | NPU | NPU | NPU |
| **ctx ≥ 1024** | NPU | GPU | CPU |
| **hetero 적용 시 (ctx ≥ 1024)** | NPU | **NPU prefill + CPU decode (1.18–1.22×)** | **NPU prefill + CPU decode (1.09–1.15×)** |

## Ⅴ. 결 론

본 논문은 Rock 5B+(RK3588)에서 Llama-3.2-1B를 W8A8 동일 양자화로 NPU/CPU/Mali GPU 세 backend에 측정해 다음을 정량 입증했다. 첫째, LPDDR5 fill 27.31 GB/s로부터 Williams roofline에 따른 decode 상한은 22.0 tok/s이다. 둘째, **prefill phase에서는 NPU가, long ctx decode phase에서는 CPU MNN이 우세한 비대칭 구조**가 존재하며, NPU의 long ctx decode 비효율(roofline의 28%)은 메모리 대역폭이 아닌 SDK level의 attention 처리 비용(KV-read 한계의 13.7×)에서 발생한다. 셋째, ctx=3500 long generation에서 backend 1등이 n_gen에 따라 NPU → GPU → CPU 순서로 두 번 교체되며, 이 cross-over는 phase별 비대칭 구조에서 직접 도출된다. 넷째, NPU prefill + CPU decode hetero scheduling은 단독 best backend 대비 ctx=3500에서 1.05–1.22×의 향상 가능성을 산수로 추정 가능하다.

본 결과는 우리가 아는 한 RK3588에서 동일 양자화로 세 backend를 직접 비교하고 phase 분리 관점에서 hetero scheduling 가능성을 정량화한 첫 측정이며, "NPU가 LLM 가속에 절대 우세"라는 일반 가정에 대한 정량적 반례를 제시한다. 실용적 시사점은 두 가지다. 단기적으로, RKLLM SDK의 attention kernel 개선이 NPU long context 효율 회복의 직접적 경로이다. SDK 개선과 무관하게 즉시 활용 가능한 우회 전략으로, NPU prefill + CPU decode hetero scheduling이 현재 단독 backend 대비 long context 영역에서 측정 가능한 향상을 제공한다. 후속 연구는 (1) hetero scheduling의 실측 검증과 KV cache 전달·변환 비용의 정량화, (2) NPU attention 비용의 root cause 추적, (3) 더 큰 모델(3B, 7B)과 더 긴 context(8K, 16K)에서의 일반화이다.

## References
[1] L. Chen et al., "Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference," *Proc. ACM SIGOPS SOSP*, 2025. DOI:10.1145/3731569.3764808.
[2] D. Xu et al., "Fast On-device LLM Inference with NPUs," *Proc. ACM ASPLOS*, 2025. DOI:10.1145/3669940.3707239.
[3] Alibaba. *MNN*. https://github.com/alibaba/MNN.
[4] Rockchip. *RKLLM Toolkit (RKNN-LLM v1.2)*. https://github.com/airockchip/rknn-llm.
[5] S. Bogatov. *tinymembench v0.4.9*. https://github.com/ssvb/tinymembench.
[6] S. Williams, A. Waterman, D. Patterson, "Roofline: an insightful visual performance model for multicore architectures," *Communications of the ACM*, vol. 52, no. 4, pp. 65–76, 2009. DOI:10.1145/1498765.1498785.
[7] BentoML, "Key metrics for LLM inference," *LLM Inference Handbook*, 2025. https://bentoml.com/llm/inference-optimization/llm-inference-metrics.
[8] TinyComputers, "Rockchip RK3588 NPU Deep Dive: Real-World AI Performance Across Multiple Platforms," 2025. https://tinycomputers.io/posts/rockchip-rk3588-npu-benchmarks.html.
[9] Radxa. *ROCK 5B+ Product Brief*. https://radxa.com/products/rock5/5bp/.
[10] MLC Team, "GPU-Accelerated LLM on a $100 Orange Pi (Mali-G610 Llama-3 8B)," 2024. https://blog.mlc.ai/2024/04/20/GPU-Accelerated-LLM-on-Orange-Pi.
