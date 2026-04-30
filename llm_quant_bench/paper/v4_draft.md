# RK3588 엣지 SoC에서 LLM 추론의 메모리 대역폭 한계와 백엔드별 효율 분석

손현호, 윤수연*
국민대학교, *국민대학교
aa000098@kookmin.ac.kr, *1104py@kookmin.ac.kr

**Memory-Bandwidth-Bounded Edge LLM Inference: Multi-Backend Characterization on RK3588**
Hyun Ho Son, Soo Yeon Yoon*
Kookmin Univ., *Kookmin Univ.

---

## 요 약

엣지 SoC RK3588은 NPU(6 TOPS), Cortex-A76 CPU, Mali-G610 GPU 세 가속기를 LPDDR5 메모리(64-bit @ 5500 MT/s)에 공유한다. 본 논문은 Llama-3.2-1B(W8A8) 추론을 동일 양자화로 세 백엔드에서 측정해 (1) LPDDR5 단일 방향 처리량(write fill 27.31 GB/s)으로부터 Williams의 roofline 방법론[6]에 따라 decode 처리량 상한 22.0 tok/s를 도출하고, (2) NPU가 short context(≤512)에서 한계의 95%를 달성하지만 long context(≥2048)에서는 토큰당 비용이 메모리 한계로부터 예측되는 attention KV-read 비용 대비 12.5× 큰 ctx-비례 오버헤드로 한계 대비 28%로 떨어지며, (3) ctx=3500 long generation에서 n_gen=128부터 OpenCL Mali GPU가 NPU를 추월하고 n_gen=512에서 GPU 7.154 > CPU 6.519 > NPU 4.912 tok/s 순서로 NPU가 꼴찌가 됨을 정량 입증한다.

---

## Ⅰ. 서 론

엣지 SoC RK3588은 NPU(6 TOPS), Cortex-A76 4코어 CPU, Mali-G610 GPU 세 가속기를 LPDDR5 메모리에 공유하며, 모두 LLM 추론에 사용 가능한 backend다. 그러나 어느 backend가 어느 시나리오에서 우세한지에 대한 정량 publication은 부재하며, 일반적으로 NPU가 LLM 가속에 절대 우세하다고 광고된다. 본 연구는 Rock 5B+(RK3588) 보드에서 Llama-3.2-1B를 W8A8 동일 양자화로 NPU(RKLLM), CPU(MNN), Mali GPU(MNN OpenCL) 세 backend에서 5개 컨텍스트 길이(15, 512, 1024, 2048, 3500)로 측정하고, LPDDR5 메모리 대역폭으로부터 도출한 roofline 위에 위치시켜 각 backend의 한계와 효율을 진단한다.

## Ⅱ. 본 론

### 2.1 LLM Decode의 메모리 대역폭 한계
트랜스포머 LLM의 토큰별 decode는 매 단계에서 모델 weight 전체를 메모리에서 읽어 GEMV 연산을 수행한다. Williams 등이 제안한 roofline 방법론[6]에 따르면 decode가 weight read에 의해 dominant하게 결정될 때 토큰당 처리량 상한은

$$\text{tok/s}_{\max} = \frac{B_{\text{read}}}{S_{\text{weight}}} \tag{1}$$

이며, $B_{\text{read}}$는 보드의 단일 방향 read 대역폭, $S_{\text{weight}}$는 모델 weight 크기다. Llama-3.2-1B의 weight 크기는 1.24B 파라미터 × 1 byte (W8A8) = **1.24 GB**다.

### 2.2 RK3588 SoC 및 Rock 5B+ 구조
RK3588은 6 TOPS NPU(3 cores), Cortex-A76×4 + A55×4 CPU, Mali-G610 MP4 GPU를 SoC에 통합한다. 실험 보드 Rock 5B+는 RK3588에 LPDDR5 64-bit @ 5500 MT/s 메모리[9]를 결합하며, **Rock 5B+의 메모리 이론 peak은 5500 MT/s × 8 bytes = 44 GB/s**다. 본 연구는 A76 big core 4개를 모든 측정에 사용한다. NPU는 closed-source RKLLM SDK[4]로, Mali GPU는 MNN OpenCL backend[3]로 접근한다.

## Ⅲ. 실험 설계

대상 모델은 Llama-3.2-1B-Instruct W8A8 양자화로 통일하고 RKLLM(NPU), MNN W8A8(CPU 4 threads, OpenCL Mali) 세 backend로 변환한다. Wikitext-2 자연어 prefix를 5개 길이 *n*∈{15, 512, 1024, 2048, 3500}로 자르고 n_gen=32 토큰을 생성한다. 메모리 대역폭은 tinymembench v0.4.9로 측정한다.

## Ⅳ. 실험 결과

### 4.1 LPDDR5 Roofline 도출
tinymembench v0.4.9[5]로 측정한 Rock 5B+의 메모리 처리량은 NEON copy 12.91 GB/s, fill (write only) 27.31 GB/s이다. copy는 read와 write가 동시에 LPDDR 버스를 사용하므로 단일 방향 처리량 추정에 부적합하고, 본 논문은 보수적으로 **fill 27.31 GB/s를 단일 방향 read 대역폭의 upper bound proxy**로 사용한다. 이는 §2.2의 Rock 5B+ 메모리 이론 peak 44 GB/s의 62%다. 이 값을 식 (1)에 대입하면 **decode 처리량 상한 = 27.31 / 1.24 = 22.0 tok/s**가 된다 (그림 1, 수평선).

![Fig 1. RK3588 LPDDR5 Roofline + Multi-backend Decode](v4_figures/fig_roofline.png)

### 4.2 백엔드별 처리량 비교
표 1은 세 backend의 동일 양자화(W8A8) 측정 결과다. NPU는 ctx=15에서 decode 20.80 tok/s (roofline 22.0 tok/s의 95%)로 메모리 한계에 근접하나, ctx=3500에서 6.12 tok/s (한계의 28%)로 추락한다. CPU MNN은 ctx에 거의 무관하게 13.0–16.2 tok/s를 유지하며, Mali GPU는 11.4–14.9 tok/s로 그보다 약간 낮다. 임계점은 ctx=1024이며, 이 이후 CPU MNN이 NPU를 능가하고 ctx=3500에서 격차가 **2.13×**로 벌어진다.

표 1. **TTFT** (Time To First Token, sec) — Llama-3.2-1B W8A8, Rock 5B+

| ctx | NPU (RKLLM) | CPU MNN | OpenCL Mali |
|---:|---:|---:|---:|
| 15 | **0.128** | 0.201 | 0.269 |
| 512 | **1.95** | 3.55 | 2.87 |
| 1024 | **4.42** | 8.10 | 5.99 |
| 2048 | **10.85** | 21.74 | 14.34 |
| 3500 | **23.06** | 38.42 | 28.83 |

표 2. **Decode** (tok/s, 메모리 대역폭 한계 22.0 tok/s 대비 효율)

| ctx | NPU | CPU MNN | OpenCL Mali | NPU/한계 |
|---:|---:|---:|---:|---:|
| 15 | **20.80** | 16.18 | 14.93 | **95%** |
| 512 | **15.09** | 15.01 | 13.76 | 69% |
| 1024 | 12.20 | **14.80** | 13.04 | 55% |
| 2048 | 8.46 | **13.84** | 12.68 | 38% |
| 3500 | 6.12 | **13.05** | 11.41 | 28% |

NPU는 모든 ctx에서 가장 빠른 TTFT를 보이며 인터랙티브 SLO(≤500 ms)[7]를 충족하는 영역은 NPU의 ctx=15(128 ms)뿐이고, ctx=512부터 모든 backend에서 1초를 초과한다. Overall(TTFT + 32 decode 합산) 처리량은 모든 ctx에서 NPU가 1등이며 ctx=3500에서 1.13 tok/s로 CPU MNN의 0.78 tok/s를 1.45× 앞선다. n_gen이 커질수록 decode 비중이 prefill을 추월하므로 시나리오에 따라 backend의 우위가 달라질 가능성이 있으며, 정확한 boundary는 별도 측정으로 확정해야 한다.

![Fig 2. Backend-by-Backend Throughput Decomposition (TTFT + Decode)](v4_figures/fig_backends.png)

### 4.3 NPU의 ctx Scaling 비효율
NPU의 토큰당 decode 비용은 ctx에 비례해 증가한다(48 ms @ ctx=15 → 163 ms @ ctx=3500). roofline(식 1)에 따른 weight read 비용 하한은 1.24 GB / 27.31 GB/s = **45 ms**이며 ctx에 무관해야 하므로, NPU의 ctx 비례 증가분 115 ms(ctx=15 → 3500)는 **weight read 외 추가 비용**이다.

이 추가 비용이 attention의 KV-cache traversal에 의한 것이라고 가정하면, KV 크기는 16 layers × 32 heads × 64 head_dim × 2 (K+V) × 1 byte = 65.5 KB/token이고 ctx=3500의 KV는 229 MB이다. 27.31 GB/s read bandwidth로는 **8.4 ms**가 한계이다. 그러나 NPU 실측 추가분 115 ms는 이 상한의 **13.7×**이며, 동일 backend의 KV reuse 정상 작동을 토큰별 timing probe로 확인했음에도(ctx=1024에서 토큰별 78.8–79.5 ms, std 6.6 ms) 이 격차가 좁혀지지 않는다. 같은 ctx 증가 구간에서 CPU MNN은 추가분이 15 ms로, attention 구현에 따라 ctx-비례 비용이 대역폭 한계 가까이 머무를 수 있음을 보여준다. NPU의 13.7× gap의 근본 원인은 RKLLM SDK가 closed-source이므로 직접 검증 불가능하다.

### 4.4 Long Generation에서의 백엔드 Cross-over (실측)
§4.2의 표 1·2는 n_gen=32에서 측정된 결과로, decode 비중이 prefill 대비 작다. 실용적 시나리오 (creative writing, code generation, agentic 응답)에서 n_gen이 더 커질 때의 cross-over를 직접 측정하기 위해 ctx=3500 고정 후 n_gen∈{32, 128, 256, 512}에서 세 backend를 측정했다(표 3, 그림 3). 각 측정은 backend 사이 5분 cooldown 후 3회 반복 평균이다.

표 3. ctx=3500, varying n_gen 측정 (mean ± std, 3 repeats)

| n_gen | NPU (tok/s) | CPU MNN (tok/s) | OpenCL Mali (tok/s) |
|---:|---:|---:|---:|
| 32 | **1.146 ± 0.026** | 0.788 | 1.031 |
| 128 | 2.996 ± 0.006 | 2.648 | **3.323** |
| 256 | 4.063 ± 0.002 | 4.385 | **5.117** |
| 512 | 4.912 ± 0.008 | 6.519 | **7.154** |

![Fig 3. Backend cross-over at ctx=3500 with varying n_gen](v4_figures/fig_longgen.png)

n_gen=32에서는 NPU가 1등이지만, **n_gen=128부터 OpenCL Mali GPU가 NPU를 추월**하고, **n_gen=512에서는 CPU MNN도 NPU를 추월**하여 backend 순위가 GPU > CPU > NPU로 역전된다. n_gen=512에서 GPU 7.154 tok/s는 NPU 4.912 tok/s 대비 1.46×, CPU MNN 대비 1.10× 빠르다.

이 결과는 외부에서 보고된 RK3588 단일 backend 측정과 다음과 같이 비교된다.

- 외부 보고[8]: RK3588 NPU + TinyLlama 1.1B Q8 = 10–15 tok/s, short ctx (n_gen 미명시).
- 우리 측정: RK3588 NPU + Llama-3.2-1B W8A8, ctx=15에서 decode 19.63 ± 0.54 tok/s, ctx=3500/n_gen=512에서 4.912 ± 0.008 tok/s.
- 외부 보고[10]: Mali-G610 + mlc-llm Llama-3-8B (Q4) = ~2 tok/s, Orange Pi 5.
- 우리 측정: Mali-G610 + MNN Llama-3.2-1B W8A8, ctx=3500/n_gen=512에서 7.154 tok/s. (모델 크기와 양자화 차이로 직접 비교는 어렵지만 같은 LPDDR5 시스템에서의 절대값.)

### 4.5 시나리오별 백엔드 선택 (측정 기반)
표 1·2·3의 측정에 따라 우리 환경(Rock 5B+, Llama-3.2-1B W8A8, n_gen=32–512)에서의 backend 순위는 다음과 같다.

- **Short prompt + short answer (ctx<512, n_gen=32)**: NPU가 overall 1등(표 1·2).
- **Long prompt + short answer (ctx≥1024, n_gen=32)**: NPU가 overall 1등(표 1·2).
- **Long prompt + long answer (ctx=3500, n_gen≥128)**: GPU OpenCL Mali가 1등(표 3).
- **Long prompt + 매우 긴 answer (ctx=3500, n_gen=512)**: GPU > CPU > NPU 순서, NPU가 꼴찌.

## Ⅴ. 결 론

본 논문은 Rock 5B+(RK3588)에서 Llama-3.2-1B를 W8A8 동일 양자화로 NPU/CPU/Mali GPU에 측정해 다음 네 가지를 정량 입증했다. 첫째, LPDDR5 단일 방향 처리량(write fill 27.31 GB/s)으로부터 Williams roofline 모델[6]에 따른 decode 상한은 22.0 tok/s다. 둘째, NPU의 ctx=15 decode는 20.80 tok/s(상한의 95%)로 메모리 한계에 근접하나 ctx=3500에서 6.12 tok/s(상한의 28%)로 추락한다. 셋째, ctx≥1024 decode 영역에서 CPU MNN이 NPU를 능가하며 ctx=3500에서 격차는 2.13×다. 넷째, ctx=3500 long generation에서 backend 순위가 n_gen에 따라 변동하여 n_gen=32에서 NPU가 1등(1.146 tok/s)이지만 n_gen=128부터 GPU OpenCL이 NPU를 추월하고 n_gen=512에서 GPU 7.154 > CPU 6.519 > NPU 4.912 tok/s 순서로 NPU가 꼴찌가 된다. NPU의 ctx-비례 추가 비용 115 ms(ctx=15→3500 구간)는 동 구간 KV-cache read의 메모리 대역폭 한계 8.4 ms 대비 13.7× 크다. 본 결과는 우리가 아는 한 RK3588에서 동일 양자화로 NPU·CPU·GPU 세 backend를 비교한 첫 측정이며, NPU의 long-generation 효율 부족이 LPDDR5 메모리 대역폭 한계 자체가 아니라 SDK level의 추가 비용에서 발생함을 시사한다. 후속 작업은 NPU의 long context 추가 비용의 root cause 추적과 OpenCL Mali GPU의 long generation 효율의 architectural 분석이다.

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

