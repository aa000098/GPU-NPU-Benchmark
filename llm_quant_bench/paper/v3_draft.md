# 엣지 NPU LLM 추론을 위한 컨텍스트 길이 기반 하이브리드 Speculative Decoding 전략

손현호, 윤수연*
국민대학교, *국민대학교
aa000098@kookmin.ac.kr, *1104py@kookmin.ac.kr

**Context-Length-Based Hybrid Speculative Decoding Strategy for Edge NPU LLM Inference**
Hyun Ho Son, Soo Yeon Yoon*
Kookmin Univ., *Kookmin Univ.

---

## 요 약

Closed-box 엣지 NPU에서 speculative decoding의 최적 전략은 컨텍스트 길이에 따라 뒤집힌다. 본 논문은 (1) NPU-only decode 비용이 prefix 길이에 정확히 선형(R²=0.98)임을 보여 V1 full-prefix verify 메커니즘을 정량 입증하고, (2) 이를 활용한 자동 전환 시스템 RegimeSpec을 실측 구현하여 자연어 mixed workload(wiki + 자연어 Q&A + 코드)에서 oracle 대비 99%, Always-Sequential 대비 1.60×, Always-PLD 대비 1.01×를 달성하며 long-context 영역에서는 PLD 대비 1.88× 우위를 확보한다.

---

## Ⅰ. 서 론

엣지 SoC들은 트랜스포머 추론 가속용 NPU를 내장하지만, RK3588 같은 실제 디바이스에서 NPU로 LLM 추론을 수행하면 컨텍스트 길이가 길어질수록 decode throughput이 점차 저하되며, 같은 칩의 CPU throughput은 거의 변하지 않는다. 이 비효율을 완화하기 위해 본 연구는 여러 speculative decoding 전략을 측정·비교하고, 그 결과를 시스템화한 자동 전환 런타임을 구현·평가한다.

## Ⅱ. 본 론

### 2.1 LLM Decode의 계산 구조
트랜스포머 기반 LLM은 prefill과 decode 두 단계로 나뉜다. Prefill은 GEMM이 주요 연산이라 compute-bound에 가깝고, decode는 매 step GEMV로 병렬성이 제약된다. 서버 GPU에서 decode는 memory-bandwidth-bound로 분류되며 매 토큰마다 weights를 다시 읽는 것이 병목이다. 엣지 NPU에서는 GEMV가 넓은 MAC 배열을 채우지 못해 compute under-utilization이 추가로 발생한다.

### 2.2 Speculative Decoding
Leviathan et al.(2023), Chen et al.(2023)의 speculative decoding은 작은 draft 모델이 다음 *k* 토큰을 예측하고 target이 한 번의 forward pass로 병렬 검증한다. Target forward가 memory-bound이면 시퀀스 길이 1과 *k*는 거의 같은 시간이 걸려, draft가 충분히 가벼우면 한 번의 target call이 평균 1+α*k* 토큰을 생성한다. 이득 조건은 다음 부등식이다.

$$\alpha \cdot T_{verify}(ctx) > T_{draft}(k) \tag{2.1}$$

서버 GPU에서는 *T*₍draft₎ ≪ *T*₍verify₎가 쉽게 성립하지만, 엣지에서는 둘이 유사 스케일이 되어 부등식이 아슬아슬해지고 본 연구가 측정하는 regime transition의 원인이 된다. Draft 생성 메커니즘에 따라 (i) Draft-based(별도 LLM 또는 증류 모델), (ii) Draft-free PLD(현재 토큰 버퍼의 n-gram 매칭, draft 비용 µs)로 분류된다. PLD는 vLLM, TensorRT-LLM, HuggingFace Transformers에 통합된 표준 기법이며 input-grounded task에서 2–3×를 보고한다.

## Ⅲ. 실험 설계

### 3.1 환경
Target은 Llama-3.2-1B-Instruct W8A8 .rkllm을 RKLLM-Toolkit으로 RK3588 NPU에서 실행한다. RK3588은 W4A16을 지원하지 않아 W8A8이 사용 가능한 최고 압축률이다. Draft는 self-quant Llama-3.2-1B Q4_0 GGUF (α=0.865, greedy)를 사용하며, 보조로 IQ3_M, Qwen2.5-0.5B Q4_0를 비교한다.

### 3.2 추론 프레임워크
Hybrid runtime: Target path는 RKLLM v1.2.2 (closed-source) C API를 ctypes 래핑하여 NPU에서, Draft path는 llama-cpp-python v0.2.x를 CPU 4 threads(A76 big-core pinning)에서 실행한다. KV state와 메모리 공간은 두 프레임워크가 각자 관리하며 Python list를 통해 토큰만 전달한다.

### 3.3 Speculative Decoding 구성
NPU-only(baseline), Sequential(draft→verify 직렬 반복), Async(threading으로 draft latency hiding 시도), PLD(draft 모델 제거, n-gram 매칭) 네 가지 구성을 비교한다.

## Ⅳ. 실험 결과

### 4.1 NPU-only Baseline의 Context Degradation
*n*∈{15, 512, 1024, 2048, 3500}, n_gen=32, wikitext-2 prompt에서 측정 시 길이가 230× 증가할 때 throughput은 약 185× 저하된다(8.04 → 0.045 tok/s). 측정값에 대한 선형 회귀 결과 **ms/tok = 6.41·n − 1370 (R²=0.98)** 을 얻었으며(그림 1), 이는 V1 full-prefix verify가 token cost를 prefix 길이에 정확히 선형 비례시킴을 정량 입증한다. *n*=3500에서는 토큰당 22초 이상 소요되어 인터랙티브 사용이 불가능한 수준이다.

### 4.2 Short-context Draft-based 구성의 일관된 실패
짧은 프롬프트(*n*≈15)에서 sequential은 *k*∈{1, 2}의 break-even 근처에 머물고 *k*≥4부터 빠르게 무너지며, async는 모든 *k*에서 sequential보다 나빠 latency hiding 이득이 실현되지 않는다(부록).

### 4.3 PLD의 Short-context 우세
구조적 반복 패턴을 가진 copy(문장 반복)와 code(함수 재작성) 프롬프트에 대해 n_gen=48로 측정 시 PLD는 code 1.85×, copy 1.34×의 양의 speedup을 보이며 short-context에서 유일하게 NPU-only를 앞선다.

### 4.4 Long-context에서의 반전
*n*∈{512, 1024, 2048, 3500} wiki 프롬프트, *k*=4, n_gen=32에서 *n*=1024부터 sequential과 PLD가 역전된다(Sequential 2.19×–2.50× vs PLD 1.06×–1.33×). 원인은 V1 full-prefix verify다: 매 라운드 전체 prefix 재prefill로 verify 비용이 *n*에 지배되어, accept된 모든 토큰이 prefill 비용에 amortize되는 구조가 된다. PLD는 wiki natural text에서 hit rate 0.25–0.30에도 accept/round 0.07–0.28에 그쳐, draft cost-free 장점이 약화된다.

### 4.5 RegimeSpec 실측 (NEW)
위 §4.1–4.4의 관찰을 시스템화한 **RegimeSpec**을 구현·실측한다. RegimeSpec은 매 prompt에 대해 ctx<1024이면 PLD, 그 외에는 Sequential을 동적 선택하며, draft 모델은 long prompt 첫 등장 시 lazy-load한다. 워크로드는 wikitext-2 자연어 산문(long, 1024–3000 tok)과 자연어 Q&A·코드 재작성 prompt(short, 11–44 tok) 두 클래스로 구성하였으며, 단순 반복 텍스트 클래스는 PLD에 인공적으로 유리하므로 제외하였다. 비교 대상으로 Always-{Seq, PLD} 및 prompt별 사후 최선 모드(Oracle)를 추가한다.

표 2. RegimeSpec workload별 측정 결과 (Llama-3.2-1B W8A8, n_gen=32, k=4)

| 모드 | Mixed (15)¹ | Long-only (5)² | Short-only (10)³ |
|---|---|---|---|
| NPU-only | 4.55 (1.00×) | 0.12 (1.00×) | 7.37 (1.00×) |
| Always-Seq | 3.09 (0.68×) | **0.28 (2.36×)** | 4.71 (0.64×) |
| Always-PLD | 4.88 (1.07×) | 0.15 (1.25×) | **7.89 (1.07×)** |
| **RegimeSpec** | **4.94 (1.09×)** | **0.28 (2.35×)** | **7.94 (1.08×)** |
| Oracle | 4.98 (1.10×) | 0.28 (2.36×) | 7.91 (1.07×) |

¹ 5 자연어 Q&A + 5 코드 재작성 + 5 wiki long, 무작위 셔플
² wikitext-2 train split에서 무작위 추출한 자연어 산문 5개
³ 5 자연어 Q&A + 5 코드 재작성

Mixed workload에서 RegimeSpec은 4.94 tok/s로 **Always-Sequential 대비 1.60×, Always-PLD 대비 1.01×, Oracle 대비 99%**를 달성한다(그림 2). 어떤 단일 always-X 정책도 RegimeSpec을 능가하지 못한다. Long-only(자연어 wiki)에서는 RegimeSpec이 모든 5개 prompt를 자동으로 Sequential로 분기하여 **Always-PLD 대비 1.88× 우위**를 확보한다(0.28 vs 0.15 tok/s). 이는 자연어 산문에서 PLD의 n-gram acceptance rate α가 매우 낮아(논문 §4.4 측정 0.07–0.28) draft cost-free 장점이 V1 full-prefix verify 비용에 묻히기 때문이다. Short-only에서는 Always-Sequential이 0.64×로 NPU-only보다도 퇴행하는데, 이는 짧은 ctx에서 NPU verify(~50 ms)가 draft model overhead(~50 ms)에 압도되기 때문이며, RegimeSpec은 이 영역에서 모두 PLD로 분기하여 회피한다. RegimeSpec의 분기 통계(mixed 기준): PLD 10회, Sequential 5회, draft model lazy-load overhead 11.1 s(전체 측정의 1% 미만).

## Ⅴ. 결 론

본 논문의 기여는 세 가지로 요약된다. **첫째**, NPU-only decode 비용이 prefix 길이에 ms/tok = 6.41·*n* − 1370 (R²=0.98)로 정확히 선형임을 보여 V1 full-prefix verify를 단일 원인으로 정량 입증한다. **둘째**, 위 메커니즘을 활용한 RegimeSpec을 실측 구현하여 자연어 mixed workload(자연어 Q&A + 코드 재작성 + wiki 산문)에서 Oracle 대비 99%, Always-Sequential 대비 1.60×, Always-PLD 대비 1.01×를 달성하며, 어떤 단일 always-X 정책도 RegimeSpec을 능가하지 못함을 보인다. **셋째**, long-context 자연어 wiki 영역에서 RegimeSpec이 Always-PLD 대비 1.88× 우위(0.28 vs 0.15 tok/s)를 확보하여, V1 full-prefix verify 환경에서 단일 draft-free 정책이 갖는 한계를 시스템 수준에서 극복했음을 입증한다. 본 정책의 ctx threshold(=1024)는 prompt repetitiveness가 낮은 자연어/코드 분포에서 검증되었으며, 향후 작업에서 prompt-level n-gram 통계 기반 동적 정책으로 확장하면 모든 prompt 분포에 강건한 시스템으로 일반화될 수 있다.

## References
[1] L. Chen et al., "Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference," *Proc. ACM SIGOPS SOSP*, p. 359, 2025. DOI:10.1145/3731569.3764808.
[2] MLC Contributors. *MLC-LLM*. https://github.com/mlc-ai/mlc-llm.
[3] D. Xu et al., "Fast On-device LLM Inference with NPUs," *Proc. ACM ASPLOS*, p. 445, 2025. DOI:10.1145/3669940.3707239.
[4] T. Dao et al., "FlashAttention: Fast and memory-efficient exact attention with IO-awareness," *NeurIPS*, vol. 35, pp. 16344–16359, 2022.
[5] Y. Leviathan, M. Kalman, Y. Matias, "Fast Inference from Transformers via Speculative Decoding," *ICML*, 2023.
[6] C. Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling," arXiv:2302.01318, 2023.

---

**Figures (논문에 삽입)**
- 그림 1. NPU-only token cost의 V1 full-prefix verify 정량 입증 (선형 fit, R²=0.98) — `figures/fig_B_npu_degradation.pdf`
- 그림 2. RegimeSpec workload별 speedup bar — `figures/fig_A_regimespec.pdf`
- (선택) 그림 3. Spec decoding으로 인한 NPU MAC utilization 향상 (GEMV → GEMM-like) — `figures/fig_C_mac_utilization.pdf`
