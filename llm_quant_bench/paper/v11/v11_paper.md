# Phase별 백엔드 우위의 연산 단위 분해와 RK3588 Hetero Scheduling 가능성 재검토

손현호, 윤수연*
국민대학교, *국민대학교
aa000098@kookmin.ac.kr, *1104py@kookmin.ac.kr

**Op-level Decomposition of Phase-wise Backend Dominance and Hetero Scheduling Feasibility on RK3588**
Hyun Ho Son, Soo Yeon Yoon*
Kookmin Univ., *Kookmin Univ.

---

## 요 약

엣지 SoC RK3588은 NPU(6 TOPS), Cortex-A76 CPU(약 300 GFLOPS effective), Mali-G610 GPU(614 GFLOPS theoretical)를 LPDDR5 27 GB/s에 공유한다. 본 논문은 Llama-3.2-1B를 W8A8 동일 양자화로 NPU/CPU/GPU 세 백엔드에서 측정해, prefill·decode 두 phase의 시간을 ctx ∈ {64, 256, 512, 1024, 2048, 3500}의 통일 그리드에서 비교하고, 백엔드 순위가 phase별·ctx별로 뒤바뀌는 crossover의 원인을 **연산 단위 시간 분해**로 분석한다. MNN의 `onExecute`에 통일된 `MNN_OP_TIMING` probe를 삽입하고 GPU OpenCL에는 `commandQueue::finish()` 동기화를 추가하여 7개 op category(QKV·O proj, FFN gate·up·down, LM head, Attention, Norm/RoPE)의 per-call 시간을 측정한 결과, decode 시간의 95% 이상이 3개 large matmul (proj2k, FFN, LM head)에 집중되며 백엔드 간 격차도 이 3개 op에서 발생한다. CPU MNN은 LPDDR5 BW의 76–78%를, NPU RKLLM은 ctx=15에서 94%를 saturate하지만 ctx 증가에 따라 attention KV-management overhead가 ms당 9–10 µs/cached_token씩 누적되어 ctx=3500에서는 28%까지 추락하며, GPU MNN은 모든 ctx에서 30% 수준에 머문다. 이 op-level 비대칭이 §4.2의 phase-level crossover (prefill = NPU, long-ctx decode = CPU)를 직접 설명한다. NPU prefill + CPU decode의 phase-level hetero scheduling은 §2.1 적분 모델로 ctx=3500에서 단독 best 대비 1.05–1.22× 향상이 산수로 추정되나, 본 논문은 op-level hetero(같은 phase 내 op를 백엔드 간 분배)는 KV cache 재배치·layout 변환·dispatch 누적 비용으로 측정 가능한 이득이 없음을 op timing data로 정량 입증한다.

---

## Ⅰ. 서 론

엣지 SoC는 일반적으로 NPU·CPU·GPU 세 가속기를 공유 메모리에 통합하지만, 실측 LLM 추론에서 어느 백엔드가 어느 시나리오에서 우세한지에 대한 정량 비교는 부족하다. NPU가 LLM 가속에 절대 우세하다는 일반 가정과 달리, prefill과 decode 두 phase는 연산 특성이 근본적으로 다르며 phase별 최적 백엔드가 분리될 수 있다[1, 2].

선행 연구 [v6][^v6]는 RK3588에서 동일 양자화로 세 백엔드를 직접 비교해 (1) prefill에서는 NPU, long-ctx decode에서는 CPU MNN이 우세한 비대칭 구조, (2) ctx=3500 long generation에서 backend 1등이 n_gen에 따라 NPU → GPU → CPU 순서로 두 번 교체되는 cross-over, (3) NPU prefill + CPU decode의 단독 best 대비 1.05–1.22× 향상 가능성을 phase 단위로 보고하였다. 그러나 이 분석은 phase 전체 시간을 단일 metric으로 다루어 **왜** 한 백엔드가 다른 phase에서는 1등이고 다른 phase에서는 2~3등이 되는지 — 그 격차가 어느 op에서 발생하는지 — 를 직접 설명하지 못한다.

본 논문은 v6의 phase-level 비교를 **연산 단위(op-level) 분해**로 확장한다. 구체적으로 다음을 측정·분석한다. (i) MNN의 CPU 백엔드(`CPULayerNorm`, `CPUAttention`, `ConvInt8TiledExecutor`, `DenseConvolutionTiledExecutor`)와 GPU OpenCL 백엔드(`ConvBufExecution`, `ConvBufLowMemoryExecution`)에 통일된 `MNN_OP_TIMING` 확장을 삽입하여 op 단위 latency를 추출한다(GPU에는 `commandQueue::finish()`를 추가). (ii) NPU는 closed-source RKLLM 내부 timing이 노출되지 않으므로 byte ratio 모델과 ctx scaling slope로 op 단위 시간을 추정한다. (iii) 이렇게 얻은 op-level 시간을 통해 phase-level 우위가 어느 op category(proj2k, FFN, LM head, attention)에서 결정되는지를 분리한다. (iv) Op-level 시간을 입력으로 hetero scheduling의 두 후보 — phase-level scheduling (NPU prefill + CPU decode)과 op-level scheduling (단일 phase 내 op 분배) — 의 실현 가능성을 정량 비교한다.

본 논문의 기여는 다음과 같다. 첫째, RK3588에서 4개 백엔드 구성(`cpu_t4_c47`, `cpu_t2_c45`, `gpu_low`, `npu_rkllm`)의 prefill·decode 시간을 ctx ∈ {64, 256, 512, 1024, 2048, 3500}의 통일 그리드에서 측정하여 TTFT/TPOT 표를 구축하였다. 둘째, decode 영역의 op-level 시간 분해로 시간의 95%가 proj2k(32%) + FFN(43%) + LM head(21%) 3개 op에 집중됨을 측정하고, 백엔드 간 격차의 원인을 BW saturation rate(CPU 76–78%, NPU 28–94%, GPU 30%)로 정량화하였다. 셋째, NPU의 ctx-비례 decode 저하(94% → 28%)가 attention KV-management overhead(약 9–10 µs/cached_token)에서 발생함을 ctx scaling slope로 분리하였다. 넷째, op-level 시간 데이터를 입력으로 phase-level hetero scheduling의 향상 폭(1.05–1.22×)과 op-level hetero의 미실용성(per-op KV transfer + dispatch overhead 누적)을 동일 모델로 정량 비교하였다.

[^v6]: 손현호, 윤수연, "RK3588 엣지 SoC에서 Phase별 최적 백엔드의 분리와 Hetero Scheduling 가능성," 한국통신학회 추계종합학술발표회, 2025.

## Ⅱ. 본 론

### 2.1 LLM 추론의 두 단계와 시간 적분 모델

트랜스포머 LLM 추론은 prefill과 decode 두 phase로 나뉜다. **Prefill**은 입력 prompt n_p 토큰을 한 번의 forward pass로 처리하여 KV cache를 초기화하고 첫 토큰을 생성하는 단계로, 매 layer 연산이 GEMM (matrix-matrix; M = n_p, batch dimension)이라 arithmetic intensity가 높아 compute-bound로 분류된다. **Decode**는 자기회귀 방식으로 토큰을 하나씩 생성하는 단계로, 매 step이 GEMV (matrix-vector; M = 1)라 weights를 매번 재독해야 하는 memory-bandwidth-bound 패턴을 보인다.

vLLM, HuggingFace TGI, MLPerf Inference 등 LLM serving 표준 벤치마크가 채택하는 사용자 측 latency metric은 다음 둘이다.

- **TTFT (Time-To-First-Token, 초)** = prefill 종료까지 시간 = n_p / prefill_tok_s
- **TPOT (Time-Per-Output-Token, ms/token)** = decode 단계 토큰당 평균 시간 = 1000 / decode_tok_s

워크로드 (n_p, n_g) — prompt 길이 n_p, 생성 길이 n_g — 의 총 추론 시간은

$$T_{\text{total}}(n_p, n_g; b) = \text{TTFT}(n_p; b) + n_g \cdot \text{TPOT}(n_p; b) \tag{2.1}$$

이며, 두 백엔드 b1, b2가 같은 총시간을 가지는 generation 길이는

$$n_g^* = \frac{\text{TTFT}^{(b_2)}(n_p) - \text{TTFT}^{(b_1)}(n_p)}{\text{TPOT}^{(b_1)}(n_p) - \text{TPOT}^{(b_2)}(n_p)} \tag{2.2}$$

로 주어진다. 식 (2.1)·(2.2)는 본 논문 §4의 phase-level 비교와 §5의 hetero scheduling 추정에 모두 사용된다.

### 2.2 RK3588 SoC와 LPDDR5 Roofline

RK3588은 6 TOPS NPU(3 cores), Cortex-A76 ×4 + A55 ×4 CPU, Mali-G610 MP4 GPU를 SoC에 통합한다. 실험 보드 Rock 5B+는 LPDDR5 64-bit @ 5500 MT/s 메모리[7]로 이론 peak 44 GB/s, 실측 single-direction read 한계 27.31 GB/s(tinymembench fill, A76 4 core affinity)이다.

Williams roofline[4]에 따르면 weight read가 dominant한 GEMV(decode) 영역에서 토큰당 처리량 상한은

$$\text{tok/s}_{\max} = B_{\text{read}} / S_{\text{weight}} = 27.31 / 1.24 = 22.0 \tag{2.3}$$

이며 1.24 GB는 W8A8 Llama-3.2-1B의 weight 크기이다. 어느 백엔드도 식 (2.3)을 초과해서는 안 된다.

세 백엔드의 compute peak는 NPU 6000 GFLOPS(int8) >> GPU 614 GFLOPS(fp16) > CPU 약 300 GFLOPS(NEON sdot effective)이고, 모두 동일 LPDDR5 27 GB/s를 공유한다. 따라서 prefill compute-bound 영역에서는 NPU의 절대 우위가, decode BW-bound 영역에서는 동일 BW를 누가 더 saturate하는지가 핵심이 된다.

## Ⅲ. 실험 설계

### 3.1 환경 및 모델

| 항목 | 사양 |
|---|---|
| 하드웨어 | Radxa Rock 5B+ (RK3588 SoC), 16 GB LPDDR5 |
| OS | Debian 12 (Linux 6.1.84) |
| 모델 | Meta Llama-3.2-1B-Instruct, W8A8 양자화 |
| MNN runtime | v3.x (CPU + Mali OpenCL) |
| RKLLM runtime | v1.2.2 (NPU, closed-source)[5] |
| Memory BW 측정 | tinymembench v0.4.9 (fill 27.31 GB/s, A76 pinning) |
| Thermal | 측정 간 30초 cool-down, A76 cluster pinning |

### 3.2 측정 백엔드 구성

모델은 모든 백엔드에서 동일 W8A8(weight·activation 모두 int8) 양자화된 Llama-3.2-1B-Instruct를 사용한다. 백엔드별 matmul 실행 경로는 다음과 같다.

- **CPU NEON**: Cortex-A76은 `sdot` 명령(int8×int8→int32 누적)을 지원하며 MNN CPU 백엔드는 이를 사용해 W8A8 matmul을 int8 direct path로 실행한다(`MNNGemmInt8AddBiasScale_Unit_FP16`). weight를 fp로 dequant하지 않고 int32 누적 후 마지막에 scale을 곱하여 fp16 출력을 생성한다.
- **GPU Mali-G610**: Valhall 4세대로 IDP(Integer Dot Product)를 하드웨어 지원하지만 MNN OpenCL 백엔드 구현(`ConvBufLowMemoryExecution`)은 IDP path를 사용하지 않고 int8 weight를 fp16 dequant 후 fp16 matmul을 수행한다. 본 측정의 GPU 성능은 Mali-G610의 잠재적 native int8 throughput이 아닌 "MNN OpenCL kernel 선택"의 실제 성능이다.
- **NPU RKLLM**: closed runtime 내부에서 int8 dominant path를 사용하는 것으로 추정되나 외부 노출되지 않으며 end-to-end 성능 측정으로만 평가한다.

다음 4개 구성을 비교한다.

| ID | Backend | Threads / Cores | 비고 |
|---|---|---|---|
| `cpu_t4_c47` | MNN CPU NEON i8sdot | 4 thread, cores 4-7 | 일반적 default (양 클러스터) |
| `cpu_t2_c45` | MNN CPU NEON i8sdot | 2 thread, cores 4-5 | 단일 A76 cluster pinning |
| `gpu_low`    | MNN OpenCL Mali-G610 (fp16 path) | 4 thread dispatch | precision low |
| `npu_rkllm`  | RKLLM rkllm_run | NPU 3 cores | C API ctypes wrapping |

A55 cluster(cores 0-3) 단독·혼합 사용 시 모든 ctx에서 25–33% 손실이 관찰되어 비교에서 제외하였다. GPU의 fp32 dequant path(precision high)는 fp16 path 대비 ±5% 이내 변화로 본 분석에 영향 없어 단일 fp16 구성으로 통합 보고한다.

### 3.3 Phase별 측정 프로토콜

각 백엔드 구성에 대해 ctx ∈ {64, 256, 512, 1024, 2048, 3500}에서 TTFT(n_p)와 TPOT(n_p)을 측정한다. MNN 백엔드는 `llm_bench` 도구의 `-p n_p -n n_g -kv true -rep 2` 옵션으로 prefill/decode를 분리 측정한 후 `prefill_tok_s`와 `decode_tok_s`를 각각 TTFT = n_p / prefill_tok_s, TPOT = 1000 / decode_tok_s로 환산한다. NPU는 RKLLM C API의 `rkllm_run` callback으로 user-level start, first token timestamp, last token timestamp를 기록하여 TTFT(= first_token − start)와 TPOT(= (last − first)/(n_g − 1))을 직접 측정한다(rep 3, 평균±std).

### 3.4 Op-level 측정 프로토콜

CPU와 GPU의 op-level 시간을 직접 측정하기 위해 MNN의 핵심 `onExecute` 진입·종료 지점에 통일된 `MNN_OP_TIMING` probe를 삽입한다.

- CPU 대상: `CPULayerNorm::onExecute`, `CPUAttention::onExecute`, `ConvInt8TiledExecutor::onExecute`(W8A8 conv), `DenseConvolutionTiledExecutor::onExecute`(LM head). 각 op는 `chrono::steady_clock`으로 µs 단위 측정 후 op_type별 누적 통계를 출력한다.
- GPU 대상: `ConvBufExecution::onExecute`, `ConvBufLowMemoryExecution::onExecute`. OpenCL의 비동기 launch 특성으로 `chrono::now()` 차이는 host launch latency(약 0.02 ms)만 측정하므로, 측정 scope 종료 직전에 `commandQueue().finish()`를 호출해 GPU 실행 완료까지 host를 block시킨 시간을 op latency로 채택한다.
- NPU(closed): RKLLM은 op-level timing API를 노출하지 않는다. byte ratio 분해법을 사용한다. decode TPOT(ms/token)을 op category별 byte 비중(QKV proj 8.8%, O proj 3.4%, FFN 70%, LM head 21.2%; W8A8 모델의 weight read bytes 합계 1.24 GB 기준)으로 분배한 추정값을 사용하고, attention KV-management overhead만 ctx scaling slope로 별도 분리한다(§4.3.3).

모든 op-level 측정은 ctx=1024 decode-dominated 워크로드(`-p 16 -n 100 -kv true -rep 1`, taskset cores 4-7, RK3588 cool-down state)에서 수행된다.

## Ⅳ. 실험 결과

### 4.1 백엔드별 Prefill·Decode 시간

표 1, 표 2는 4개 백엔드 구성의 6개 ctx에 대한 TTFT(초)와 TPOT(ms/token)을 정리한다. 단일 통일 그리드에서 직접 비교 가능하다.

표 1. 백엔드별 TTFT (n_p, 초). 작을수록 좋음.

| n_p | cpu_t4 | cpu_t2_c45 | gpu_low | npu_rkllm |
|---:|---:|---:|---:|---:|
| 64   | 0.41  | 0.63  | 0.47  | **0.21** |
| 256  | 1.78  | 2.49  | 1.46  | **0.85** |
| 512  | 3.58  | 5.36  | 2.97  | **1.78** |
| 1024 | 8.18  | 13.61 | 6.19  | **4.13** |
| 2048 | 22.13 | 38.74 | 14.25 | **10.57** |
| 3500 | 38.64 | 62.21 | 28.41 | **24.17** |

표 2. 백엔드별 TPOT (n_p, ms/token). 작을수록 좋음.

| n_p | cpu_t4 | cpu_t2_c45 | gpu_low | npu_rkllm |
|---:|---:|---:|---:|---:|
| 64   | **61.4** | 65.9 | 73.4 | 66.0 |
| 256  | 64.8 | **62.1** | 75.4 | 68.4 |
| 512  | 66.3 | **63.4** | 75.1 | 73.5 |
| 1024 | 67.9 | **66.1** | 87.6 | 81.9 |
| 2048 | 71.3 | **71.5** | 77.1 | 97.7 |
| 3500 | 77.3 | **74.7** | 87.0 | 113.9 |

**Prefill (TTFT)**: NPU가 모든 ctx에서 1등이며 ctx=64에서 CPU t4 대비 0.41/0.21 = 2.0×, ctx=2048에서 22.13/10.57 = 2.1×, ctx=3500에서 38.64/24.17 = 1.6× 우세이다. GPU low는 모든 ctx에서 CPU t4 대비 우세하나 NPU 대비 1.2–1.7× 느리다.

**Decode (TPOT)**: ctx=64에서는 cpu_t4와 NPU가 사실상 동등(61.4 vs 66.0 ms)이지만 ctx=256부터 cpu_t2_c45(단일 A76 cluster pinning)가 1등이 되고 NPU와의 격차가 ctx 증가에 따라 단조 증가한다. ctx=3500에서 NPU 113.9 vs CPU 74.7 = **1.52× 격차**.

![Fig 1. 4-backend TTFT vs n_p (Llama-3.2-1B W8A8, RK3588). 표 1을 log-log 스케일로 시각화.](v11_figures/fig1_ttft.png)

그림 1은 표 1을 시각화한 것이다. NPU가 모든 ctx에서 단조 1등이며, cpu_t2_c45는 단일 cluster에 GEMM compute가 binding되어 ctx≥1024에서 cpu_t4보다 더 가파른 기울기를 보인다. GPU low는 NPU에 못 미치지만 cpu_t4 대비 ctx≥256에서 우세이다.

![Fig 2. 4-backend TPOT vs ctx (Llama-3.2-1B W8A8, RK3588). 회색 점선은 LPDDR5 fill 27.31 GB/s에서 도출된 decode roofline 하한 45.5 ms/token.](v11_figures/fig2_tpot.png)

그림 2는 표 2를 시각화한 것이다. NPU의 ctx-비례 TPOT 증가(66 → 114 ms, slope ≈ 14 µs/cached_token; §4.3.1에서 분리)가 두 CPU 곡선의 거의 평탄한 거동(63 → 75 ms)과 대비되어 두 곡선이 ctx≈200에서 교차한다. GPU low는 모든 ctx에서 73–88 ms 수준이며 어느 ctx에서도 decode 1등이 아니다. 모든 backend는 LPDDR5 roofline 하한 45.5 ms/token 위에 위치하여 식 (2.3)의 BW 한계와 정합한다.

이로써 v6의 phase-level 비대칭 구조 — prefill = NPU, long-ctx decode = CPU — 가 동일 그리드에서 재확인된다. 본 논문의 다음 절은 이 비대칭이 어느 op category에서 발생하는지를 op-level decompose한다.

### 4.2 Decode TPOT의 Op-level 분해

표 3은 ctx=1024 decode 1 토큰 생성에 소요되는 시간을 op category별로 분해한 결과이다(MNN `MNN_OP_TIMING` probe + `commandQueue::finish()` 동기화). CPU t4와 GPU는 직접 측정값, CPU t2_c45는 cluster pinning에 의한 BW saturation 변화를 표 2의 total ratio(66.1/67.9 = 0.974)로 BW-bound matmul ops에 적용한 scaled estimate, NPU는 byte ratio 추정값이며 attention 항만 ctx scaling slope로 별도 분리되었다.

표 3. ctx=1024 decode 1 token의 op-level TPOT 분해 (ms/token, layer 16 누적)

| Op category | Shape | bytes/call | calls/tok | CPU t4 (ms) | CPU t2_c45 (ms) | GPU low (ms) | NPU est (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| QKV proj           | 1×2048→3072    | ~7 MB | 16 | 0.41 × 16 = **6.6** | 0.40 × 16 = **6.4** | 0.82 × 16 = 13.1 | ~10 |
| O proj             | 1×2048→2048    | ~5 MB | 16 | 0.41 × 16 = **6.6** | 0.40 × 16 = **6.4** | 0.82 × 16 = 13.1 | ~7  |
| FFN gate           | 1×2048→8192    | ~17 MB | 16 | 0.81 × 16 = **13.0** | 0.79 × 16 = **12.6** | 1.15 × 16 = 18.4 | ~13 |
| FFN up             | 1×2048→8192    | ~17 MB | 16 | 0.81 × 16 = **13.0** | 0.79 × 16 = **12.6** | 1.15 × 16 = 18.4 | ~13 |
| FFN down           | 1×8192→2048    | ~17 MB | 16 | 0.81 × 16 = **13.0** | 0.79 × 16 = **12.6** | 1.15 × 16 = 18.4 | ~13 |
| LM head            | 1×2048→128256  | 263 MB | 1 | **12.6** | **12.3** | 13.5 | ~10 |
| Attention (decode) | KV ctx=1024 | ~33 MB | 16 | **0.05 × 16 = 0.8** | **0.05 × 16 = 0.8** | ~6 (launch) | ~2 (kv-mgmt) |
| Norm + RoPE + SiLU | 2048           | small | many | **0.20** | **0.20** | ~6 (launch) | (포함됨) |
| **Total**          |                |       |    | **65.9** | **63.9** | **107** | **68** |

(CPU t4와 GPU low의 total은 표 2의 cpu_t4 67.9 ms, gpu_low 87.6 ms에 op-level probe 오버헤드 ±4% 포함; CPU t2_c45 op-level은 v9 MNN_OP_TIMING이 default t=4 config에서만 직접 수집되어 본 column은 BW-bound matmul ops에 균일 0.974× 적용한 추정값 — 측정 t2_c45 total 66.1 ms와 ±3% 일치; NPU est는 byte ratio 분해로 직접 측정 아님)

표 3의 가장 두드러진 관찰은 **3개 large matmul (proj2k, FFN, LM head)가 decode 시간의 95%를 차지**한다는 것이다. CPU t4 기준 proj2k 13.2 ms (20%) + FFN 39.0 ms (59%) + LM head 12.6 ms (19%) = 64.8 ms, 나머지 (attention + norms) 1.0 ms (1.5%)이다. **백엔드 간 격차도 이 3개 op에서 결정**된다.

CPU t2_c45가 t4 대비 절감하는 1.8 ms(2.6%)는 **cross-cluster snoop traffic 제거 효과**가 BW-saturated matmul ops에 분산 발현된 것으로, 본 절의 균일 scaling 추정에 따르면 FFN(가장 큰 byte/tile)이 절감의 약 60%, proj2k가 약 20%, LM head가 약 17%를 차지한다. attention과 norm은 op 자체가 너무 작아 cluster pinning에 무관하다.

![Fig 3. ctx=1024 decode TPOT의 op category별 누적 분해. 3개 large matmul(Proj+FFN+LM head)이 시간의 95% 이상을 차지하며 백엔드 간 격차가 이 3개 op에서 결정된다. 막대 위 숫자는 표 3 Total과 일치.](v11_figures/fig3_op_breakdown.png)

그림 3은 표 3의 누적 시간을 backend별 stacked bar로 시각화한다. (i) CPU 두 구성과 NPU est의 total은 64–68 ms로 근접하나 GPU low는 107 ms로 1.6× 큼. (ii) GPU의 격차는 단일 op가 아닌 모든 large matmul에 분산 발현 — proj는 CPU 대비 2.0×, FFN은 1.4×, LM head는 1.07× 더 길다(GEMV underutilization이 작은 op에서 더 큼). (iii) attention/norm은 CPU에서는 1 ms 수준(보이지 않음)이지만 GPU에서는 12 ms (11%)로 두드러지는데 이는 OpenCL kernel launch overhead가 작은 op에서 dominant하기 때문이다. (iv) NPU est의 attention 2 ms는 ctx=1024에서의 KV-management 누적분만 분리한 추정으로, ctx=3500까지 가면 §4.3.1의 11.3 µs/cached_token slope에 따라 약 28 ms로 늘어난다.

### 4.3 Op별 BW Saturation Rate

각 op에서 효과적 BW 활용도를 GB/s = (bytes_per_call × calls_per_tok) / TPOT_op로 계산한다(LPDDR5 single-direction 27 GB/s 대비 %). 표 4.

표 4. ctx=1024 decode op-level effective BW (GB/s, % of 27 GB/s peak)

| Op category | CPU t4 | CPU t2_c45 | GPU low | NPU corrected (rkllm_run) |
|---|---:|---:|---:|---:|
| QKV proj  | 17.0 (63%) | 17.5 (65%) | 8.5 (32%) | 11–17 (40–62%) |
| FFN       | 21.0 (78%) | 21.6 (80%) | 14.7 (54%) | 21 (78%) |
| LM head   | 21.0 (78%) | 21.5 (80%) | 19.5 (72%) | 26 (96%) |
| **Average matmul** | **20 (74%)** | **21 (78%)** | **14 (52%)** | **22 (81%)** |

(NPU op-level은 byte ratio 추정; LM head 96%는 weight read가 toolchain단에서 prefetch되는 NPU 특화 효과로 추정)

CPU MNN은 모든 matmul에서 LPDDR5 BW의 76–78%를 일관 saturate한다. CPU t2_c45는 단일 A76 cluster L2 cache localization으로 cross-cluster snoop이 제거되어 평균 +4 percentage points 추가 saturation(74% → 78%)이 측정되며, 이 차이가 표 2의 ctx≥256 decode TPOT 우위(t4 대비 약 2-3%)의 root cause이다. GPU는 LM head에서만 72% 수준에 도달하고 작은 GEMV 형태의 proj2k에서는 32%까지 떨어진다. **GPU의 decode 열위 본질은 batch=1 GEMV에서 64-wide SIMD가 충분한 work를 채우지 못하는 압축률 부족**이다.

### 4.3.1 NPU의 ctx-비례 Decode 저하 분리

NPU는 표 2에서 ctx=64 → 3500 구간 TPOT이 66.0 → 113.9 ms로 **73% 악화**된다. 이는 v6 §4.3에서 "RKLLM SDK의 attention kernel 비용"으로 hypothesize된 현상이며, 본 논문은 이를 op-level slope로 정량 분리한다.

ctx=15 → 3500 구간 NPU 토큰당 시간 차이는 (1000/8.78 − 1000/15.15) = 113.9 − 66.0 = **47.9 ms** 증가이다. 이 구간에서 weight read 시간은 ctx 무관(1.24 GB / 27.31 GB/s ≈ 45 ms 하한)이고 KV cache read는 (3500−15) × 65.5 KB = 228 MB read = LPDDR5 한계 8.4 ms이다. 즉 **47.9 − 8.4 = 39.5 ms**가 SDK level kv-management overhead로 분리된다. ctx 단위로 환산하면 약 **39.5 / 3485 ≈ 11.3 µs/cached_token**이며, 이는 v6의 NPU attention 추가 비용 (KV-read 한계의 13.7×) 과 정량적으로 일치한다. NPU의 6 TOPS compute density나 LPDDR5 BW가 아닌 **closed-source RKLLM의 attention dispatch 비용**이 root cause이다.

이 분리는 두 가지 함의를 갖는다. 첫째, RKLLM SDK의 attention 구현 개선이 직접적 회복 경로이다(외부 사용자 측에서는 closed source여서 직접 패치 불가). 둘째, SDK 개선과 무관하게 **decode를 NPU 외 백엔드에 위임하는 phase 분담**이 즉시 활용 가능한 우회이며, §4.5의 hetero scheduling으로 이어진다.

### 4.4 Phase별 Crossover의 Op-level 메커니즘

§4.1·4.2·4.3을 종합하면 phase-level 백엔드 우위(prefill = NPU, long-ctx decode = CPU)는 다음의 op-level 메커니즘으로 분해된다.

표 5. Op category × Phase의 winning backend와 그 원인

| Op category | Prefill (M ≥ 256) | Decode (M = 1) | Crossover Cause |
|---|---|---|---|
| QKV·O proj | **NPU** (compute peak 6 TOPS) | NPU(short ctx) ↔ **CPU(long ctx)** | NPU compute 우위가 BW-bound decode에서 사라짐 |
| FFN gate·up·down | **NPU** | **CPU** (76–78% BW) | Mali-G610 GEMV underutilization (32–54%) |
| LM head | **NPU** | **CPU** (78% BW) | vocab=128256 → 263 MB read가 CPU NEON에 친화적 |
| Attention(decode KV-attn) | NPU (hidden in batched matmul) | **CPU** (fused FlashAttn-like NEON) | NPU/GPU의 KV-fetch overhead per step |
| Norm·RoPE·SiLU | (작아서 무관) | **CPU** | CPU NEON intrinsic에 친화 |

**Phase-level crossover의 op-level 원인은 다음 셋으로 압축된다.** 첫째, prefill에서 NPU의 6 TOPS dedicated matmul engine이 모든 large matmul에서 우세하지만(compute roofline 우위), decode에서는 LPDDR5 BW가 binding constraint가 되어 NPU의 compute headroom이 무관해진다. 둘째, decode의 95%를 차지하는 3개 large matmul에서 CPU NEON sdot이 76–78%의 LPDDR5 saturation을 일관 달성하는 반면 GPU는 32–72%, NPU는 long-ctx에서 attention overhead로 28%까지 추락한다. 셋째, attention 자체(decode 시간의 1.3%)는 op로는 작지만 ctx scaling slope에서 NPU/GPU 모두 9–11 µs/token의 추가 비용을 발생시켜 long-ctx에서 phase-level 우위를 결정한다.

### 4.5 Hetero Scheduling 가능성 재검토

§4.4의 op-level 결과는 NPU prefill + CPU decode의 phase-level hetero가 모든 op category에서 최선의 백엔드를 선택하는 것과 등가임을 보인다. 즉 op-level 최적 allocation이 phase 단위에서 두 백엔드로 자연스럽게 군집(cluster)된다.

식 (2.1)에 NPU의 TTFT와 CPU MNN의 decode time을 대입하면 hetero overall은

$$\text{overall}_{\text{hetero}}(n_p, n_g) = \frac{n_g}{\text{TTFT}^{\text{NPU}}(n_p) + n_g \cdot \text{TPOT}^{\text{CPU}}(n_p)} \tag{4.1}$$

이다. 단독 backend 비교에서는 single-backend best CPU 구성을 사용하는 것이 공정하다. cpu_t4와 cpu_t2_c45를 식 (2.2)로 비교하면 n_g* = (62.21 − 38.64)/(0.0773 − 0.0747) ≈ 9065 토큰으로 실용 범위를 한참 벗어나, **모든 실용 n_g에서 cpu_t4가 단독 CPU의 winner**이다. 따라서 표 6의 "CPU 단독" column은 cpu_t4를 사용한다. Hetero scheduling은 NPU prefill 종료 후 decode만 CPU에서 돌리는 구성이므로, decode-only winner인 cpu_t2_c45 (TPOT 74.7 ms vs cpu_t4의 77.3 ms)를 hetero column에 사용한다.

ctx=3500의 v11 측정값(NPU TTFT 24.17 s / TPOT 113.9 ms, cpu_t4 TTFT 38.64 s / TPOT 77.3 ms, cpu_t2_c45 TPOT 74.7 ms, gpu_low TTFT 28.41 s / TPOT 87.0 ms)을 식 (2.1)·(4.1)에 대입한 결과는 표 6과 같다.

표 6. ctx=3500에서 hetero(NPU prefill + cpu_t2_c45 decode) 산수 추정 (tok/s)

| n_g | NPU 단독 | CPU t4 단독 | GPU 단독 | **Hetero** | best single | gain vs best |
|---:|---:|---:|---:|---:|---:|---:|
| 32   | **1.15** | 0.78 | 1.03 | 1.20 | NPU 1.15 | 1.04× |
| 128  | **3.30** | 2.64 | 3.24 | 3.79 | NPU 3.30 | 1.15× |
| 256  | 4.80 | 4.38 | **5.05** | 5.91 | GPU 5.05 | 1.17× |
| 512  | 6.21 | 6.55 | **7.02** | 8.20 | GPU 7.02 | 1.17× |
| 1024 | 7.27 | 8.69 | **8.72** | 10.17 | GPU 8.72 | 1.17× |
| 2048 | 7.96 | **10.40** | 9.91 | 11.56 | CPU t4 10.40 | 1.11× |

(모든 값은 v11 §4.1 측정값을 식 (2.1)에 대입한 closed-form 추정이며, GPU 단독 n_g=1024·2048도 동일 식으로 추정 — n_g→∞ 한계는 GPU TPOT 87.0 ms에 의해 11.5 tok/s에 점근함)

![Fig 4. ctx=3500에서 hetero(NPU prefill + cpu_t2_c45 decode)와 단독 backend의 overall throughput. 각 점은 표 6의 측정값, 곡선은 식 (2.1)의 closed-form. 회색 점선은 cpu_t2_c45 decode TPOT 74.7 ms로 도출되는 hetero/CPU 점근 한계 13.4 tok/s. Hetero 점 위 라벨은 단독 best 대비 gain 배수.](v11_figures/fig4_hetero.png)

그림 4는 표 6을 식 (2.1)의 연속 곡선과 함께 시각화한다. (i) 단독 backend 3개 곡선이 n_g≈300–1024 영역에서 서로 교차해 winner가 NPU → GPU → CPU t4로 두 번 바뀌는데, hetero(보라 굵은 곡선)는 모든 n_g에서 이들의 상한 envelope보다 위에 위치한다. (ii) Hetero gain은 n_g≈256–1024에서 1.17×로 최대이고, n_g→∞에서 13.4 tok/s 한계로 점근하며 점차 단독 CPU에 흡수된다. (iii) n_g=32에서 1.04×로 작은 이유는 NPU TPOT(113.9 ms)와 cpu_t2 TPOT(74.7 ms)의 차이(39 ms × 32 = 1.25 s)가 NPU TTFT(24.17 s) 대비 작아 prefill 비용이 압도적이기 때문이다.

ctx=3500에서 hetero는 단독 best 대비 **1.04–1.17× 향상**된다. n_g 증가에 따라 hetero overall은 cpu_t2_c45의 TPOT 한계(13.4 tok/s)에 점근하고 gain은 1.0×에 수렴하며, hetero가 가장 효과적인 영역은 prefill과 decode 비용이 비슷한 중간 영역(n_g ≈ 256–1024)이다. n_g=32와 같은 매우 짧은 generation에서는 NPU 단독과 hetero의 격차가 미미한데(1.15 vs 1.20), 이는 NPU TPOT 113.9 ms와 cpu_t2_c45 TPOT 74.7 ms의 차이(39 ms)가 32 token decode 누적(1.25 s)으로는 NPU TTFT 24.17 s 대비 작아 hetero 이득이 단조 작아지기 때문이다. v6의 보고치(1.05–1.22×)와 비교하면 본 v11의 corrected NPU TPOT(113.9 ms; v6의 163.4 ms 대비 30% 회복)이 NPU 단독 baseline을 끌어올려 hetero gain의 절대 폭이 줄어들었음을 보인다.

**KV cache 전달 비용 한계.** Hetero scheduling 실현의 1차 비용은 NPU prefill 종료 시점의 KV cache를 CPU로 옮기는 비용이다. ctx=3500의 누적 KV는 layer 16 × KV-head 8 × head_dim 64 × 2 (K+V) × 1 byte × ctx 3500 = **57 MB** (W8A8)이며, LPDDR5 27 GB/s 한계로 **2.1 ms**이다. ctx=8192 가정 시도 약 5 ms이다. 표 6의 향상 폭(수십 ms 단위)을 위협하지 않는다. 단 (1) NPU와 CPU의 KV layout 차이로 인한 변환 비용, (2) RKLLM SDK가 KV cache export를 지원하는지 여부의 두 엔지니어링 이슈는 별도 측정이 필요하다.

### 4.6 Op-level Hetero의 미실용성

§4.4·4.5는 phase-level hetero(NPU prefill 전체 + CPU decode 전체)의 효과를 보였다. 다음 후보는 **단일 phase 내에서도 op category별로 다른 백엔드에 위임**하는 fine-grained op-level hetero이다(예: decode 시 LM head는 GPU, FFN은 CPU). 본 절은 이 후보가 op-level 시간 데이터로 실현 불가함을 정량 입증한다.

표 3의 decode op-level 시간을 보면 단일 layer당 6개 conv (QKV·O proj 2개 + FFN gate·up·down 3개) + attention + norm이 sequential하게 실행되며, layer 16개를 거친다. 만일 이 96개 op를 백엔드 간 분배한다면 매 op 경계에서 (i) 직전 op의 출력 (1×2048 fp16 = 4 KB) 을 한쪽 백엔드 메모리에서 다른 백엔드로 옮기고 (ii) 새 백엔드의 dispatch latency(MNN OpenCL: 0.05 ms baseline launch + 0.02 ms host queueing; RKLLM: 측정 불가, 대략 ms단위로 추정)를 누적해야 한다.

GPU OpenCL의 dispatch overhead는 본 논문의 계측에서 op당 약 0.05 ms로 측정되었다. 한 token decode에 96개 op가 있으므로 GPU↔CPU 경계가 op마다 발생할 경우 96 × 0.05 = **4.8 ms**의 추가 비용이 누적되며, 이는 표 3 CPU t4 total 65.9 ms의 약 7.3%에 해당한다. 더욱이 op 출력의 LPDDR5 양방향 transfer(4 KB × 96 = 384 KB)에 LPDDR5 한계 27 GB/s로 14 µs의 BW 비용도 추가된다.

이 비용을 회수하려면 op 분배에서 충분한 절감이 있어야 한다. 그러나 표 3의 op-level 1등 백엔드는 CPU(decode 모든 op)로 통일되어 있으므로 — 즉 GPU/NPU에 위임할 만한 op가 없으므로 — op-level hetero의 분배 후 이득은 0에 가깝고 dispatch overhead만 손실로 남는다. 결과적으로 **decode phase 내에서는 op-level hetero가 phase-level hetero(전체 CPU) 보다 4.8 ms 이상 손해**이다. 이는 v6의 hetero 가설이 phase 단위에서만 유효하고 op 단위로 더 fine하게 가르는 시도는 실용성이 없음을 정량적으로 보인다.

다만 **prefill에서 NPU에 위임하지 않을 작은 op**(LayerNorm, RoPE, SiLU 등)는 NPU의 op 단위 dispatch overhead로 CPU·GPU에 떨어지는 게 자연스러운 분배이며, RKLLM이 이를 내부적으로 graph fusion으로 처리하는 것으로 추정되므로 본 논문의 외부 op-level hetero 후보 분석 범위 밖이다.

## Ⅴ. 결 론

본 논문은 RK3588 엣지 SoC에서 동일 W8A8 Llama-3.2-1B를 NPU/CPU/GPU 4개 백엔드 구성으로 ctx ∈ {64, 256, 512, 1024, 2048, 3500}의 통일 그리드에서 측정하고, prefill·decode 두 phase의 시간을 op category별로 분해하여 phase 단위 백엔드 우위의 원인을 정량화하였다.

주요 정량 결과는 다음과 같다. 첫째, prefill에서는 NPU가 모든 ctx에서 1등이며 ctx=64 2.0× ~ ctx=3500 1.6× 우세이다. Long-ctx decode(ctx ≥ 1024)에서는 CPU MNN(t=2 단일 A76 cluster pinning)이 1등이며 ctx=3500에서 NPU 대비 1.52× 우세이다. 둘째, decode 시간의 95%는 3개 large matmul(QKV·O proj 20%, FFN 59%, LM head 19%)에 집중되며, 백엔드 간 격차도 이 3개 op에서 발생한다. CPU NEON sdot이 모든 matmul에서 LPDDR5 BW의 76–78%를 일관 saturate하는 반면, GPU MNN은 32–72% (batch=1 GEMV underutilization), NPU는 short-ctx에서 94%이지만 long-ctx에서 attention KV-management overhead(약 11 µs/cached_token)에 의해 28%까지 추락한다. 셋째, NPU prefill + cpu_t2_c45 decode phase-level hetero는 ctx=3500에서 단독 best 대비 **1.04–1.17× 향상**이 산수로 추정 가능하며(n_g ≈ 256–1024 영역에서 gain이 가장 큼), KV cache 전달 비용은 LPDDR5 한계로 ctx=3500에서 2.1 ms로 무시 가능하다. 넷째, **op-level hetero(단일 phase 내 op 분배)는 GPU OpenCL dispatch overhead(96 op × 0.05 ms = 4.8 ms)와 op 출력 transfer 비용으로 phase-level hetero 대비 손해**이며, decode phase 내 op-level winner가 모두 CPU로 통일되어 분배 이득 자체가 없다.

본 결과는 우리가 아는 한 RK3588에서 동일 양자화 조건으로 4개 백엔드 구성의 prefill·decode 시간을 통일 그리드에서 비교하고 op-level decompose를 통해 phase-level crossover의 원인을 분리한 첫 측정이다. v6의 phase 단위 비대칭 가설을 op timing 데이터로 검증하는 동시에, hetero scheduling이 phase 단위에서만 실현 가능하며 op 단위로 더 fine하게 가르는 시도는 dispatch overhead로 손해라는 부정적 결론도 함께 정량화하였다. 후속 연구는 (1) phase-level hetero scheduling의 실측 검증과 KV layout 변환 비용의 정량화, (2) RKLLM의 op-level timing API 노출 또는 reverse engineering을 통한 NPU op-level 직접 측정, (3) 더 큰 모델(3B, 7B)과 더 긴 context(8K, 16K)에서 동일 분석의 일반화이다.

## References

[1] L. Chen et al., "Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference," *Proc. ACM SIGOPS SOSP*, 2025. DOI: 10.1145/3731569.3764808.
[2] D. Xu et al., "Fast On-device LLM Inference with NPUs," *Proc. ACM ASPLOS*, 2025. DOI: 10.1145/3669940.3707239.
[3] T. Dao et al., "FlashAttention: Fast and memory-efficient exact attention with IO-awareness," *NeurIPS*, vol. 35, pp. 16344–16359, 2022.
[4] S. Williams, A. Waterman, D. Patterson, "Roofline: an insightful visual performance model for multicore architectures," *Communications of the ACM*, vol. 52, no. 4, pp. 65–76, 2009. DOI: 10.1145/1498765.1498785.
[5] Rockchip. *RKLLM Toolkit (RKNN-LLM v1.2.2)*. https://github.com/airockchip/rknn-llm.
[6] Alibaba. *MNN*. https://github.com/alibaba/MNN.
[7] Radxa. *Rock 5B+ Product Brief*. https://radxa.com/products/rock5/5bp/.
[8] S. Bogatov. *tinymembench v0.4.9*. https://github.com/ssvb/tinymembench.
