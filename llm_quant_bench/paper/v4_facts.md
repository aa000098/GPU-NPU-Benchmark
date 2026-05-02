# 논문 작성용 사실/실험결과 노트

본 문서는 v4 논문 작성에 활용할 수 있는 측정 사실과 외부 reference를 모은 것이다. 의견·추측은 배제하고 측정값과 reference만 적는다.

---

## 1. 실험 환경

**보드**: Radxa Rock 5B+ (Rockchip RK3588 SoC, LPDDR5 16 GB)
- LPDDR5 메모리: 64-bit 폭, 5500 MT/s, 이론 peak bandwidth = 5500 × 8 = **44 GB/s** [9]
- NPU: 6 TOPS, 3 cores, RKLLM SDK v1.2.1b1 (closed-source)
- CPU: Cortex-A76 ×4 (big core, 측정에 사용) + Cortex-A55 ×4 (little core, 측정 미사용)
- GPU: ARM Mali-G610 MP4, MNN OpenCL backend [3]

**모델**: Meta Llama-3.2-1B-Instruct
- 파라미터 수: 1.24B
- W8A8 동일 양자화 (weight 8-bit, activation 8-bit)로 NPU/CPU/GPU 세 backend에 통일 변환
- weight 크기: 1.24B params × 1 byte = **1.24 GB** (decode 한 step에 읽어야 하는 양)

**SDK 버전**:
- RKLLM (NPU): rkllm-runtime v1.2.1b1, rknpu driver v0.9.8 [4]
- MNN (CPU/GPU): build 시점 main branch, llm_bench 도구 사용
- llama-cpp-python: v0.2.x (참고용 Q4_0 측정에만 사용)

**측정 옵션 (모든 측정 통일)**:
- CPU thread: 4 (A76 big core에 affinity pinning)
- MNN llm_bench 인자: `-rep 3` (3회 반복 평균), `-kv true` (KV cache 활성화)
- NPU 측정: 자체 Python 스크립트로 KV cache reuse 사용, `-rep 3` 통일
- 컨텍스트 길이: *n* ∈ {15, 512, 1024, 2048, 3500}
- 생성 길이: n_gen=32 (기본), long-generation 측정엔 n_gen ∈ {32, 128, 256, 512}
- 입력 prompt: Wikitext-2 train split의 자연어 산문, 위 5개 길이로 truncate

---

## 2. 메모리 대역폭 실측 (tinymembench v0.4.9 [5])

Rock 5B+에서 tinymembench를 A76 big core 4개에 affinity pin하여 실행.

| 측정 항목 | 값 |
|---|---:|
| C copy backwards | 11832.0 MB/s |
| C copy | 12263.4 MB/s |
| C copy prefetched (32 bytes step) | 12630.9 MB/s |
| C copy prefetched (64 bytes step) | 12626.2 MB/s |
| C 2-pass copy | 5169.4 MB/s |
| C 2-pass copy prefetched | 10419.0 MB/s |
| C fill | 27308.3 MB/s |
| standard memcpy | 12629.6 MB/s |
| standard memset | 27367.8 MB/s |
| NEON LDP/STP copy (no prefetch) | 12628.2 MB/s |
| NEON LDP/STP copy + pldl2strm | 12909.6 MB/s |
| NEON STP fill | 27330.7 MB/s |
| NEON STNP fill | 27432.4 MB/s |

**copy throughput peak (단방향 환산 어려움)**: 12.91 GB/s (NEON copy + pldl2strm, read+write 동시)
**fill throughput peak (단방향, write only)**: 27.43 GB/s

본 논문은 write-only fill을 단일 방향 메모리 처리량의 conservative upper bound proxy로 사용한다. 이 값을 LPDDR5 이론 peak 44 GB/s와 비교하면 utilization은 27.43/44 = **62%**다.

memory latency 결과:
- 4 KB block random read: 0.0 ns (L1 cache)
- 64 MB block random read: 239.1 ns (DRAM)
- raw log: `llm_quant_bench/results/raw/tinymembench.log`

---

## 3. Roofline 도출

Williams roofline 모델 [6]에 따라, decode가 weight read로 dominant하게 결정될 때:

```
tok/s_max = B_read / S_weight
```

본 환경에 대입:
```
B_read = 27.43 GB/s (tinymembench fill)
S_weight = 1.24 GB (Llama-3.2-1B W8A8)
tok/s_max = 27.43 / 1.24 ≈ 22.0 tok/s
```

이는 어떤 single-prompt single-batch decode 측정도 넘을 수 없는 상한이다.

---

## 4. NPU Baseline 측정 (RKLLM, KV reuse incremental decode)

5 ctx × 3-rep 측정. 각 ctx에서 prefix prefill 1회 후 n_gen=32 토큰을 incremental decode (clear_kv_cache 없이 새 토큰 1개씩 추가). 결과 mean ± std (3 repeats).

| ctx | TTFT (prefill) ms | decode ms/tok | decode tok/s | overall tok/s |
|---:|---:|---:|---:|---:|
| 15 | 74 ± 2 | 51.0 ± 1.4 | 19.63 ± 0.54 | 18.75 ± 0.51 |
| 512 | 1929 ± 133 | 65.8 ± 1.2 | 15.20 ± 0.28 | 7.94 ± 0.27 |
| 1024 | 4227 ± 235 | 83.7 ± 1.5 | 11.95 ± 0.22 | 4.64 ± 0.14 |
| 2048 | 10317 ± 325 | 116.9 ± 1.2 | 8.56 ± 0.09 | 2.28 ± 0.06 |
| 3500 | 22242 ± 557 | 166.2 ± 1.0 | 6.02 ± 0.04 | 1.16 ± 0.02 |

**Roofline 22.0 tok/s 대비 효율 (decode tok/s 기준)**:
- ctx=15: 19.63 / 22.0 = **89%**
- ctx=512: 15.20 / 22.0 = 69%
- ctx=1024: 11.95 / 22.0 = 54%
- ctx=2048: 8.56 / 22.0 = 39%
- ctx=3500: 6.02 / 22.0 = **27%**

raw 데이터: `llm_quant_bench/results/raw/npu_only_kvreuse_v3.json`

### 4.1 NPU 토큰별 decode 시간 안정성 검증

prefill 후 8개 토큰을 한 번에 1 토큰씩 incremental decode하여 토큰별 시간 측정. ctx 내에서 토큰 시간이 일정한지(KV reuse 정상 작동) 확인.

| ctx | 토큰별 시간 (ms) | mean ± std |
|---:|:---|:---|
| 15 | 49.5, 51.5, 47.9, 49.3, 49.0, 49.1, 49.6, 69.3 | 51.9 ± 6.7 |
| 1024 | 99.0, 78.8, 79.0, 79.5, 79.1, 79.4, 79.0, 78.9 | 81.6 ± 6.6 |
| 3500 | 182.6, 155.8, 153.5, 204.1, 157.1, 155.7, 153.6, 153.2 | 164.5 ± 17.6 |

각 ctx에서 첫 토큰의 cold-start 영향을 제외하면 토큰별 시간이 안정적임이 확인된다 (예: ctx=1024는 token 1~7에서 78.8–79.5 ms 범위). 이는 토큰 추가에 따라 시간이 점진적으로 증가하지 않음, 즉 KV cache가 토큰별 재계산 없이 reuse됨을 보여준다.

raw 데이터: `llm_quant_bench/results/raw/probe_npu_attention.log`

---

## 5. CPU MNN W8A8 측정 (Cortex-A76 4 cores)

MNN llm_bench (`-a cpu -t 4 -kv true -rep 3`)로 5 ctx 측정. CPU affinity는 A76 big core 4개(`taskset -c 4-7`).

| ctx | prefill (tok/s) | decode (tok/s) | TTFT (sec) |
|---:|---:|---:|---:|
| 15 | 74.47 ± 1.85 | 16.18 ± 0.12 | 0.20 |
| 512 | 144.37 ± 0.29 | 15.01 ± 0.15 | 3.55 |
| 1024 | 126.48 ± 0.42 | 14.80 ± 0.19 | 8.10 |
| 2048 | 94.19 ± 0.50 | 13.84 ± 0.08 | 21.74 |
| 3500 | 91.09 ± 0.16 | 13.05 ± 0.10 | 38.42 |

decode tok/s가 ctx에 약하게 의존하나 안정적인 13–16 tok/s를 유지한다.

raw 데이터: `llm_quant_bench/results/raw/mnn_cpu_ctx*.txt`

---

## 6. Mali GPU OpenCL MNN W8A8 측정

MNN llm_bench (`-a opencl -t 4 -kv true -rep 3`)로 5 ctx 측정.

| ctx | prefill (tok/s) | decode (tok/s) | TTFT (sec) |
|---:|---:|---:|---:|
| 15 | 55.79 ± 0.06 | 14.93 ± 0.04 | 0.27 |
| 512 | 178.22 ± 6.32 | 13.76 ± 0.06 | 2.87 |
| 1024 | 171.01 ± 0.25 | 13.04 ± 0.02 | 5.99 |
| 2048 | 142.82 ± 0.97 | 12.68 ± 0.04 | 14.34 |
| 3500 | 121.42 ± 0.82 | 11.41 ± 0.77 | 28.83 |

raw 데이터: `llm_quant_bench/results/raw/mnn_opencl_ctx*.txt`

---

## 7. Multi-backend 비교 표 (overall tok/s, n_gen=32)

| ctx | NPU (RKLLM) | CPU MNN | OpenCL Mali |
|---:|---:|---:|---:|
| 15 | 19.17 | 14.68 | 13.27 |
| 512 | 7.87 | 5.64 | 6.16 |
| 1024 | 4.54 | 3.12 | 3.79 |
| 2048 | 2.19 | 1.33 | 1.90 |
| 3500 | 1.13 | 0.78 | 1.01 |

n_gen=32에서 모든 ctx에서 NPU가 1등이다. 단 격차는 ctx가 커질수록 좁아진다 (ctx=3500에서 NPU/CPU 비율 1.45×).

decode tok/s만 비교하면 (prefill 비용 제외):

| ctx | NPU | CPU MNN | OpenCL Mali |
|---:|---:|---:|---:|
| 15 | **20.80** | 16.18 | 14.93 |
| 512 | **15.09** | 15.01 | 13.76 |
| 1024 | 12.20 | **14.80** | 13.04 |
| 2048 | 8.46 | **13.84** | 12.68 |
| 3500 | 6.12 | **13.05** | 11.41 |

decode 영역에서는 ctx=1024부터 CPU MNN이 NPU를 능가하며, ctx=3500에서 CPU/NPU = **2.13×**.

---

## 8. NPU의 ctx-비례 추가 비용 정량화

NPU 토큰당 decode 시간을 두 항으로 분해:
- weight read (메모리 BW 한계로 결정): 1.24 GB / 27.43 GB/s = **45 ms**, ctx에 무관
- ctx-비례 추가 비용: 측정값 - 45 ms

| ctx | NPU dec ms/tok 측정 | weight 한계 (45) | ctx 비례 추가 |
|---:|---:|---:|---:|
| 15 | 51.0 | 45 | 6 |
| 512 | 65.8 | 45 | 21 |
| 1024 | 83.7 | 45 | 39 |
| 2048 | 116.9 | 45 | 72 |
| 3500 | 166.2 | 45 | 121 |

ctx=15→3500 구간에서 추가 비용 증가분은 121 - 6 = **115 ms**이다.

같은 구간의 KV-cache read 메모리 비용 한계는 다음과 같이 계산된다:
- Llama-3.2-1B는 16 layers × 32 heads × 64 head_dim × 2 (K+V) × 1 byte = 65.5 KB / token
- ctx=3500의 KV size = 65.5 KB × 3500 = 229 MB
- 27.43 GB/s read에서 229 MB read는 **8.4 ms**

NPU 실측 추가분 115 ms는 KV-read 한계 8.4 ms의 **13.7×**다.

같은 구간(ctx=15→3500)에서 CPU MNN의 추가 비용은 (1000/13.05 - 1000/16.18) ≈ 14.8 ms로, KV-read 한계 8.4 ms의 1.76×다.

raw 데이터: `llm_quant_bench/results/raw/npu_only_kvreuse_v3.json`, `mnn_cpu_ctx*.txt`

---

## 9. Long Generation Cross-over (ctx=3500, n_gen 변화)

ctx=3500 고정, n_gen ∈ {32, 128, 256, 512} 변화. 각 backend 사이 5분 cooldown 후 -rep 3 평균.

| n_gen | NPU (tok/s) | CPU MNN (tok/s) | OpenCL Mali (tok/s) |
|---:|---:|---:|---:|
| 32 | **1.146 ± 0.026** | 0.788 | 1.031 |
| 128 | 2.996 ± 0.006 | 2.648 | **3.323** |
| 256 | 4.063 ± 0.002 | 4.385 | **5.117** |
| 512 | 4.912 ± 0.008 | 6.519 | **7.154** |

**관찰된 cross-over**:
- n_gen=32: NPU 1등 (1.146)
- n_gen=128: GPU 1등 (3.323) — GPU가 NPU 추월
- n_gen=256: GPU 1등 (5.117), CPU(4.385) > NPU(4.063) — CPU도 NPU 추월
- n_gen=512: GPU > CPU > NPU 순서 (7.154 > 6.519 > 4.912), GPU/NPU = 1.46×, CPU/NPU = 1.33×

NPU 측정의 prefill/decode 분해:
| n_gen | prefill ms | decode ms/tok |
|---:|---:|---:|
| 32 | 22768 ± 663 | 164.5 ± 1.4 |
| 128 | 22484 ± 98 | 159.3 ± 0.1 |
| 256 | 22463 ± 36 | 158.9 ± 0.1 |
| 512 | 22568 ± 135 | 159.7 ± 0.3 |

NPU prefill은 n_gen 무관하게 ~22.5 sec 일정, decode는 ~160 ms/tok 일정. n_gen이 커질수록 prefill이 더 많은 토큰에 분할 상각되어 overall tok/s가 증가한다 (1.146 → 4.912).

CPU MNN 측정:
| n_gen | prefill (tok/s) | decode (tok/s) |
|---:|---:|---:|
| 32 | 91.8 | 12.88 |
| 128 | 91.2 | 12.88 |
| 256 | 91.1 | 12.82 |
| 512 | 91.0 | 12.77 |

CPU prefill 91 tok/s, decode 12.8 tok/s 모두 n_gen 무관하게 안정.

GPU OpenCL 측정값은 raw 데이터 `llm_quant_bench/results/raw/longgen_mnn_opencl_n*.txt` 참조. prefill 130–172 tok/s, decode 11–13 tok/s 범위.

raw 데이터: `llm_quant_bench/results/raw/long_generation_v4.json`

---

## 10. TTFT (Time To First Token, 인터랙티브 SLO 비교)

`overall = n_gen / (TTFT + n_gen × decode_time/tok)`에서 TTFT는 prefill 시간이다. 인터랙티브 chatbot의 TTFT SLO는 ≤500 ms로 일반적으로 권고된다 [7].

| ctx | NPU TTFT | CPU MNN TTFT | OpenCL Mali TTFT | SLO 충족? |
|---:|---:|---:|---:|:---:|
| 15 | 0.128 s | 0.201 s | 0.269 s | NPU만 OK |
| 512 | 1.95 s | 3.55 s | 2.87 s | 모두 위반 |
| 1024 | 4.42 s | 8.10 s | 5.99 s | 모두 위반 |
| 2048 | 10.85 s | 21.74 s | 14.34 s | 모두 위반 |
| 3500 | 23.06 s | 38.42 s | 28.83 s | 모두 위반 |

본 환경에서 ctx=512 이상에서는 어떤 backend도 인터랙티브 chatbot SLO를 충족하지 못한다.

---

## 11. 외부 RK3588 / Mali GPU LLM 추론 보고 (직접 비교용)

### 11.1 RK3588 NPU LLM 추론

- TinyLlama 1.1B Q8 on RK3588 NPU: **10–15 tok/s** (short ctx, n_gen 미명시) [8]
- TinyLlama 1.1B on Radxa ROCK 5C (RK3582): 17.67 tok/s
- Qwen 2.5 3B Instruct on ODROID-M2 (RK3588): ~8.45 tok/s

본 측정 비교 (Llama-3.2-1B W8A8):
- ctx=15에서 NPU decode 19.63 tok/s, overall 19.17 tok/s
- ctx=3500/n_gen=512에서 NPU overall 4.91 tok/s

### 11.2 Mali-G610 GPU LLM 추론

- mlc-llm Llama-3 8B (Q4) on Orange Pi 5 (RK3588 + Mali-G610): **~2 tok/s** [10]
- llama.cpp Mali OpenCL: CPU 대비 일반적으로 느림 (사용자 보고)

본 측정 비교 (Mali-G610, MNN W8A8, Llama-3.2-1B):
- ctx=15 decode 14.93 tok/s, overall 13.27 tok/s
- ctx=3500/n_gen=512 overall 7.15 tok/s

### 11.3 인터랙티브 LLM TTFT/throughput 권고

[7] BentoML LLM Inference Handbook:
- chatbot TTFT ≤ 500 ms (interactive 응답성)
- 일반 chat throughput ≥ 10 tok/s acceptable
- code completion ≥ 25 tok/s 권장

### 11.4 인접 SoC의 spec decoding 적용 publication (RK3588 외)

- HeteroLLM (Chen et al., SOSP'25 [1]): Qualcomm Snapdragon 환경 GPU+NPU 분담, RK3588 미평가
- Fast On-device LLM Inference with NPUs (Xu et al., ASPLOS'25 [2]): mobile NPU + chunked prefill, RK3588 미평가

본 연구를 우리가 아는 한, **Rock 5B+ (RK3588) 환경에서 동일 양자화로 NPU/CPU/GPU 세 backend를 직접 비교한 publication은 없다**.

---

## 12. 측정 방법론 요약

- **모델 통일**: Llama-3.2-1B-Instruct, W8A8 양자화로 NPU/CPU/GPU 모두 변환
- **Prompt 통일**: Wikitext-2 자연어 prefix
- **반복 수 통일**: -rep 3 (3회 평균 + std)
- **Cooldown**: long-generation 측정에서 backend 사이 5분 idle (thermal sensor 38°C 부근 회복 확인)
- **CPU pinning**: A76 big core 4개로 모든 측정 통일 (`taskset -c 4-7`)
- **KV cache**: 모든 측정에서 활성 (`-kv true` 또는 RKLLM `keep_history=1` + clear 안 함)

---

## 13. 정량 핵심 수치 (논문 인용용 요약)

| 항목 | 값 |
|---|---:|
| LPDDR5 이론 peak | 44 GB/s |
| tinymembench fill (단방향 read proxy) | 27.43 GB/s |
| Decode roofline (식 1) | 22.0 tok/s |
| NPU short ctx decode (ctx=15) | 19.63 ± 0.54 tok/s (roofline 89%) |
| NPU long ctx decode (ctx=3500) | 6.02 ± 0.04 tok/s (roofline 27%) |
| NPU ctx-비례 추가 비용 (15→3500 구간) | 115 ms |
| KV-read 메모리 한계 (15→3500 구간) | 8.4 ms |
| NPU 추가 비용 / KV-read 한계 비율 | 13.7× |
| CPU MNN decode 안정 영역 | 13.0–16.2 tok/s |
| OpenCL Mali decode 안정 영역 | 11.4–14.9 tok/s |
| Decode crossover 1 (NPU vs CPU MNN) | ctx=1024 |
| ctx=3500 NPU vs CPU MNN decode 비율 | CPU 2.13× faster |
| Long-gen crossover 1 (NPU vs GPU) | n_gen=128 (3.323 vs 2.996) |
| Long-gen crossover 2 (NPU vs CPU) | n_gen=512 (6.519 vs 4.912) |
| ctx=3500 n_gen=512 GPU/NPU 비율 | 1.46× |
| ctx=3500 n_gen=512 CPU/NPU 비율 | 1.33× |
| 인터랙티브 SLO 충족 ctx 영역 | NPU ctx=15만 (TTFT 128 ms) |

---

## 14. References (논문 reference 후보)

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

---

## 15. 측정 raw 데이터 위치 (논문 보충자료 / 재현용)

- 메모리 BW: `llm_quant_bench/results/raw/tinymembench.log`
- NPU baseline (5 ctx × 3 rep): `llm_quant_bench/results/raw/npu_only_kvreuse_v3.json`, `npu_only_kvreuse_v4repeat.log`
- NPU per-token 안정성: `llm_quant_bench/results/raw/probe_npu_attention.log`
- CPU MNN (5 ctx): `llm_quant_bench/results/raw/mnn_cpu_ctx{15,512,1024,2048,3500}.txt`
- OpenCL Mali (5 ctx): `llm_quant_bench/results/raw/mnn_opencl_ctx{15,512,1024,2048,3500}.txt`
- Long-gen cross-over: `llm_quant_bench/results/raw/long_generation_v4.json`, `longgen_v4_cool_*.log`
- 측정 스크립트:
  - `llm_quant_bench/benchmark/measure_npu_kvreuse.py`
  - `llm_quant_bench/benchmark/measure_long_generation.py`
  - `llm_quant_bench/benchmark/measure_long_generation_gpu.py`
  - `llm_quant_bench/benchmark/probe_npu_attention.py`
  - `llm_quant_bench/scripts/run_mnn_baseline.sh`
  - `llm_quant_bench/scripts/run_longgen_cooldown.sh`
- 정리 CSV:
  - `llm_quant_bench/paper/v4_data/v4_multibackend.csv`
  - `llm_quant_bench/paper/v4_data/v4_longgen.csv`
  - `llm_quant_bench/paper/v4_data/v4_roofline.csv`
- 그림 (PDF + PNG):
  - `llm_quant_bench/paper/v4_figures/fig_roofline.{pdf,png}`
  - `llm_quant_bench/paper/v4_figures/fig_backends.{pdf,png}`
  - `llm_quant_bench/paper/v4_figures/fig_longgen.{pdf,png}`
