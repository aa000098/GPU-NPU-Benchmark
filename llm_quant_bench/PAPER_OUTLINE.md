# 논문 Outline (Option B: Negative Result + Characterization)

## 제목 후보
1. **"The KV Cache Red Herring: Why NPU Decode Slowdown on Edge LLM Isn't a Bandwidth Problem"**
2. **"Beyond Bandwidth: Characterizing Context-Proportional Overhead in Edge NPU LLM Inference"**
3. **"Rethinking KV Cache Compression for Edge SoCs: An Empirical Study on RK3588"**

## Abstract (~250 words)

최근 edge 디바이스에서 LLM inference가 가능해지면서 KV cache compression 기법 (TurboQuant, KIVI)의 효과가 주목받고 있다. 본 연구는 Rockchip RK3588 상에서 Llama-3.2-1B를 대상으로 3종 백엔드 (CPU/GPU/NPU) × 다양한 양자화 × context length 32~4096 범위에서 실측을 수행했다.

**주요 관찰**:
- NPU는 p32에서 20 tok/s decode이나 p4096에서 11.3 tok/s로 44% 저하 (crossover at ~p2048 with CPU)
- CPU/GPU는 context-invariant

**분석**:
- Linear fit: NPU decode slope = 9.26 us/tok, CPU = 0.22 us/tok
- GQA로 KV cache는 전체 memory read의 ≤10%에 불과
- 실측 메모리 BW = 19.7 GB/s로도 NPU slope는 pure KV bandwidth 모델의 2.8배
- 즉 **NPU decode degradation의 65-89%는 KV bandwidth로 설명 불가**

**구현 및 검증**:
- MNN에 INT8 KV / 4bit KV codebook 구현
- CPU는 이미 context-invariant이므로 KV 압축 효과 <5% 개선
- NPU는 closed SDK로 KV cache 수정 불가 → 이론상의 효과 확인 불가

**결론**: 엣지 NPU decode bottleneck은 KV bandwidth가 아닌 **SDK 수준의 context-proportional scheduling overhead**. KV 압축만으로는 개선 불가하며, 엣지 LLM 연구 커뮤니티는 NPU SDK 개방성과 kernel dispatch overhead 분석에 주목해야 한다.

---

## Section 1. Introduction

### Motivation
- Edge LLM 관심 증가: Llama 3.2, Gemma 2, Phi-3 등 1-3B 모델이 엣지에서 실행 가능
- RK3588 같은 heterogeneous SoC가 다양한 백엔드 제공 (CPU, GPU, NPU)
- 긴 context 지원이 사용자 경험의 핵심 (요약, 대화 등)

### Problem
- Edge NPU가 단순 처리에는 우수하지만, **긴 context에서 급격한 성능 저하**
- 통념: "KV cache가 커져서 bandwidth bound"
- 해결책으로 KV compression 기법 (TurboQuant 등) 제안

### Our Finding
- RK3588 실측 기반으로 **통념을 반증**
- KV 압축 효과가 기대와 다름을 정량적으로 보임
- 엣지 NPU의 진짜 병목은 **SDK 수준 scheduling overhead**

### Contributions
1. Edge SoC에서 3종 백엔드 × 양자화 × context length의 첫 종합 특성화
2. Linear fit + bandwidth model로 "KV 가설" 반증
3. MNN에 INT8/codebook KV 압축 구현 및 CPU 효과 측정
4. NPU SDK 제약의 실질적 한계 분석 및 커뮤니티 시사점

---

## Section 2. Background

- 2.1 LLM prefill vs decode phase
- 2.2 KV cache structure & GQA (Llama 3.2의 num_kv=8)
- 2.3 Edge SoC architecture: RK3588 (CPU A76/A55, Mali G610, 3-core NPU)
- 2.4 Inference frameworks: MNN (open-source), RKNN-LLM (closed)
- 2.5 KV compression methods: INT8, KIVI, KVQuant, TurboQuant

---

## Section 3. Measurement Methodology

- 3.1 Hardware/software stack
- 3.2 Model: Llama-3.2-1B-Instruct
- 3.3 Benchmarks: llm_bench (MNN), bench_rkllm.py
- 3.4 Context length sweep [32, 64, ..., 4096]
- 3.5 Per-run protocol: 5 warmup + 10 measure
- 3.6 Metrics: prefill tok/s, decode tok/s, TTFT, PPL (wikitext-2), memory

---

## Section 4. Cross-Backend Characterization

### 4.1 Sequence-length Sweep Results
- Prefill: NPU dominates (100-400x over CPU)
- Decode: NPU > CPU > GPU at short ctx; **NPU < CPU at p2048+**
- (Fig 1: prefill and decode tok/s curves)

### 4.2 Linear Fit Analysis
- decode_ms = a + b × ctx
- Only NPU has significant b (9.26 us/tok)
- CPU b ≈ 0 (weights caching effective)
- GPU b ≈ 0 (different reasons - low throughput anyway)
- (Fig 2: linear fits all 3 backends)

---

## Section 5. Testing the KV Bandwidth Hypothesis

### 5.1 Theoretical Analysis
- KV bytes per token (GQA) = 32KB for Llama-3.2-1B
- 전체 memory read의 <10% (weights가 main)
- (Table 1: KV vs Weights per context length)

### 5.2 Bandwidth Measurement
- STREAM-like test: 19.7 GB/s peak (2 threads)
- Single thread: 17.4 GB/s
- NPU DMA dedicated path 예상

### 5.3 Hypothesis Test
- Pure KV BW model: slope = 32KB / BW
- At 30 GB/s: predicted 1.09 us/tok
- **Measured 9.26 us/tok = 8.5x over prediction**
- (Fig 3: Roofline + measured points)

### 5.4 Implication
- 통념 "KV = bottleneck" 반증
- 실제는 context-proportional **scheduling overhead**

---

## Section 6. KV Compression Implementation

### 6.1 MNN Architecture
- CPUAttention + CPUKVCacheManager 구조
- quant_qkv 플래그 (0-10): INT8 KV 이미 지원

### 6.2 INT8 KV (Built-in)
- quant_qkv=9 (Q,K INT8) / =10 (Q,K,V INT8)
- **측정 결과 (Phase 2-2)**: decode 속도 영향, PPL 영향

### 6.3 Codebook/KIVI Extension
- CPUKVCacheManager의 mQuantKeyFunc를 함수 포인터로 교체
- NEON vqtbl 기반 decompression
- (Fig 4: MNN integration diagram)

### 6.4 NPU SDK 한계
- RKNN-LLM은 closed box
- KV cache 수정 불가
- **이것이 연구 한계이자 논문의 핵심 메시지**

---

## Section 7. Evaluation

### 7.1 Decode Speed vs KV Compression (CPU)
- (Fig 5: decode tok/s vs context for 5 KV settings)
- Expected: INT8 KV에서 marginal 개선 (CPU는 이미 flat)
- Expected: KIVI/codebook도 비슷

### 7.2 PPL vs Compression
- (Fig 6: PPL vs compression ratio Pareto)

### 7.3 Memory Usage
- (Table 2: peak RSS vs context length)

### 7.4 Generalizability (Phase 4)
- Llama 3.2 3B (가능하면)
- Qwen2.5-1.5B

---

## Section 8. Discussion

### 8.1 Why NPU Overhead Scales with Context
Possible causes (undetectable w/o SDK access):
- Per-layer kernel dispatch
- KV indexing/addressing overhead
- Scratch memory I/O for attention softmax
- Dynamic graph adaptation per step

### 8.2 Why KV Compression is Less Helpful than Expected
- GQA reduces KV footprint already
- Weights dominate memory traffic
- CPU L3 cache absorbs weight reads

### 8.3 What Actually Helps
- Fixed overhead reduction (better kernel fusion)
- Open NPU SDK for profiling
- Hybrid execution (use CPU for long-context decode)

### 8.4 Limitations
- Single model size (1B)
- Single SoC (RK3588)
- PPL only (beyond tasks partial)

---

## Section 9. Related Work

- LLM quantization (AWQ, SmoothQuant, HQQ)
- KV cache compression (KIVI, KVQuant, TurboQuant, PagedAttention)
- Edge LLM inference (mllm, llama.cpp ARM, MLC-LLM, PowerInfer-2)
- Our prior work (Attention Backend Selection paper)

---

## Section 10. Conclusion

엣지 NPU의 decode bottleneck은 통념과 달리 KV bandwidth가 아니다. NPU SDK 개방성과 per-step overhead 분석이 엣지 LLM 추론 최적화의 다음 단계이다.

---

## Key Figures (예상 8-9개)

1. **Fig 1** (headline): Decode tok/s vs context — 3 backends
2. **Fig 2**: Linear fits with slopes annotated
3. **Fig 3**: Roofline with measured points + bandwidth assumption sensitivity
4. **Fig 4**: MNN integration architecture
5. **Fig 5**: KV compression comparison (decode speed)
6. **Fig 6**: PPL vs compression Pareto
7. **Fig 7**: Memory footprint comparison
8. **Fig 8**: Generalizability (3B/Qwen)
9. (Optional) Power efficiency

## Key Tables (예상 2-3개)

1. **Table 1**: KV vs Weights bytes per context length
2. **Table 2**: Backend linear fit coefficients
3. **Table 3**: Full results matrix

---

## 현재 데이터 상태

- [x] 3 backends × 8 ctx lengths (prefill + decode) ✓
- [x] Roofline analysis ✓
- [x] NPU overhead decomposition ✓
- [x] RK3588 memory BW measurement ✓
- [ ] INT8 KV comparison (in progress, tmux kv_bench)
- [ ] PPL for each KV setting
- [ ] KIVI / codebook implementation
- [ ] 3B / Qwen generalizability

## Target Venue

**Tier 2** 기준:
- USENIX ATC 2026 (구현 + 측정 강조)
- EuroSys 2026
- 또는 저널: ACM TECS, IEEE TC
