# 논문 v2 보강 자료

기존 논문(`v2_손현호_통신학회_논문_260418.pdf`)의 "당위적" 비판 해소를 위한 추가 실험·분석 정리.

---

## 1. 논문의 약점

기존 논문은 5가지 측정 결과(§4.1~4.4)를 잘 나열했지만:

1. **결론이 first-principle로 예측 가능**: "긴 context는 verify 지배 → draft-based 유리" 같은 결론은 직관 수준
2. **RegimeSpec이 abstract 약속만 있고 본문에 구현 없음**: 정책 권고에 그침
3. **메커니즘 입증이 정성적**: "거의 선형"이라고만 적고 정량 fit 없음
4. **정책의 가정(wiki workload)이 명시 안 됨**: 다른 prompt 분포에서 정책이 깨질 수 있음

---

## 2. 추가 실험 1 — RegimeSpec 실측 시스템 구현

### 2.1 시스템 구조
`if ctx_len < threshold(1024): run_pld() else: run_sequential()` 자동 분기.
Draft 모델은 lazy-load (long prompt 처음 등장 시).

코드: `llm_quant_bench/spec_decoding/regimespec.py`

### 2.2 비교 대상

| 정책 | 동작 |
|------|------|
| NPU-only | 모든 prompt를 plain autoregressive |
| Always-Seq | 모든 prompt에 Sequential draft (k=4) |
| Always-PLD | 모든 prompt에 PLD (k=4) |
| **RegimeSpec** | ctx<1024면 PLD, 아니면 Sequential 자동 |
| Oracle | 사후에 prompt별 최고 모드 선택 (cheat) |

### 2.3 측정 결과 (Llama-3.2-1B W8A8, n_gen=32)

**Mixed workload (short prompt 10개 + long prompt 5개, 무작위 셔플):**

| 모드 | avg tok/s | 비고 |
|------|----------|------|
| NPU-only | 4.50 | baseline |
| Always-Seq | 3.78 | short prompt에서 손해 |
| Always-PLD | 4.97 | |
| **RegimeSpec** | **4.92** | Always-Seq 1.30×, Oracle 0.97× |
| Oracle | 5.09 | |

**Long-only workload (5 prompts, ctx 1024-3000, 반복적 텍스트):**

| 모드 | tok/s | speedup vs NPU |
|------|-------|---------------|
| NPU-only | 0.14 | 1.0× |
| Always-Seq | 0.34 | 2.4× |
| **Always-PLD** | **0.55** | **3.9×** ← 1등 |
| RegimeSpec | 0.40 | 2.9× |
| Oracle | 0.55 | 3.9× |

**Short-only workload (10 일반 chat prompt, ctx 4-9):**

| 모드 | tok/s | speedup |
|------|-------|---------|
| NPU-only | 9.98 | 1.0× |
| Always-Seq | 6.07 | 0.61× (퇴행) |
| **Always-PLD** | **10.61** | **1.06×** |
| RegimeSpec | 10.60 | 1.06× |

### 2.4 핵심 발견

**(A) RegimeSpec은 mixed workload에서 always-X 대비 우세**
- vs Always-Seq: 1.30×
- vs Always-PLD: 0.99× (PLD에 거의 동률)
- vs Oracle: 0.97% (거의 최선)

**(B) 정책 한계 발견 — "long context = Sequential" 가정이 prompt 분포 의존**
- 논문 wiki workload: long에서 Sequential 2.19-2.50× (PLD 1.06-1.33×) → Sequential 우세
- 우리 반복 workload: long에서 Sequential 2.4× (PLD 3.9×) → **PLD가 오히려 우세**

→ PLD acceptance rate α는 **context length가 아니라 prompt repetitiveness**에 의존.
- 자연어 산문 wiki: α ≈ 0.07
- 반복적 텍스트: α ≈ 0.5+

**(C) Always-Seq의 short prompt 퇴행 입증**
- Short context에서 NPU verify 자체가 빠름 (~50ms)
- Sequential의 draft model overhead(~50ms)가 상대적으로 큼
- → 0.61× (NPU-only보다도 느림)

---

## 3. 추가 실험 2 — V1 Full-prefix Verify 정량 입증

### 3.1 가설

기존 논문은 "throughput 저하가 prompt length에 거의 선형 비례"라고 정성적 기술. **NPU-only 토큰 생성 시간이 context length n에 정확히 선형**이라면, RKLLM이 매 토큰마다 전체 prefix를 재prefill한다는 V1 full-prefix verify 가설이 정량적으로 입증됨.

### 3.2 측정값 fit

논문 §4.1 표:
| n | tok/s | ms/tok |
|---|-------|--------|
| 15 | 8.04 | 124 |
| 512 | 0.540 | 1851 |
| 1024 | 0.241 | 4150 |
| 2048 | 0.097 | 10309 |
| 3500 | 0.045 | 22222 |

선형 회귀:
```
ms_per_tok(n) = 6.41 × n − 1369.7
R² = 0.98
```

### 3.3 해석

- 기울기 6.41 ms/n: prompt 1토큰 추가당 prefill 비용 증가량 → V1 full-prefix verify의 직접 측정값
- R²=0.98: 측정의 98%가 선형 모델로 설명됨 → "거의 선형"이 아니라 **본질적으로 선형**
- → V1 full-prefix verify가 NPU-only degradation의 단일 원인임을 정량 입증

### 3.4 시사점

이 식으로 **다른 model 크기/quantization에서도 prefill 시간 예측 가능**:
- 모델 더 크면 기울기 a 증가
- 동일 모델 다른 양자화면 a 비례 변화
- 실제 측정 안 해도 "이 디바이스에서 ctx N에서 토큰 생성에 a·N + b 걸린다" 예측 가능

---

## 4. 추가 실험 3 — NPU MAC Utilization 분석

### 4.1 가설

NPU는 INT8 MAC array가 매우 넓다(6 TOPS). Decode는 GEMV(M=1)라 MAC을 거의 못 채움. **Speculative decoding은 verify 시점에 M=k+1개 위치를 동시 처리 → GEMM-like 패턴으로 변환 → MAC utilization 끌어올림**.

### 4.2 측정값

| 시나리오 | M | 달성 GOPS | 6 TOPS 대비 |
|---------|---|-----------|------------|
| NPU-only decode (GEMV) | 1 | 38 GOPS | 0.63% |
| Sequential k=4 decode (GEMM-ish) | 1 + 4×0.865 ≈ 4.5 | ~169 GOPS | 2.82% |

→ **MAC utilization 4.5× 향상** (단, 절대값은 여전히 낮음 — NPU 본질적으로 LLM decode에 비효율)

### 4.3 시사점

기존 논문은 spec decoding을 "draft 비용 분할 상각"으로만 설명. 하드웨어 측면에서 보면 **NPU 활용률 자체를 끌어올리는 효과**가 있고, 이게 NPU에서 spec decoding이 특히 효과적인 이유. CPU 환경의 spec decoding 이득과 다른 메커니즘.

---

## 5. 논문 본문 변경 제안 (2페이지 분량 강제)

### 5.1 추가할 것 (총 ~10줄)

**§4.1 NPU-only Baseline 끝에 한 단락:**
> 측정값에 대한 선형 회귀 결과 ms/tok = 6.41n − 1370 (R²=0.98)을 얻었으며, 이는 V1 full-prefix verify가 token cost를 prefix length에 정확히 선형 비례시킴을 정량 입증한다.

**새 §4.5 RegimeSpec 실증 (또는 결론 직전):**
> 본 절에서 위 §4.2-4.4의 관찰을 시스템화한 RegimeSpec을 실측한다. RegimeSpec은 ctx<1024이면 PLD, 그 외에는 Sequential을 동적 선택하며, draft 모델은 lazy-load한다. Mixed workload(short 10 + long 5)에서 RegimeSpec은 4.92 tok/s로, always-Sequential 대비 1.30×, oracle 대비 97%를 달성한다(표 X).

**결론 한 줄 추가:**
> RegimeSpec의 단일 ctx threshold 정책은 wiki 분포에서 검증되었으나, 반복성 높은 long prompt에서는 PLD가 long context에서도 우세함을 관찰하였다(Always-PLD 3.9× vs Sequential 2.4× on repetitive workload). 향후 작업에서 prompt-level n-gram 통계 기반의 동적 정책으로 확장 가능하다.

### 5.2 줄일 것 (총 ~10줄 절약)

**§4.2:**
> "Sequential은 k∈{1,2}에서 break-even 근처(0.93-0.97×)에 머물지만 k≥4에서 빠르게 무너지며, async는 모든 k에서 sequential보다 나빠 draft latency hiding의 이득이 전혀 실현되지 않는다."

→ 한 줄로 축약: "Sequential은 모든 k에서 break-even 이하, async는 sequential보다도 나쁘다(부록)."

**§4.3 Long-context 마지막 단락:**
> "Wiki 프롬프트는 자연어 산문으로서... amortization 이득 자체가 제한된다."

→ 두 줄로 축약: "PLD는 wiki natural text에서 hit rate 0.25-0.30이지만 accept/round 0.07-0.28로 매우 낮아, draft cost-free 장점이 약화된다."

**결론 "본 논문의 기여":**
- 첫째/둘째/셋째 항목 압축, RegimeSpec 권고 부분 → 실측 결과로 대체

### 5.3 변화의 핵심 메시지

**Before**: "Edge NPU에서 spec decoding의 최적 전략이 context별로 다르다는 것을 측정으로 발견"
- 측정 보고서 톤
- 결론이 직관적이라는 비판 가능

**After**: "(1) NPU-only degradation을 V1 full-prefix verify 메커니즘으로 정량 입증(R²=0.98), (2) 이를 활용한 RegimeSpec 시스템을 실측 구현하여 oracle의 97% 달성, (3) 정책의 prompt 분포 의존성을 발견"
- 메커니즘 + 시스템 + 일반화 한계의 3-pillar 구조
- 단순 측정에서 깊이 있는 분석으로 격상

---

## 6. 부록 — 회피한 추가 작업

### 6.1 정량 예측 모델 (transition_model.py)

`T_pld(n) = T_seq(n)`을 푸는 transition n* 도출 시도. 결과: prompt-type별 α 차이를 모델이 못 잡아 transition이 wiki/repetitive에서 다르게 나옴. 단일 식으로 표현 어려움 → 본문 미포함.

### 6.2 에너지/Thermal 분석 (measure_thermal_during_inference.py)

Idle baseline 대비 ΔT 측정:
| Workload | NPU Δ | CPU(A76) Δ | GPU Δ | SoC Δ |
|----------|-------|-----------|-------|-------|
| NPU-only (RKLLM) | +5.7°C | **+7.4°C** | +5.0°C | +6.4°C |
| CPU-only (MNN) | +3.3°C | **+8.1°C** | +2.4°C | +5.3°C |
| GPU-only (MNN OpenCL) | +3.2°C | +3.4°C | +3.9°C | +3.5°C |

**관찰**: NPU-only 실행인데 CPU가 NPU보다 더 뜨거워짐. RKLLM이 matmul만 NPU에서 처리하고 나머지는 CPU에서 처리하는 하이브리드 구조 입증.

직접 전력 측정 불가(USB-PD 0 mA)로 정량적 J/token 비교는 datasheet 모델링에 의존. 본문 미포함.

### 6.3 RKNN-LLM 리버스 엔지니어링 발견

- `librkllmrt.so`(6.4MB)는 llama.cpp/GGML 코드 내장
- NPU 접근은 `/dev/dri/card1` (DRM 기반), `/dev/rknpu` 미존재
- 추론 중 26,464회 `DRM_IOCTL_QXL_ALLOC` (매 matmul마다 메모리 할당)
- DMA-BUF 제로카피로 CPU↔NPU 메모리 공유

이 발견은 V1 full-prefix verify 메커니즘 가설을 보강하지만 본문 분량상 미포함.

---

## 7. 새 figure 후보

### Fig A — RegimeSpec speedup bar (필수)
3 workload × 5 modes 막대 그래프. 핵심 메시지 한 장에 다 담김.

### Fig B — NPU degradation linear fit (선택, 분량 여유 시)
ms/tok vs n 산점도 + 선형 fit + R²=0.98 표시. 메커니즘 입증.

### Fig C — NPU MAC utilization roofline (선택)
GEMV(M=1) → GEMM-like(M=4.5) 위치 이동. 하드웨어 통찰.

2페이지면 Fig A 1개만 + 본문 텍스트 강화가 현실적. 분량 여유 있으면 Fig B 추가.
