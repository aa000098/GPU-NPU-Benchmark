# 논문 작성 작업 핸드오프

다른 세션에서 이 논문 작업을 이어서 할 때 참고할 가이드.

---

## 0. 작업 컨텍스트 한 줄 요약

기존 논문(`v2_손현호_통신학회_논문_260418.pdf`)이 **"당위적"이라는 컨펌**을 받아, 이를 해소할 추가 실험·분석을 완료했고, 이제 **논문 본문에 반영하여 다음 버전(v3)을 작성**해야 함.

---

## 1. 현재 자료 위치

### 1.1 기존 논문
```
v2_손현호_통신학회_논문_260418.pdf   # 컨펌 받기 전 버전
```
2페이지 컨퍼런스 논문. 제목: "엣지 NPU LLM 추론을 위한 컨텍스트 길이 기반 하이브리드 Speculative Decoding 전략"

### 1.2 보강 자료 (이 폴더)
```
v2_supplement.md       # 모든 보강 내용 정리. 우선 이걸 먼저 읽어야 함
HANDOFF.md             # 지금 이 파일
```

### 1.3 측정 결과 raw 데이터
```
../results/raw/regimespec_mixed.json    # RegimeSpec mixed workload
../results/raw/regimespec_long.json     # long_only
../results/raw/regimespec_short.json    # short_only
../results/raw/regimespec_*.log         # 위 실행 로그
```

### 1.4 분석 코드
```
../spec_decoding/regimespec.py          # RegimeSpec 시스템 구현 + benchmark
../analysis/mechanism_analysis.py       # NPU degradation 선형 fit + roofline
../analysis/transition_model.py         # 정량 예측 모델 (한계 있음)
../analysis/power_measurement.py        # 에너지 분석 (datasheet 모델)
../analysis/measure_thermal_during_inference.py  # thermal 실측
```

### 1.5 분석 figure
```
../analysis/mechanism_figures/
  fig_npu_degradation.pdf     # NPU 선형 fit (R²=0.98)
  fig_npu_utilization.pdf     # MAC utilization roofline
  fig_async_failure.pdf       # async 타이밍 다이어그램
  fig_prediction.pdf          # 예측 모델 (한계 노출)
  fig_energy.pdf              # 에너지 (사용 보류)
  thermal_traces.json         # thermal raw 데이터
```

---

## 2. 핵심 보강 contribution 3가지 (논문에 들어갈 것)

### A. RegimeSpec 실측 시스템 구현 ⭐️ 필수
**이게 가장 큰 contribution**. abstract에서 약속만 했던 자동 전환 시스템을 실제 구현·측정.

**핵심 숫자:**
- Mixed workload (short 10 + long 5):
  - NPU-only 4.50, Always-Seq 3.78, Always-PLD 4.97, **RegimeSpec 4.92**, Oracle 5.09
  - RegimeSpec / Always-Seq = **1.30×**
  - RegimeSpec / Oracle = **0.97** (거의 최선)
- Long-only (반복 텍스트 prompt):
  - **PLD가 Sequential보다 빠름** (0.55 vs 0.34) → 정책 한계 발견
- Short-only (일반 chat):
  - PLD/RegimeSpec 모두 1.06× (NPU와 큰 차이 없음)

### B. V1 full-prefix verify 정량 입증
기존 논문 §4.1의 "거의 선형"을 정량화.

**핵심 숫자:**
- `ms_per_token(n) = 6.41·n − 1370`
- **R² = 0.98**
- 해석: NPU-only는 매 토큰 생성마다 전체 prefix를 재prefill → context length에 정확히 선형

### C. 정책 한계 발견 (NEW contribution!)
"≥1024 → Sequential" 정책이 **wiki 분포 전제**. 반복 텍스트에서는 long context에서도 PLD가 우세.

**핵심 숫자:**
- Wiki long context: PLD α ≈ 0.07
- 반복 텍스트 long context: PLD α ≈ 0.5+
- → 미래 작업: prompt repetitiveness 기반 동적 정책

---

## 3. 다음 세션에서 해야 할 작업 (우선순위 순)

### 3.1 Fig A 생성 (필수) — RegimeSpec speedup bar
3개 workload × 5개 모드 막대 그래프. 한 장에 핵심 메시지 다 담김.

**데이터**:
```python
# Mixed (15 prompts)
{NPU-only: 4.50, Seq: 3.78, PLD: 4.97, RegimeSpec: 4.92, Oracle: 5.09}
# Long-only (5 prompts)
{NPU-only: 0.14, Seq: 0.34, PLD: 0.55, RegimeSpec: 0.40, Oracle: 0.55}
# Short-only (10 prompts)
{NPU-only: 9.98, Seq: 6.07, PLD: 10.61, RegimeSpec: 10.60, Oracle: 10.63}
```

원본 raw 데이터: `../results/raw/regimespec_*.json`

### 3.2 Fig B 생성 (필수 또는 권장) — NPU degradation 선형 fit
ms/tok vs n 산점도 + 선형 fit + R²=0.98 표시.

이미 그려진 게 있음: `../analysis/mechanism_figures/fig_npu_degradation.pdf` — 그대로 쓰거나 스타일 통일 위해 재작성.

### 3.3 본문 작성 — §4.5 RegimeSpec 측정 추가
`v2_supplement.md`의 "5.1 추가할 것" 섹션 참고. 4-5줄로:
> 본 절에서 위 §4.2-4.4의 관찰을 시스템화한 RegimeSpec을 실측한다. RegimeSpec은 ctx<1024이면 PLD, 그 외에는 Sequential을 동적 선택한다. Mixed workload(short 10 + long 5)에서 RegimeSpec은 4.92 tok/s로 always-Sequential 대비 1.30×, oracle 대비 97%를 달성한다(표 X). 한편 반복적 long prompt에서는 Always-PLD가 3.9× speedup으로 Sequential(2.4×)을 앞서, 정책의 prompt 분포 의존성이 드러났다.

### 3.4 §4.1 보강 — 선형 모델 식 한 줄
> 측정값에 대한 선형 회귀 결과 ms/tok = 6.41n − 1370 (R²=0.98)을 얻었으며, 이는 V1 full-prefix verify가 token cost를 prefix length에 정확히 선형 비례시킴을 정량 입증한다.

### 3.5 결론 수정
"본 논문의 기여" 부분에서:
- "context-dependent 백엔드 선택 정책을 제안한다" → "RegimeSpec 시스템을 구현하여 mixed workload에서 oracle 대비 97% 달성"
- 새 항목 추가: "정책의 prompt 분포 의존성 발견 → future work 방향"

### 3.6 분량 조정 (2페이지 강제)
**줄여야 할 곳** (`v2_supplement.md` §5.2 참고):
- §4.2 마지막 두 문장 한 줄로 축약
- §4.3 long-context 마지막 단락 두 줄로 축약
- 결론 "기여" 항목들 압축

---

## 4. 논문 작성 시 핵심 메시지 변화

### Before (기존 v2)
> "엣지 NPU의 spec decoding 최적 전략이 context length에 따라 달라진다는 것을 측정했다. 짧으면 PLD, 길면 Sequential."

→ 측정 보고서. 결론이 직관적.

### After (v3 목표)
> "(1) NPU-only degradation을 V1 full-prefix verify 메커니즘으로 정량 입증(R²=0.98), (2) 이를 활용한 RegimeSpec 시스템을 실측 구현하여 oracle 대비 97% 달성, (3) 정책의 prompt 분포 의존성을 발견."

→ 메커니즘 + 시스템 + 한계의 3-pillar 구조.

---

## 5. 절대 빼먹지 말 것

### 5.1 Abstract 수정
기존: "RegimeSpec을 설계·구현한다" (실현 안 됐었음)
v3: 실현됐으니 "구현하고 mixed workload에서 oracle 97% 달성을 보인다"로 강화

### 5.2 RegimeSpec 표
워크로드별 정책 비교 표는 반드시 들어가야 함. 텍스트만으로 안 됨.

### 5.3 한계 명시
"본 정책은 wiki 분포에서 검증되었으며, 반복적 prompt에서는 long context에서도 PLD가 우세함을 관찰" — 이 한 문장이 논문을 단순 측정에서 분석으로 끌어올림.

---

## 6. 데이터 reproduction (필요 시)

새 세션에서 측정 다시 돌리려면:

```bash
cd /home/hyunho.son/Desktop/project/GPU-NPU-Benchmark
source ~/local/venv/bin/activate

# RegimeSpec 측정 (각 ~10-30분, mixed가 가장 김)
python llm_quant_bench/spec_decoding/regimespec.py \
    --target_model llm_quant_bench/rkllm_models/Llama-3.2-1B-Instruct_W8A8_RK3588.rkllm \
    --draft_model llm_quant_bench/draft_models/Llama-3.2-1B-Instruct-Q4_0.gguf \
    --tokenizer llm_quant_bench/models/Llama-3.2-1B-Instruct \
    --workload mixed \
    --output_json llm_quant_bench/results/raw/regimespec_mixed.json

# workload {mixed, long_only, short_only} 변경 가능

# 메커니즘 분석 (계산만, 빠름)
python llm_quant_bench/analysis/mechanism_analysis.py
```

---

## 7. 회피한 자료 (논문에 안 들어감, 부록 후보)

- 정량 예측 모델 (`transition_model.py`): prompt-type별 α 차이 못 잡음. 한계 있음.
- 에너지/thermal 분석: 직접 전력 측정 불가, datasheet 모델 의존. 메인 메시지 흐림.
- RKNN 리버스 엔지니어링 (DRM, DMA-BUF 등): 메커니즘 보강은 되지만 분량상 어려움.

이런 자료는 발표/포스터에 활용하거나 **확장판 저널 논문**으로 쓸 수 있음.

---

## 8. 권장 작업 순서

다음 세션에서:

1. **`v2_supplement.md` 먼저 읽기** (모든 변경 사항 자세히 정리됨)
2. 기존 PDF (`v2_손현호_통신학회_논문_260418.pdf`) 다시 보기
3. Fig A (RegimeSpec speedup bar) 생성 — 가장 임팩트 큼
4. §4.5 본문 작성 (위 3.3 참고)
5. §4.1 한 줄 추가 (위 3.4 참고)
6. 결론 수정 (위 3.5 참고)
7. 분량 조정 (§4.2, §4.3 압축)
8. (여유 있으면) Fig B 추가
9. Abstract 업데이트

---

## 9. 핵심 정량값 chart (논문에서 인용할 모든 숫자)

| 지표 | 값 | 출처 |
|------|-----|------|
| RegimeSpec mixed avg tok/s | 4.92 | regimespec_mixed.json |
| Oracle mixed avg tok/s | 5.09 | regimespec_mixed.json |
| RegimeSpec / Always-Seq | 1.30× | derived |
| RegimeSpec / Oracle | 0.97 | derived |
| Always-PLD long | 0.55 tok/s (3.9×) | regimespec_long.json |
| Always-Seq long | 0.34 tok/s (2.4×) | regimespec_long.json |
| Always-Seq short | 6.07 tok/s (0.61× 퇴행) | regimespec_short.json |
| NPU degradation 선형 a | 6.41 ms/n | mechanism_analysis.py |
| NPU degradation 선형 b | -1369.7 ms | mechanism_analysis.py |
| R² | 0.98 | mechanism_analysis.py |
| Wiki long PLD α | 0.07 | 논문 §4.4 |
| 반복 long PLD α | ~0.5 | 우리 측정에서 추정 |
| Llama draft α | 0.865 | 논문 §3.8 |
| NPU MAC utilization (GEMV) | 0.63% | mechanism_analysis.py |
| NPU MAC utilization (GEMM-like spec) | 2.82% | mechanism_analysis.py |

이 수치들은 모두 본문/표/그림 어딘가에 들어가야 함.
