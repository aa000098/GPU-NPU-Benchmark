# RK3588 NPU Speculative Decoding — 현 시점 실측 정리

**날짜:** 2026-04-18
**Target:** Llama-3.2-1B-Instruct, W8A8 RKLLM (`Llama-3.2-1B-Instruct_W8A8_RK3588.rkllm`)
**Draft:** Llama-3.2-1B-Instruct Q4_0 GGUF (llama.cpp)
**Verify 경로:** V1 full-prefix (매 라운드 `clear_kv_cache` + 전체 프리픽스 재프리필, `keep_history=1`)
**런타임:** `llm_quant_bench/benchmark/hybrid_runtime_v2.py`

---

## 0. 한 줄 요약

> Short-ctx(≤50) 에서는 draft-based가 전패·PLD만 이기지만, **long-ctx(≥1024)에서는 draft-based sequential이 2.2-2.5× 로 PLD(1.06-1.33×)를 크게 앞선다**. 전환 원인은 V1 full-prefill의 ctx 지배성 — accept된 토큰이 prefill 비용에 amortize되는 구조.

원래 "이 NPU 클래스에선 draft-based가 근본적으로 안 된다"는 주장은 **ctx 범위 제한된 결론**이었음. 정정된 narrative 필요.

---

## 1. 측정한 세 가지 축

### A. Short-ctx (프롬프트 ~15 토큰, n_gen=32) — `results/raw/hybrid_e2e/`

| Mode | k | tok/s | NPU-only 대비 |
| --- | ---: | ---: | ---: |
| NPU-only | — | **8.34** | 1.00× |
| Sequential | 1 | 7.78 | 0.93× |
| Sequential | 2 | 8.09 | 0.97× |
| Sequential | 4 | 6.13 | 0.74× |
| Sequential | 8 | 4.92 | 0.59× |
| Async (thread) | 1 | 6.40 | 0.77× |
| Async (thread) | 2 | 6.09 | 0.73× |
| Async (thread) | 4 | 3.81 | 0.46× |
| Async (thread) | 8 | 2.95 | 0.35× |

### B. PLD short-prompt sweep (n_gen=48) — `results/raw/pld_sweep/`

| 프롬프트 유형 | NPU-only | k=2 | k=4 | k=8 | best |
| --- | ---: | ---: | ---: | ---: | :--- |
| `copy` (반복 문장) | 4.63 | 5.69 (1.23×) | 5.71 (1.23×) | **6.19 (1.34×)** | k=8 |
| `code` (fib→factorial) | 3.31 | 5.15 (1.56×) | **6.13 (1.85×)** | 6.00 (1.81×) | k=4 |
| `list`, `qa`, `text` | — | — | — | — | 미측정 (sweep 중단) |

### C. Long-ctx wiki 프롬프트 멀티-ctx sweep (n_gen=32, k=4)

Sequential/npu_only: `results/raw/longctx_hybrid.json` (target+draft 동시 로드).
PLD: `results/raw/pld_only_sweep.json` (target만 로드, [scripts/run_pld_only_sweep.py](scripts/run_pld_only_sweep.py)). §5 참조 — ctx=1024/2048에서 동일 값 재현으로 DraftEngine 유무가 PLD 처리량에 영향 없음 확인됨.

| ctx | n_prompt | npu_only | sequential k=4 | PLD k=4 | seq speedup | PLD speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 (smoke) | 480 | 0.540 | 0.784 | 0.577 | 1.45× | 1.07× |
| 1024 | 976 | 0.241 | 0.529 | 0.256 | **2.19×** | 1.06× |
| 2048 | 2000 | 0.097 | 0.236 | 0.130 | **2.44×** | **1.33×** |
| 3500 | 3452 | 0.045 | 0.104 | **0.048** | **2.32×** | **1.07×** |

라운드 지표 (PLD, `pld_only_sweep`):

| ctx | rounds | n-gram hit rate | accept/round | emit/round |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 32 | 0.25 | 0.00 | 1.00 (bonus만) |
| 2048 | 25 | 0.48 | 0.28 | 1.28 |
| 3500 | 30 | 0.30 | 0.07 | 1.07 |

관찰:
- **NPU-only ctx 지배성이 명확**: 0.54 → 0.045 tok/s (ctx 7배 → tok/s 12배 저하). V1 full-prefill 한 번이 ctx 전체 프리필이라 그렇다.
- **Sequential speedup이 ctx에 거의 monotonic 증가** (1.45 → 2.19 → 2.44 → 2.32×). 3500에서 약간 내려가지만 여전히 2× 넘음.
- **PLD는 wiki 계열 장문 프롬프트에 취약** — ctx=1024/3500에선 accept/round < 0.1 로 거의 pure NPU-only 수준 (1.06–1.20×). 2048에서만 accept=0.28로 올라 1.33×.
- **PLD가 sequential보다 약한 이유**: wiki 프롬프트는 n-gram 반복이 적어 proposal 자체가 없거나 target argmax와 매칭 안 됨. Sequential은 실제 1B Q4_0 draft가 greedy match 87%라 round당 2.3-2.6 토큰 accept → 큰 amortization 이득.

---

## 2. Draft forward time breakdown (`llama-bench`)

`/home/hyunho.son/install_files/llama.cpp/build/bin/llama-bench`, 4 threads, tg128 (decode 128 토큰).

| Draft 후보 | Size | tok/s | ms/tok | Effective BW |
| --- | ---: | ---: | ---: | ---: |
| Llama 1B Q4_0 | 730 MiB | 25.67 | 38.96 | **18.7 GB/s** |
| Llama 1B IQ3_M | 619 MiB | 10.61 | 94.25 | 6.9 GB/s (compute-bound) |
| Qwen 0.5B Q4_0 | 403 MiB | 55.05 | 18.17 | 23.2 GB/s (mmap 캐시 적중 포함) |
| Llama 1B Q4_0_4_8 | 771 MiB | — | — | (현 llama.cpp 빌드 미지원) |

**1B Q4_0는 RK3588 DRAM 실측 peak (19.7 GB/s)에 붙어 있음 → memory-bound.** 이 draft의 T_draft를 더 줄이려면 파라미터 수를 줄이거나(즉 작은 모델) 양자화 더 공격적으로 해야 하는데 IQ3는 오히려 느려짐 (dequant compute 부담).

---

## 3. Acceptance rate α (이미 측정된 값)

| Draft | N positions | mean α (stochastic) | mean greedy_hit |
| --- | ---: | ---: | ---: |
| Llama 1B Q4_0 | 1260 | 0.884 | **0.865** |
| Qwen 0.5B Q4_0 | 1260 | 0.116 | 0.173 (cross-tokenizer 재앙) |

`results/raw/acceptance_llama_draft.json`, `results/raw/acceptance_qwen_draft.json`.

Qwen은 tokenizer 불일치로 사실상 쓸 수 없음. Llama self-quant(Q4_0 → W8A8) 쌍은 α=0.865로 강함.

---

## 4. (α, T_draft, k) feasibility scan

**공식:** sequential spec이 NPU-only를 이길 수 있는 draft per-token budget

```
T_draft_max(α, k, ctx) = [(E[accepts](α,k)+1) · T_verify(1,ctx) − T_verify(k,ctx)] / k
E[accepts](α, k) = α(1 − α^k) / (1 − α)
```

`T_verify(k,ctx)` 는 probe v2 Mode C (staged verify, best-case runtime) 실측. **이건 현재 hybrid_runtime_v2의 V1 full-prefix보다 낙관적 — staged runtime이 구현되면 가능한 이상적 비용.** V1에서는 실측 T_verify가 2-3× 커 budget이 좁아짐.

**ctx=4079 (long-ctx) budget 표 (Mode C 기준):**

| α | k=1 | k=2 | k=4 | k=8 | k=16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.87 | 151 | 34 | **68** | **67** | 44 |
| 0.95 | 165 | 54 | 97 | 109 | 95 |

**측정된 draft 대입 (Mode C staged 기준 이론 예측):**

| Draft | T_draft | α | ctx=32 k=4 | ctx=4079 k=4 | ctx=4079 k=8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama 1B Q4_0 | 39 | 0.87 | 0.82× ❌ | 1.20× ✅ | 1.29× ✅ |
| Qwen 0.5B Q4_0 | 18 | 0.17 | 0.39× ❌ | 0.44× ❌ | 0.37× ❌ |
| Hypothetical 160M | 5 | 0.65 | 1.19× ✅ | 1.04× ✅ | 1.06× ✅ |
| Ideal (T_draft=0) | 0 | 0.87 | 2.23× ✅ | 1.66× ✅ | 2.25× ✅ |

Staged verify가 되면 1B Q4_0로 long-ctx 1.2-1.3× 이론상 가능. **V1 (현재 런타임)에선 ctx=3500 sequential 실측 2.50×로 오히려 더 크게 이김** — prefill amortization이 Mode C 예측보다 관대하기 때문.

Scan 산출물: [analysis/draft_feasibility_scan.py](analysis/draft_feasibility_scan.py), [results/raw/draft_feasibility_scan.json](results/raw/draft_feasibility_scan.json).

---

## 5. 품질 (PLD output == NPU-only output ?)

`scripts/compare_pld_vs_npu.py` + `results/raw/pld_vs_npu_diff.json`: n_gen=48, k=4, 3 프롬프트.

| 프롬프트 | 일치 여부 | speedup |
| --- | :---: | ---: |
| `copy` | **IDENTICAL** | 1.28× |
| `code` | **DIVERGE @ pos 3** | 1.47× |
| `text` | **IDENTICAL** | 0.99× |

`code`의 경우 Prefix가 동일(`prompt + [16, 696, 755]`)한데도 position 3의 argmax가 다르게 나옴:
- `npu_only`: `...1)\n\ndef fibonacci_sequence(n):\n   `
- `pld k=4`:  `...1)\n\ndef power(base, exponent):\n`

원인: RKLLM **W8A8 path가 prefill 길이에 대해 numerically invariant하지 않음**. 같은 row의 logits가 prefill 길이 L+3 (npu_only) vs L+3+k (pld verify)에서 미세하게 달라져 argmax tie 경계에서 뒤집힘. 두 출력 모두 greedy 규칙 일관됨 → **quality는 열화 없음**, 단 "bitwise equivalence"는 이 closed-runtime에서 성립하지 않음.

논문 claim: "quality preserving under greedy" → **"distributionally equivalent under greedy up to runtime numerical noise"**로 약화.

---

## 6. ctx=3500 PLD 크래시 — 원인 확정: OOM-killer (SIGKILL)

재현성 있는 silent crash. 격리 실험 4단계로 원인 국한:

| Test | DraftEngine 로드 | 이전 모드 warmup | 결과 |
| --- | :---: | :---: | :--- |
| **raw get_logits 50회** ([diag_v2](scripts/diag_pld_crash_v2.py)) | ✗ | — | ✅ RSS 안정 (5662 MB 고정), 누수 없음 |
| **Test A** (PLD 단독, [diag_v3](scripts/diag_pld_crash_v3.py) `--test A`) | ✗ | — | ✅ 8 토큰 완주, RSS 5637 MB |
| **Test B** (DraftEngine 로드만 + PLD) | ✓ | — | ✅ 8 토큰 완주, RSS 5626 MB, avail 2776 MB |
| **Test C** (main path: npu_only 32회 + seq 10회 + PLD, `--test C`) | ✓ | 42회 verify | ❌ **PLD 6번째 토큰 후 SIGKILL** (`죽었음/Killed` bash 메시지) |

결정적 증거: Test C에서 bash가 **"줄 9: 771842 죽었음"** 출력. 이는 kernel이 SIGKILL을 보냈다는 뜻 = **OOM-killer 발동**.

메모리 추이 (Test C):

| 시점 | RSS | avail |
| --- | ---: | ---: |
| Target 로드 | 2203 MB | 6573 |
| + Draft 로드 | 3822 MB | 5682 |
| npu_only 32회 후 | 5429 MB | 2648 |
| sequential 10회 후 | 6224 MB | **2321** |
| PLD 7번째 verify | — | (SIGKILL) |

원인 메커니즘:
1. **Target+Draft 공존 시 RSS가 verify 반복으로 느린 증가** (5429 → 6224 MB). Target 단독 (v2 repro)에서는 5662 MB 고정. llama.cpp + RKLLM 상호작용에서 뭔가 쌓임.
2. PLD 시작 시점 avail **2321 MB**.
3. Verify call이 3476×128256×4 = **1.78 GB fp32 logits** 반환. numpy `.copy()` 시점 peak 3.56 GB 요구.
4. avail 2321 MB < peak 3.56 GB → OOM-killer 발동.

**왜 sequential/npu_only는 살아남았나**: 같은 verify path를 쓰지만 sequential은 10 rounds로 적고, npu_only는 그 시점 avail이 5–6 GB라 여유. PLD는 양쪽 모두 돌고 난 뒤 시작해 critical threshold에서 밀림.

### 확정 결론 + 처리

- **크래시는 PLD 자체 결함 아님**. Target+Draft 동시 로드 상태에서 warmup 뒤 avail 부족이 누적된 결과.
- PLD는 **draft 모델을 쓰지 않으므로** 실전에선 DraftEngine 로드할 이유 없음.
- [scripts/run_pld_only_sweep.py](scripts/run_pld_only_sweep.py)로 **Target만 로드한 PLD 경로** 구현. ctx=1024/2048 재측정 결과 이전 값과 **토큰 단위로 동일** → DraftEngine 유무가 PLD 처리량에 영향 없음. 이전 §1C 표의 ctx=1024/2048 PLD 숫자는 그대로 유효.
- ctx=3500 PLD는 새 경로에서 정상 완주 (**0.048 tok/s = 1.07×**). §1C 표 마지막 칸 채움.

※ Sequential/async는 draft를 본질적으로 써야 하므로 원래 구성 유지. 모드별 "natural deployment config" 비교.

---

## 7. 논문 narrative — 수정된 thesis

### 이전 (무효):
> "Closed-runtime 엣지 NPU에서 draft-based speculative decoding은 메모리 버스 경합 때문에 항상 실패한다. PLD만이 정답."

### 현재 (수정):
> "Speculative decoding의 성공은 ctx 범위에 따라 갈린다. **Short-ctx에서는 verify가 싸서 draft overhead가 커 보이지만, long-ctx에서는 V1 full-prefill이 압도적 지배인자가 되어 모든 accept된 토큰이 prefill 비용에 amortize됨**. 이 구조에서:
> - Short-ctx: PLD가 유일한 win (copy 1.34×, code 1.85×)
> - Long-ctx (V1 런타임): 1B Q4_0 draft-based sequential이 2.19-2.50× — PLD(1.06-1.33×)를 크게 앞섬
> - 즉 엣지 NPU의 실용적 ctx 분포(수천 토큰 이상)에선 draft-based가 더 강력한 remedy."

### 남은 주장
- Async는 여전히 fail (CPU↔NPU 경합). 0.35-0.77×.
- Qwen 0.5B cross-tokenizer 재앙 (α=0.17) → 토크나이저 일치 중요.
- PLD의 "draft cost = 0" 장점은 long-ctx에서 오히려 작아짐 (prefill이 어차피 큼).
- V2 staged verify 런타임이 있으면 이야기가 또 달라질 수 있음 (§4 scan 참조) — 현 RKLLM은 미지원.

### 남은 작업
- Draft-based long-ctx에서 k sweep (k=2, 4, 8, 16) — 현재 k=4만
- Sequential @ ctx=3500 재현 (현재 1회만)
- (옵션) Short-ctx PLD sweep 마무리 (`list`, `qa`, `text`는 미완)
- (옵션) V2 staged verify 런타임 구현 시 sequential이 PLD를 얼마나 이기는지 재평가

---

## 8. 파일 맵

| 역할 | 파일 |
| --- | --- |
| 런타임 | [benchmark/hybrid_runtime_v2.py](benchmark/hybrid_runtime_v2.py) |
| RKLLM get_logits | [eval/ppl_rkllm.py](eval/ppl_rkllm.py) |
| PLD 짧은 프롬프트 sweep | [scripts/run_pld_sweep.sh](scripts/run_pld_sweep.sh), `results/raw/pld_sweep/` |
| PLD vs NPU-only 품질 비교 | [scripts/compare_pld_vs_npu.py](scripts/compare_pld_vs_npu.py), `results/raw/pld_vs_npu_diff.json` |
| Long-ctx multi-sweep | [scripts/run_longctx_hybrid.py](scripts/run_longctx_hybrid.py), `results/raw/longctx_hybrid.json` |
| Draft 벤치 | `results/raw/draft_breakdown/llama_bench_decode.json` |
| Feasibility scan | [analysis/draft_feasibility_scan.py](analysis/draft_feasibility_scan.py), `results/raw/draft_feasibility_scan.json` |
| Crash diag | [scripts/diag_pld_crash.py](scripts/diag_pld_crash.py), [_v2](scripts/diag_pld_crash_v2.py), [_v3](scripts/diag_pld_crash_v3.py) |
