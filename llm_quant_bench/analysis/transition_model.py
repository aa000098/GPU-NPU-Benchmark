"""
Transition Point 정량 예측 모델

가설: NPU-only / Sequential / PLD의 token cost를 a priori 예측 가능.

T_npu_only(n) = T_prefill(n)                                     # 매 token마다 full prefix
T_seq(n, k) = (T_prefill(n+k) + T_draft(k)) / (1 + α_seq · k)    # k tokens accept 평균
T_pld(n, k) = T_prefill(n+k) / (1 + α_pld · k)                   # draft cost 0

T_prefill(n) = a·n + b  (mechanism_analysis.py에서 측정)

Transition (PLD = Sequential):
  T_prefill(n*) · (1/(1+α_pld·k) - 1/(1+α_seq·k)) = T_draft / (1+α_seq·k)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

OUTPUT_DIR = "/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/analysis/mechanism_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════
# 측정 파라미터 (논문 본문 + mechanism_analysis.py에서 fit)
# ════════════════════════════════════════════════════════════════

# T_prefill(n) = a·n + b (NPU-only ms/tok linear fit, R²=0.98)
PREFILL_SLOPE = 6.41   # ms per token of context
PREFILL_INTERCEPT = -1369.7  # 음수 → 짧은 context에서 다른 메커니즘 (모델 로딩 등)
# 실용적 사용: max(0, a·n + b) 또는 b 값 조정 필요

# Acceptance rates (논문 본문)
ALPHA_SEQ = 0.865    # Llama Q4_0 self-quant draft
ALPHA_PLD_LONG = 0.07  # wiki long context (n≥1024)
ALPHA_PLD_SHORT = 0.5  # repetitive code/copy short context

# Draft cost (논문: 18.7 GB/s draft throughput → ~16 ms for 4 tokens)
T_DRAFT_PER_TOKEN_MS = 4.0  # CPU Q4_0 draft
K = 4  # default speculation depth


# ════════════════════════════════════════════════════════════════
# Token cost 모델
# ════════════════════════════════════════════════════════════════

def T_prefill(n):
    """V1 full-prefix verify cost in ms."""
    return max(50, PREFILL_SLOPE * n + PREFILL_INTERCEPT)  # floor at 50ms (short ctx)

def T_npu_only(n):
    """NPU-only baseline: 1 token per round."""
    return T_prefill(n)

def T_sequential(n, k=K, alpha=ALPHA_SEQ):
    """Sequential: 1 + α·k tokens per round, draft cost added."""
    t_round = T_prefill(n + k) + T_DRAFT_PER_TOKEN_MS * k
    tokens_per_round = 1 + alpha * k
    return t_round / tokens_per_round

def T_pld(n, k=K, alpha=ALPHA_PLD_LONG):
    """PLD: 1 + α·k tokens per round, no draft cost."""
    return T_prefill(n + k) / (1 + alpha * k)


def predicted_speedup(n, mode="seq"):
    """Speedup vs NPU-only baseline."""
    base = T_npu_only(n)
    if mode == "seq":
        cand = T_sequential(n)
    elif mode == "pld_long":
        cand = T_pld(n, alpha=ALPHA_PLD_LONG)
    elif mode == "pld_short":
        cand = T_pld(n, alpha=ALPHA_PLD_SHORT)
    return base / cand


# ════════════════════════════════════════════════════════════════
# Transition point 분석
# ════════════════════════════════════════════════════════════════

def find_transition_point():
    """PLD = Sequential인 n* 탐색."""
    n_range = np.arange(50, 4000, 10)
    diff = [T_sequential(n) - T_pld(n, alpha=ALPHA_PLD_LONG) for n in n_range]

    # First sign change: PLD becomes worse than Seq (long context)
    for i in range(1, len(diff)):
        if diff[i-1] > 0 and diff[i] <= 0:
            return n_range[i]
    return None


# ════════════════════════════════════════════════════════════════
# Figure: Predicted vs Measured speedup
# ════════════════════════════════════════════════════════════════

def fig_prediction_vs_measurement():
    n_smooth = np.linspace(50, 4000, 200)

    seq_speedups = [predicted_speedup(n, "seq") for n in n_smooth]
    pld_long_speedups = [predicted_speedup(n, "pld_long") for n in n_smooth]
    pld_short_speedups = [predicted_speedup(n, "pld_short") for n in n_smooth]

    # 측정값 (논문 표 4.3)
    n_measured = [512, 1024, 2048, 3500]
    seq_measured = [0.784/0.540, 0.529/0.241, 0.236/0.097, 0.104/0.045]  # tok_s ratio
    pld_measured = [0.577/0.540, 0.256/0.241, 0.130/0.097, 0.048/0.045]

    fig, ax = plt.subplots(figsize=(11, 7))

    # Predicted curves
    ax.plot(n_smooth, seq_speedups, 'r-', linewidth=2,
            label=f'Predicted Sequential (k={K}, α={ALPHA_SEQ})')
    ax.plot(n_smooth, pld_long_speedups, 'b-', linewidth=2,
            label=f'Predicted PLD long (α={ALPHA_PLD_LONG})')
    ax.plot(n_smooth, pld_short_speedups, 'b--', linewidth=1.5, alpha=0.6,
            label=f'Predicted PLD repetitive (α={ALPHA_PLD_SHORT})')

    # Measured points
    ax.scatter(n_measured, seq_measured, c='red', s=100, marker='o',
               edgecolors='black', linewidth=1, zorder=5,
               label='Measured Sequential')
    ax.scatter(n_measured, pld_measured, c='blue', s=100, marker='^',
               edgecolors='black', linewidth=1, zorder=5,
               label='Measured PLD (wiki)')

    # Baseline line
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, label='NPU-only baseline')

    # Transition point
    n_star = find_transition_point()
    if n_star:
        ax.axvline(x=n_star, color='purple', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.text(n_star + 50, 0.5, f'Predicted\ntransition\nn*={n_star}',
                color='purple', fontsize=10, fontweight='bold')

    ax.set_xlabel('Context Length n (tokens)', fontsize=12)
    ax.set_ylabel('Speedup vs NPU-only', fontsize=12)
    ax.set_title('Predicted vs Measured Speedup\n'
                 f'Model: T_token = T_prefill(n+k) / (1 + α·k) + T_draft·k/(1+α·k)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 3.5)
    ax.set_xlim(0, 4000)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(OUTPUT_DIR, f"fig_prediction.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_prediction.pdf/png")
    return n_star, seq_measured, pld_measured, n_measured


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Transition Point Prediction Model")
    print("=" * 70)

    # 모델 파라미터 출력
    print(f"\nModel parameters:")
    print(f"  T_prefill(n) = {PREFILL_SLOPE}·n + {PREFILL_INTERCEPT} ms")
    print(f"  α_seq (Llama Q4_0) = {ALPHA_SEQ}")
    print(f"  α_pld (wiki long) = {ALPHA_PLD_LONG}")
    print(f"  T_draft per token = {T_DRAFT_PER_TOKEN_MS} ms (CPU Q4_0)")
    print(f"  k (speculation depth) = {K}")

    # Transition point
    n_star = find_transition_point()
    print(f"\n→ Predicted PLD↔Sequential transition: n* = {n_star} tokens")
    print(f"  (논문 측정 transition: ~1024 tokens)")
    if n_star:
        error = abs(n_star - 1024) / 1024 * 100
        print(f"  Error: {error:.1f}%")

    # Predicted speedups at key contexts
    print(f"\nPredicted speedups (vs NPU-only):")
    print(f"  {'n':>6s} {'NPU-only':>10s} {'Sequential':>12s} {'PLD long':>10s} {'PLD short':>10s}")
    for n in [15, 512, 1024, 2048, 3500]:
        sp_seq = predicted_speedup(n, "seq")
        sp_pld_l = predicted_speedup(n, "pld_long")
        sp_pld_s = predicted_speedup(n, "pld_short")
        print(f"  {n:>6d} {1.0:>10.2f} {sp_seq:>12.2f} {sp_pld_l:>10.2f} {sp_pld_s:>10.2f}")

    # Figure
    print()
    n_star, seq_m, pld_m, n_m = fig_prediction_vs_measurement()

    # Save
    summary = {
        "parameters": {
            "T_prefill_slope_ms_per_tok": PREFILL_SLOPE,
            "T_prefill_intercept_ms": PREFILL_INTERCEPT,
            "alpha_seq": ALPHA_SEQ,
            "alpha_pld_long": ALPHA_PLD_LONG,
            "alpha_pld_short": ALPHA_PLD_SHORT,
            "t_draft_per_tok_ms": T_DRAFT_PER_TOKEN_MS,
            "k": K,
        },
        "predicted_transition_n_star": int(n_star) if n_star else None,
        "measured_transition_paper": 1024,
        "error_pct": abs(n_star - 1024) / 1024 * 100 if n_star else None,
        "predicted_speedups": {
            str(n): {
                "seq": predicted_speedup(n, "seq"),
                "pld_long": predicted_speedup(n, "pld_long"),
                "pld_short": predicted_speedup(n, "pld_short"),
            }
            for n in [15, 512, 1024, 2048, 3500]
        },
    }
    with open(os.path.join(OUTPUT_DIR, "transition_model.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved transition_model.json")


if __name__ == "__main__":
    main()
