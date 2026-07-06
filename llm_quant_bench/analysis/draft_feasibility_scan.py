"""Feasibility scan for draft-based speculative decoding on RK3588 NPU.

For each (ctx, k, alpha), compute the maximum T_draft (ms) such that sequential
speculative decoding beats NPU-only decoding:

    T_draft_max = [(E[accepts] + 1) * T_verify(1) - T_verify(k)] / k
    E[accepts](alpha, k) = alpha * (1 - alpha^k) / (1 - alpha)  for alpha < 1

Uses probe v2 Mode C (staged verify, keep_history=1) as T_verify(k) — this is
the *best-case* verify cost that a properly incremental RKLLM runtime could
achieve. The realistic V1 full-prefix path used in hybrid_runtime_v2 is strictly
worse, so any negative conclusion here is a lower bound on the real difficulty.

Overlays measured draft points (Llama 1B Q4_0, Qwen 0.5B Q4_0) plus a
hypothetical "memory-BW-limited tiny draft" floor to show no realistic draft
lands in the win region at long context.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "llm_quant_bench/results/raw/npu_verify_v2_ctx4096.json"
OUT = ROOT / "llm_quant_bench/results/raw/draft_feasibility_scan.json"


def load_T_verify():
    """Return dict[ctx][k] = T_verify_ms (Mode C staged verify median)."""
    data = json.load(open(PROBE))
    tbl = {}
    for r in data:
        ctx, k = r["ctx"], r["k"]
        c = r.get("C_staged_verify")
        if c is None:
            continue
        tbl.setdefault(ctx, {})[k] = c["median_ms"]
    return tbl


def expected_accepts(alpha, k):
    if k <= 0:
        return 0.0
    if alpha >= 1.0:
        return float(k)
    return alpha * (1 - alpha ** k) / (1 - alpha)


def T_draft_max(alpha, k, ctx_tbl):
    """Max T_draft (ms) for which sequential spec beats NPU-only."""
    if 1 not in ctx_tbl or k not in ctx_tbl:
        return None
    T_base = ctx_tbl[1]                  # NPU-only per-tok cost
    T_v = ctx_tbl[k]                     # verify cost for k-token extension
    E = expected_accepts(alpha, k)
    budget = (E + 1) * T_base - T_v      # headroom for draft work
    return budget / k                    # max per-token draft cost


def speedup(alpha, T_draft, k, ctx_tbl):
    if 1 not in ctx_tbl or k not in ctx_tbl:
        return None
    T_base = ctx_tbl[1]
    T_v = ctx_tbl[k]
    E = expected_accepts(alpha, k)
    emit = E + 1
    round_ms = k * T_draft + T_v
    return emit * T_base / round_ms


def main():
    verify_tbl = load_T_verify()
    ctxs = sorted(verify_tbl.keys())
    ks = [1, 2, 4, 8, 16]
    alphas = [0.30, 0.50, 0.70, 0.80, 0.87, 0.95, 0.99]

    # Measured draft points (name -> (T_draft_ms, alpha_greedy_hit))
    drafts = {
        "Llama 1B Q4_0":      (38.96, 0.865),   # llama-bench tg128 + acceptance_llama_draft
        "Llama 1B IQ3_M":     (94.25, 0.865),   # assume same α (same base model, smaller quant)
        "Qwen 0.5B Q4_0":     (18.17, 0.173),   # llama-bench + acceptance_qwen_draft (cross-tok disaster)
        # Hypothetical tiny draft: 100 MB Q4 at 19 GB/s BW-floor = 5.3 ms/tok; alpha plausibly 0.5-0.7
        "Hypothetical 160M":  (5.3, 0.65),
        # Infinitely fast draft: keep alpha realistic
        "Ideal (0 ms, α=0.87)": (0.0, 0.865),
    }

    # Build output table of T_draft_max for each (ctx, k, alpha)
    out = {
        "T_verify_ms (Mode C staged)": verify_tbl,
        "alpha_grid": alphas,
        "k_grid": ks,
        "T_draft_max_ms (by ctx, k, alpha)": {},
        "measured_drafts_speedup": {},
    }
    for ctx in ctxs:
        out["T_draft_max_ms (by ctx, k, alpha)"][ctx] = {}
        for k in ks:
            row = {}
            for a in alphas:
                m = T_draft_max(a, k, verify_tbl[ctx])
                row[a] = m
            out["T_draft_max_ms (by ctx, k, alpha)"][ctx][k] = row

    for name, (td, a) in drafts.items():
        out["measured_drafts_speedup"][name] = {"T_draft_ms": td, "alpha": a, "speedup_by_ctx_k": {}}
        for ctx in ctxs:
            out["measured_drafts_speedup"][name]["speedup_by_ctx_k"][ctx] = {}
            for k in ks:
                s = speedup(a, td, k, verify_tbl[ctx])
                out["measured_drafts_speedup"][name]["speedup_by_ctx_k"][ctx][k] = s

    # Pretty-print for the reader
    print("="*90)
    print("T_verify (ms, staged mode C) per (ctx, k)")
    print("="*90)
    print(f"{'ctx':>6}", *[f"{'k='+str(k):>9}" for k in ks], sep="")
    for ctx in ctxs:
        print(f"{ctx:>6}", *[f"{verify_tbl[ctx].get(k, 0):>9.2f}" for k in ks], sep="")

    print()
    print("="*90)
    print("T_draft_max (ms) to beat NPU-only — BUDGET for per-token draft time")
    print("  Negative = spec can never win at this (ctx, k, α), even with T_draft=0")
    print("="*90)
    for ctx in ctxs:
        print(f"\n-- ctx={ctx}, T_base=T_verify(1)={verify_tbl[ctx][1]:.1f} ms --")
        print(f"{'α':>6}", *[f"{'k='+str(k):>9}" for k in ks], sep="")
        for a in alphas:
            row = out["T_draft_max_ms (by ctx, k, alpha)"][ctx]
            vals = []
            for k in ks:
                m = row[k][a]
                vals.append(f"{m:>9.1f}" if m is not None else "       --")
            print(f"{a:>6.2f}", *vals, sep="")

    print()
    print("="*90)
    print("Measured/hypothetical drafts — speedup vs NPU-only (sequential spec)")
    print("  Bold values > 1.00× indicate a win cell.")
    print("="*90)
    for name, info in out["measured_drafts_speedup"].items():
        td, a = info["T_draft_ms"], info["alpha"]
        print(f"\n{name}  (T_draft={td} ms, α_greedy_hit={a:.2f})")
        hdr = "ctx / k"
        print(f"{hdr:>8}", *[f"{'k='+str(k):>8}" for k in ks], sep="")
        for ctx in ctxs:
            vals = []
            for k in ks:
                s = info["speedup_by_ctx_k"][ctx][k]
                if s is None:
                    vals.append("      --")
                else:
                    mark = "*" if s > 1.0 else " "
                    vals.append(f"{mark}{s:>7.2f}")
            print(f"{ctx:>8}", *vals, sep="")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
