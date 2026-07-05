"""
Parse raw llama-bench output and produce proper draft cost comparison.
Fixes draft_bench.py JSON output parsing issue (test names appeared as empty).

Data source: hand-transcribed from the draft bench log, since the JSON
had empty 'test' names but raw text output was readable.
"""
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# From draft_bench_full.log
DRAFT_RESULTS = {
    "Llama-3.2-1B-Q4_0": {
        "size_gb": 0.765,
        "tg16_tok_s": 24.02,
        "ms_per_tok": 41.64,
        "tokenizer_compat_with_target": True,
    },
    "Llama-3.2-1B-IQ3_M": {
        "size_gb": 0.649,
        "tg16_tok_s": 10.47,
        "ms_per_tok": 95.55,
        "tokenizer_compat_with_target": True,
    },
    "Qwen2.5-0.5B-Q4_0": {
        "size_gb": 0.423,
        "tg16_tok_s": 51.21,
        "ms_per_tok": 19.53,
        "tokenizer_compat_with_target": False,  # Qwen BPE ≠ Llama BPE
    },
    "Llama-3.2-1B-Q4_0_4_8": {
        "status": "load failed on llama-cli 1f30ac0ce",
    },
}

T_VERIFY_K8_MS = 418.0  # from probe v2, ctx=4079 k=8
K_VALUES = [1, 2, 4, 8, 16]


def main():
    rows = []
    print("=" * 110)
    print(f"Draft cost vs NPU verify (T_verify(k=8, ctx=4079) = {T_VERIFY_K8_MS:.1f} ms)")
    print("=" * 110)
    header = f"{'Model':<28} {'ms/tok':>8} " + " ".join(f"{'k='+str(k)+'(ms)':>10}" for k in K_VALUES)
    header += f" {'d/v@k=8':>10} {'tok_compat':>12} {'Verdict':>16}"
    print(header)
    print("-" * 110)

    for name, info in DRAFT_RESULTS.items():
        if "ms_per_tok" not in info:
            continue
        mpt = info["ms_per_tok"]
        costs = [mpt * k for k in K_VALUES]
        ratio = costs[K_VALUES.index(8)] / T_VERIFY_K8_MS
        if ratio < 0.3:
            v = "EXCELLENT"
        elif ratio < 0.5:
            v = "GOOD"
        elif ratio < 0.8:
            v = "MARGINAL"
        else:
            v = "TOO SLOW"
        tc = "yes" if info.get("tokenizer_compat_with_target") else "no (BPE≠)"
        cols = f"{name:<28} {mpt:>8.1f} " + " ".join(f"{c:>10.1f}" for c in costs)
        cols += f" {ratio*100:>9.1f}% {tc:>12} {v:>16}"
        print(cols)
        rows.append({
            "model": name,
            **info,
            "k_costs_ms": dict(zip(K_VALUES, costs)),
            "draft_vs_verify_at_k8": ratio,
            "verdict": v,
        })

    print("\n=== Key observations ===")
    print("  • Qwen2.5-0.5B (diff tokenizer): fastest by 2× but requires retokenize.")
    print("  • Llama Q4_0 (same tokenizer): marginal speedup zone, acceptance likely higher.")
    print("  • IQ3_M: 3-bit decoding overhead makes it SLOWER than Q4_0. Dropped.")
    print("  • Q4_0_4_8: failed to load; not supported by current llama.cpp build.")
    print("")
    print("=== Draft selection ===")
    print("  PRIMARY   — Llama-3.2-1B-Q4_0  (same tokenizer, acceptance upper-bound)")
    print("  SECONDARY — Qwen2.5-0.5B-Q4_0 (retokenize ablation, speed lower-bound)")

    out = {
        "t_verify_k8_ms": T_VERIFY_K8_MS,
        "k_values": K_VALUES,
        "drafts": rows,
    }
    out_path = os.path.join(BASE, "results", "raw", "draft_bench_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
