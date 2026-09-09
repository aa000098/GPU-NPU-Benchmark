"""
RegimeSpec: Context-length 기반 자동 전환 speculative decoding 시스템

Policy:
- ctx_len < THRESHOLD: PLD (draft-free, 짧은 prompt에서 우세)
- ctx_len >= THRESHOLD: Sequential draft (긴 prompt에서 우세)

기존 hybrid_runtime_v2의 run_pld, run_sequential, run_npu_only 재활용.

Usage:
    python regimespec.py --target_model rkllm_models/Llama-3.2-1B-Instruct_W8A8_RK3588.rkllm \\
        --draft_model draft_models/Llama-3.2-1B-Instruct-Q4_0.gguf \\
        --tokenizer models/Llama-3.2-1B-Instruct \\
        --workload mixed \\
        --output_json results/regimespec.json
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from llm_quant_bench.benchmark.hybrid_runtime_v2 import (
    DraftEngine, TargetVerifier, run_sequential, run_async, run_pld, run_npu_only,
)


class RegimeSpec:
    """Context-length adaptive speculative decoding dispatcher."""

    def __init__(self, target_path, draft_path, max_context_len=4096,
                 threshold=1024, k_draft=4, k_pld=4):
        self.threshold = threshold
        self.k_draft = k_draft
        self.k_pld = k_pld

        # Target은 항상 필요 (PLD든 Seq든 verify 단계는 NPU)
        print(f"[RegimeSpec] Loading target model...")
        t0 = time.perf_counter()
        self.target = TargetVerifier(target_path, max_context_len=max_context_len)
        self.t_target_load = time.perf_counter() - t0

        # Draft는 lazy-load (long prompt에서만 쓰임)
        self.draft_path = draft_path
        self.draft = None
        self.t_draft_load = None

        # 통계
        self.stats = {"pld_count": 0, "seq_count": 0, "switching_overhead_ms": 0.0}

    def _load_draft_if_needed(self):
        if self.draft is None:
            print(f"[RegimeSpec] Lazy-loading draft model...")
            t0 = time.perf_counter()
            self.draft = DraftEngine(self.draft_path, n_threads=4, n_ctx=4096)
            self.t_draft_load = time.perf_counter() - t0
            self.stats["switching_overhead_ms"] += self.t_draft_load * 1000

    def generate(self, prompt_tokens, n_gen):
        """Auto-route to PLD or Sequential based on context length."""
        ctx_len = len(prompt_tokens)
        t_dispatch_start = time.perf_counter()

        if ctx_len < self.threshold:
            mode = "PLD"
            self.stats["pld_count"] += 1
            result = run_pld(self.target, prompt_tokens, n_gen, self.k_pld,
                             n_gram_min=2, n_gram_max=3)
        else:
            mode = "Sequential"
            self._load_draft_if_needed()
            self.stats["seq_count"] += 1
            result = run_sequential(self.draft, self.target, prompt_tokens,
                                     n_gen, self.k_draft)

        t_dispatch = time.perf_counter() - t_dispatch_start
        result["regime_mode"] = mode
        result["dispatch_time_s"] = t_dispatch
        result["ctx_len"] = ctx_len
        return result

    def close(self):
        self.target.close()


def build_workload(tokenizer, kind="mixed", n_short=10, n_long=10, seed=42):
    """Generate mixed workload of short and long prompts."""
    random.seed(seed)

    # 두 클래스만 사용: natural (자연어 Q&A/대화) + code (함수/코드 재작성).
    # 순수 반복(copy) 클래스는 PLD에 인공적으로 유리하므로 제외 (v2 §4.3 한계).
    short_natural = [
        "What is the capital of France, and what is it famous for?",
        "Explain in two sentences why the sky appears blue during the day.",
        "Define machine learning and give one real-world example.",
        "Summarize the plot of Hamlet in three sentences.",
        "What are the main causes of inflation in a modern economy?",
    ]
    short_code = [
        "Rewrite this Python function to use list comprehension:\n"
        "def squares(n):\n    out = []\n    for i in range(n):\n        out.append(i*i)\n    return out\n",
        "Convert the following loop to a NumPy vectorized version:\n"
        "for i in range(len(a)):\n    c[i] = a[i] * b[i] + 3\n",
        "Add type hints and a docstring to this function:\n"
        "def add(a, b):\n    return a + b\n",
        "Refactor this function to remove the nested if:\n"
        "def grade(s):\n    if s >= 60:\n        if s >= 80:\n            return 'A'\n        return 'B'\n    return 'F'\n",
        "Translate this Python snippet to JavaScript:\n"
        "def fact(n):\n    if n <= 1: return 1\n    return n * fact(n-1)\n",
    ]
    short_prompts = short_natural + short_code
    short_kinds = ["short_nat"] * len(short_natural) + ["short_code"] * len(short_code)

    # Wikitext-2 long passages (자연어 산문, n-gram 자기반복이 희박해야 PLD가 wiki 분포에 부합)
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    wiki_text = "\n".join(t for t in ds["text"] if len(t.strip()) > 100)
    full_toks = tokenizer.encode(wiki_text, add_special_tokens=False)

    long_prompts = []
    target_lens = [1024, 1500, 2048, 2500, 3000]
    rng = random.Random(seed)
    for tlen in target_lens:
        # 서로 다른 wiki 구간을 무작위로 잘라 prompt 다양성 확보
        start = rng.randint(0, max(0, len(full_toks) - tlen - 1))
        toks = [tokenizer.bos_token_id] + full_toks[start:start + tlen - 1]
        long_prompts.append(toks[:tlen])

    # Convert short to tokens (kind 보존)
    short_tokens = [(k, tokenizer.encode(p, add_special_tokens=True))
                    for k, p in zip(short_kinds, short_prompts)]

    if kind == "short_only":
        items = short_tokens[:n_short]
    elif kind == "long_only":
        items = [("long_wiki", t) for t in long_prompts[:n_long]]
    else:  # mixed
        items = (short_tokens[:n_short]
                 + [("long_wiki", t) for t in long_prompts[:n_long]])
        random.shuffle(items)

    return items


def benchmark(target_path, draft_path, tokenizer_path, workload_kind="mixed",
              n_gen=32, threshold=1024, output_json=None):
    """Run RegimeSpec vs baselines on a workload."""
    print(f"\n{'='*70}")
    print(f"RegimeSpec Benchmark — workload={workload_kind}, n_gen={n_gen}")
    print(f"{'='*70}\n")

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    workload = build_workload(tok, kind=workload_kind)

    print(f"Workload: {len(workload)} prompts")
    for i, (kind, toks) in enumerate(workload):
        print(f"  [{i}] {kind:<11s} (ctx={len(toks)})")
    print()

    # 매 prompt마다 TargetVerifier를 새로 만들어 NPU 누적 상태를 reset하고,
    # 한 prompt가 실패해도 다음 prompt가 NaN으로 기록만 되고 진행되도록 감싼다.
    def _safe_npu(kind, toks):
        target = TargetVerifier(target_path)
        try:
            r = run_npu_only(target, toks, n_gen)
        except Exception as e:
            print(f"  [WARN] NPU-only failed on {kind} ctx={len(toks)}: {e}")
            r = {"mode": "npu_only", "tok_s": float("nan"), "error": str(e)}
        finally:
            target.close()
        return r

    def _safe_seq(kind, toks, draft):
        target = TargetVerifier(target_path)
        try:
            r = run_sequential(draft, target, toks, n_gen, k=4)
        except Exception as e:
            print(f"  [WARN] Sequential failed on {kind} ctx={len(toks)}: {e}")
            r = {"mode": "sequential", "tok_s": float("nan"), "error": str(e)}
        finally:
            target.close()
        return r

    def _safe_async(kind, toks, draft):
        target = TargetVerifier(target_path)
        try:
            r = run_async(draft, target, toks, n_gen, k=4)
        except Exception as e:
            print(f"  [WARN] Async failed on {kind} ctx={len(toks)}: {e}")
            r = {"mode": "async", "tok_s": float("nan"), "error": str(e)}
        finally:
            target.close()
        return r

    def _safe_pld(kind, toks):
        target = TargetVerifier(target_path)
        try:
            r = run_pld(target, toks, n_gen, k=4, n_gram_min=2, n_gram_max=3)
        except Exception as e:
            print(f"  [WARN] PLD failed on {kind} ctx={len(toks)}: {e}")
            r = {"mode": "pld", "tok_s": float("nan"), "error": str(e)}
        finally:
            target.close()
        return r

    # ── 1. NPU-only baseline ──
    print(f"{'─'*60}\n[NPU-only baseline]\n{'─'*60}")
    npu_only_results = []
    for kind, toks in workload:
        t0 = time.perf_counter()
        r = _safe_npu(kind, toks)
        r["kind"] = kind
        r["ctx_len"] = len(toks)
        r["wall_s"] = time.perf_counter() - t0
        npu_only_results.append(r)
        ts = r.get("tok_s", float("nan"))
        print(f"  {kind:<11s} ctx={len(toks):>5d}: {ts:>6.2f} tok/s")

    # ── 2. Always-Sequential ──
    print(f"\n{'─'*60}\n[Always-Sequential (k=4)]\n{'─'*60}")
    draft = DraftEngine(draft_path, n_threads=4)
    seq_results = []
    for kind, toks in workload:
        t0 = time.perf_counter()
        r = _safe_seq(kind, toks, draft)
        r["kind"] = kind
        r["ctx_len"] = len(toks)
        r["wall_s"] = time.perf_counter() - t0
        seq_results.append(r)
        ts = r.get("tok_s", float("nan"))
        print(f"  {kind:<11s} ctx={len(toks):>5d}: {ts:>6.2f} tok/s")

    # ── 2.5 Always-Async (speculative prefetch overlap) ──
    print(f"\n{'─'*60}\n[Always-Async (k=4)]\n{'─'*60}")
    async_results = []
    for kind, toks in workload:
        t0 = time.perf_counter()
        r = _safe_async(kind, toks, draft)
        r["kind"] = kind
        r["ctx_len"] = len(toks)
        r["wall_s"] = time.perf_counter() - t0
        async_results.append(r)
        ts = r.get("tok_s", float("nan"))
        hit = r.get("spec_hit_rate", float("nan"))
        print(f"  {kind:<11s} ctx={len(toks):>5d}: {ts:>6.2f} tok/s  spec_hit={hit:.2f}")

    # ── 3. Always-PLD ──
    print(f"\n{'─'*60}\n[Always-PLD (k=4)]\n{'─'*60}")
    pld_results = []
    for kind, toks in workload:
        t0 = time.perf_counter()
        r = _safe_pld(kind, toks)
        r["kind"] = kind
        r["ctx_len"] = len(toks)
        r["wall_s"] = time.perf_counter() - t0
        pld_results.append(r)
        ts = r.get("tok_s", float("nan"))
        print(f"  {kind:<11s} ctx={len(toks):>5d}: {ts:>6.2f} tok/s")

    # ── 4. RegimeSpec (auto) — prompt마다 fresh handle로 재시도 ──
    print(f"\n{'─'*60}\n[RegimeSpec (auto, threshold={threshold})]\n{'─'*60}")
    rs_results = []
    rs_stats = {"pld_count": 0, "seq_count": 0, "switching_overhead_ms": 0.0}
    for kind, toks in workload:
        t0 = time.perf_counter()
        rs = RegimeSpec(target_path, draft_path, threshold=threshold)
        try:
            r = rs.generate(toks, n_gen)
        except Exception as e:
            print(f"  [WARN] RegimeSpec failed on {kind} ctx={len(toks)}: {e}")
            mode = "PLD" if len(toks) < threshold else "Sequential"
            r = {"mode": "regimespec", "regime_mode": mode,
                 "tok_s": float("nan"), "ctx_len": len(toks), "error": str(e)}
        finally:
            for k_, v_ in rs.stats.items():
                if isinstance(v_, (int, float)):
                    rs_stats[k_] = rs_stats.get(k_, 0) + v_
            rs.close()
        r["kind"] = kind
        r["wall_s"] = time.perf_counter() - t0
        rs_results.append(r)
        ts = r.get("tok_s", float("nan"))
        print(f"  {kind:<11s} ctx={len(toks):>5d}: {ts:>6.2f} tok/s [{r.get('regime_mode','?')}]")

    # ── 5. Oracle (best of {NPU-only, Seq, PLD} per prompt, NaN 회복) ──
    import math
    def _is_num(x):
        return isinstance(x, (int, float)) and not math.isnan(x)

    oracle_results = []
    for npu_r, seq_r, async_r, pld_r in zip(
            npu_only_results, seq_results, async_results, pld_results):
        cands = [r for r in (npu_r, seq_r, async_r, pld_r) if _is_num(r.get("tok_s"))]
        if cands:
            best = max(cands, key=lambda r: r["tok_s"])
            oracle_results.append({"kind": npu_r["kind"], "ctx_len": npu_r["ctx_len"],
                                    "tok_s": best["tok_s"], "best_mode": best["mode"]})
        else:
            oracle_results.append({"kind": npu_r["kind"], "ctx_len": npu_r["ctx_len"],
                                    "tok_s": float("nan"), "best_mode": "none"})

    # ── Summary (NaN 무시 평균) ──
    def avg(rs):
        valid = [r["tok_s"] for r in rs if _is_num(r.get("tok_s"))]
        return sum(valid) / len(valid) if valid else 0

    def avg_by_kind(rs):
        out = {}
        kinds = sorted({r["kind"] for r in rs})
        for k in kinds:
            sub = [r["tok_s"] for r in rs if r["kind"] == k and _is_num(r.get("tok_s"))]
            out[k] = sum(sub) / len(sub) if sub else float("nan")
        return out

    summary = {
        "workload_kind": workload_kind,
        "n_prompts": len(workload),
        "n_gen": n_gen,
        "threshold": threshold,
        "averages_tok_s": {
            "npu_only": avg(npu_only_results),
            "always_seq": avg(seq_results),
            "always_async": avg(async_results),
            "always_pld": avg(pld_results),
            "regimespec": avg(rs_results),
            "oracle": avg(oracle_results),
        },
        "averages_by_kind_tok_s": {
            "npu_only": avg_by_kind(npu_only_results),
            "always_seq": avg_by_kind(seq_results),
            "always_async": avg_by_kind(async_results),
            "always_pld": avg_by_kind(pld_results),
            "regimespec": avg_by_kind(rs_results),
            "oracle": avg_by_kind(oracle_results),
        },
        "regimespec_stats": rs_stats,
        "details": {
            "npu_only": npu_only_results,
            "seq": seq_results,
            "async": async_results,
            "pld": pld_results,
            "regimespec": rs_results,
            "oracle": oracle_results,
        },
    }

    print(f"\n{'='*70}\nSummary (avg tok/s across {len(workload)} prompts)\n{'='*70}")
    avgs = summary["averages_tok_s"]
    print(f"  NPU-only:     {avgs['npu_only']:>6.2f}")
    print(f"  Always-Seq:   {avgs['always_seq']:>6.2f}")
    print(f"  Always-Async: {avgs['always_async']:>6.2f}")
    print(f"  Always-PLD:   {avgs['always_pld']:>6.2f}")
    print(f"  RegimeSpec:   {avgs['regimespec']:>6.2f}  ←")
    print(f"  Oracle:       {avgs['oracle']:>6.2f}")
    print(f"\nRegimeSpec stats: PLD calls={rs_stats['pld_count']}, "
          f"Seq calls={rs_stats['seq_count']}, "
          f"draft load overhead={rs_stats['switching_overhead_ms']:.0f}ms")

    rs_vs_seq = avgs["regimespec"] / avgs["always_seq"] if avgs["always_seq"] > 0 else 0
    rs_vs_pld = avgs["regimespec"] / avgs["always_pld"] if avgs["always_pld"] > 0 else 0
    rs_vs_oracle = avgs["regimespec"] / avgs["oracle"] if avgs["oracle"] > 0 else 0
    print(f"\n  RegimeSpec / Always-Seq: {rs_vs_seq:.2f}×")
    print(f"  RegimeSpec / Always-PLD: {rs_vs_pld:.2f}×")
    print(f"  RegimeSpec / Oracle:     {rs_vs_oracle:.2f}× (1.0 = perfect)")

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved: {output_json}")

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_model", required=True, help="RKLLM target model (.rkllm)")
    p.add_argument("--draft_model", required=True, help="GGUF draft model")
    p.add_argument("--tokenizer", required=True, help="HuggingFace tokenizer dir")
    p.add_argument("--workload", choices=["mixed", "short_only", "long_only"], default="mixed")
    p.add_argument("--n_gen", type=int, default=32)
    p.add_argument("--threshold", type=int, default=1024)
    p.add_argument("--output_json", default=None)
    args = p.parse_args()

    benchmark(args.target_model, args.draft_model, args.tokenizer,
              workload_kind=args.workload, n_gen=args.n_gen,
              threshold=args.threshold, output_json=args.output_json)


if __name__ == "__main__":
    main()
