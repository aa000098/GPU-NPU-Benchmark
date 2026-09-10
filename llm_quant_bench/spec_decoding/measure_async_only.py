"""Measure Always-Async only and merge into existing v3 RegimeSpec JSON.

기존 v3 측정 (NPU/Seq/PLD/RegimeSpec/Oracle)이 완료된 후 Async 결과만
추가로 재고 싶을 때 사용. 워크로드는 기존 build_workload와 같은 시드.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from llm_quant_bench.benchmark.hybrid_runtime_v2 import (
    DraftEngine, TargetVerifier, run_async,
)
from llm_quant_bench.spec_decoding.regimespec import build_workload


def measure_async(target_path, draft_path, tokenizer_path,
                   workload_kind, n_gen=32, k=4):
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    workload = build_workload(tok, kind=workload_kind)

    print(f"\n{'='*70}")
    print(f"Always-Async measurement — workload={workload_kind}, n_gen={n_gen}, k={k}")
    print(f"{'='*70}\n")
    print(f"Workload: {len(workload)} prompts")
    for i, (kind, toks) in enumerate(workload):
        print(f"  [{i}] {kind:<11s} (ctx={len(toks)})")
    print()

    draft = DraftEngine(draft_path, n_threads=4)

    async_results = []
    for kind, toks in workload:
        target = TargetVerifier(target_path)
        t0 = time.perf_counter()
        try:
            r = run_async(draft, target, toks, n_gen, k=k)
        except Exception as e:
            print(f"  [WARN] Async failed on {kind} ctx={len(toks)}: {e}")
            r = {"mode": "async", "tok_s": float("nan"), "error": str(e)}
        finally:
            target.close()
        r["kind"] = kind
        r["ctx_len"] = len(toks)
        r["wall_s"] = time.perf_counter() - t0
        async_results.append(r)
        ts = r.get("tok_s", float("nan"))
        hit = r.get("spec_hit_rate", float("nan"))
        try:
            ts_str = f"{ts:>6.2f}"
        except (ValueError, TypeError):
            ts_str = "  nan "
        try:
            hit_str = f"{hit:.2f}"
        except (ValueError, TypeError):
            hit_str = "nan"
        print(f"  {kind:<11s} ctx={len(toks):>5d}: {ts_str} tok/s  spec_hit={hit_str}")

    valid = [r["tok_s"] for r in async_results
             if isinstance(r.get("tok_s"), (int, float))
             and r["tok_s"] == r["tok_s"]]
    avg_tok_s = sum(valid) / len(valid) if valid else float("nan")
    print(f"\nAvg async tok/s: {avg_tok_s:.3f}")
    return async_results, avg_tok_s


def merge_into_json(existing_json, async_results, avg_async):
    """기존 v3 json을 읽어 async 결과를 추가하고 oracle을 갱신해 다시 저장."""
    with open(existing_json) as f:
        d = json.load(f)

    # Inject async into averages and details
    d.setdefault("averages_tok_s", {})["always_async"] = avg_async
    d.setdefault("details", {})["async"] = async_results

    # Oracle 재계산 (async 후보 포함)
    import math
    def _is_num(x):
        return isinstance(x, (int, float)) and not math.isnan(x)

    npu = d["details"]["npu_only"]
    seq = d["details"]["seq"]
    pld = d["details"]["pld"]
    n = min(len(npu), len(seq), len(pld), len(async_results))
    new_oracle = []
    for i in range(n):
        cands = [r for r in (npu[i], seq[i], async_results[i], pld[i])
                 if _is_num(r.get("tok_s"))]
        if cands:
            best = max(cands, key=lambda r: r["tok_s"])
            new_oracle.append({"kind": npu[i].get("kind"),
                                "ctx_len": npu[i].get("ctx_len"),
                                "tok_s": best["tok_s"],
                                "best_mode": best.get("mode")})
        else:
            new_oracle.append({"kind": npu[i].get("kind"),
                                "ctx_len": npu[i].get("ctx_len"),
                                "tok_s": float("nan"), "best_mode": "none"})
    d["details"]["oracle"] = new_oracle
    valid_o = [r["tok_s"] for r in new_oracle if _is_num(r.get("tok_s"))]
    d["averages_tok_s"]["oracle"] = (sum(valid_o) / len(valid_o)) if valid_o else float("nan")

    with open(existing_json, "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nMerged into: {existing_json}")
    print(f"  always_async = {d['averages_tok_s']['always_async']:.3f}")
    print(f"  oracle (recomputed) = {d['averages_tok_s']['oracle']:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_model", required=True)
    p.add_argument("--draft_model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--workload", choices=["mixed", "short_only", "long_only"], required=True)
    p.add_argument("--n_gen", type=int, default=32)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--existing_json", required=True,
                   help="기존 v3 json 경로 (여기에 async 결과를 merge)")
    args = p.parse_args()

    async_results, avg = measure_async(
        args.target_model, args.draft_model, args.tokenizer,
        args.workload, n_gen=args.n_gen, k=args.k,
    )
    merge_into_json(args.existing_json, async_results, avg)


if __name__ == "__main__":
    main()
