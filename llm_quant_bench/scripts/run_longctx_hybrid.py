"""Long-ctx hybrid measurement: validate (or falsify) the feasibility scan.

Scan predicts at ctx=4079, Llama 1B Q4_0 draft should win sequential spec
(1.20× at k=4). That's under *staged verify* (Mode C). Our hybrid_runtime_v2.py
actually uses V1 full-prefix verify, which is strictly worse. This script
measures the real thing at long ctx and compares to the scan prediction.

Prompt: first N tokens of calib_data.txt (wikitext), ask the model to
continue. n_gen=32.
"""
import argparse, json, os, sys, time, functools
from pathlib import Path
print = functools.partial(print, flush=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from llm_quant_bench.benchmark.hybrid_runtime_v2 import (
    TargetVerifier, DraftEngine, run_sequential, run_pld, run_npu_only,
)


def build_prompt(tok, calib_path, target_ctx, n_gen, margin=16):
    text = open(calib_path).read()
    budget = target_ctx - n_gen - margin
    toks = tok.encode(text, add_special_tokens=True)
    if len(toks) < budget:
        raise RuntimeError(f"calib text has only {len(toks)} tokens, need {budget}")
    return toks[:budget]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_model_path", required=True)
    p.add_argument("--draft_model_path", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--calib_path", default="llm_quant_bench/calib_data/calib_data.txt")
    p.add_argument("--ctx_list", default="512,1024,2048,3500",
                   help="Comma-separated target ctx lengths")
    p.add_argument("--n_gen", type=int, default=32)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max_context_len", type=int, default=4096)
    p.add_argument("--output_json", default="llm_quant_bench/results/raw/longctx_hybrid.json")
    p.add_argument("--modes", default="npu_only,sequential,pld",
                   help="Comma-separated modes in execution order")
    args = p.parse_args()

    ctx_list = [int(x) for x in args.ctx_list.split(",")]
    modes = args.modes.split(",")
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)

    print(f"Loading target (RKLLM)...")
    target = TargetVerifier(args.target_model_path, max_context_len=args.max_context_len)
    print(f"Loading draft (llama.cpp)...")
    draft = DraftEngine(args.draft_model_path, n_threads=4, n_ctx=args.max_context_len)

    all_out = {"args": vars(args), "runs": []}

    try:
        for target_ctx in ctx_list:
            print(f"\n\n{'#'*70}\n# target_ctx={target_ctx}\n{'#'*70}")
            prompt_tokens = build_prompt(tok, args.calib_path, target_ctx, args.n_gen)
            print(f"Prompt: {len(prompt_tokens)} tokens (target_ctx={target_ctx})")

            run_entry = {
                "target_ctx": target_ctx,
                "n_prompt_tokens": len(prompt_tokens),
                "n_gen": args.n_gen,
                "k": args.k,
                "results": {},
            }

            for mode in modes:
                print(f"\n--- {mode} (k={args.k if mode != 'npu_only' else 1}) ---")
                t0 = time.perf_counter()
                if mode == "npu_only":
                    res = run_npu_only(target, prompt_tokens, args.n_gen)
                elif mode == "sequential":
                    res = run_sequential(draft, target, prompt_tokens, args.n_gen, args.k)
                elif mode == "pld":
                    res = run_pld(target, prompt_tokens, args.n_gen, args.k,
                                  n_gram_min=2, n_gram_max=3)
                wall = time.perf_counter() - t0
                print(f"  tok/s={res['tok_s']:.2f}, ms/tok={res['ms_per_token']:.1f}, "
                      f"total={res['total_ms']:.0f}ms (wall={wall*1000:.0f}ms)")
                if "rounds" in res:
                    print(f"  rounds={res['rounds']}, avg accepted/round={res['mean_accepted_per_round']:.2f}")
                if "ngram_hit_rate" in res:
                    print(f"  ngram_hit_rate={res['ngram_hit_rate']:.2f}")
                clean = {kk: vv for kk, vv in res.items() if kk not in ("accepted_tokens", "trace")}
                run_entry["results"][mode] = clean

            # Per-ctx summary
            base = run_entry["results"]["npu_only"]["tok_s"]
            print(f"\n  Summary ctx={target_ctx}:")
            for m, r in run_entry["results"].items():
                print(f"    {m:12s}: {r['tok_s']:>6.2f} tok/s  ({r['tok_s']/base:.2f}×)")

            all_out["runs"].append(run_entry)

            # Incremental save after each ctx
            os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
            json.dump(all_out, open(args.output_json, "w"), indent=2)
            print(f"  Saved checkpoint -> {args.output_json}")

    finally:
        target.close()

    print(f"\n{'='*70}\nAll runs complete. Final summary:\n{'='*70}")
    print(f"{'ctx':>6} | {'npu_only':>10} | {'sequential':>12} | {'pld':>10}")
    for run_entry in all_out["runs"]:
        r = run_entry["results"]
        base = r["npu_only"]["tok_s"]
        print(f"{run_entry['target_ctx']:>6} | {r['npu_only']['tok_s']:>10.2f} | "
              f"{r['sequential']['tok_s']:>6.2f} ({r['sequential']['tok_s']/base:>4.2f}×) | "
              f"{r['pld']['tok_s']:>4.2f} ({r['pld']['tok_s']/base:>4.2f}×)")


if __name__ == "__main__":
    main()
