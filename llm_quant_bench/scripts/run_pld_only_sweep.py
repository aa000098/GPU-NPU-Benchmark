"""PLD-only sweep across ctx, no DraftEngine loaded.

For comparison against the previous run_longctx_hybrid.py runs where
DraftEngine was loaded alongside. If throughput matches, previous numbers
are valid under the alternative config. If not, we update them.
"""
import argparse, json, os, sys, time, functools
from pathlib import Path

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from llm_quant_bench.benchmark.hybrid_runtime_v2 import TargetVerifier, run_pld


def build_prompt(tok, calib_path, target_ctx, n_gen, margin=16):
    text = open(calib_path).read()
    budget = target_ctx - n_gen - margin
    toks = tok.encode(text, add_special_tokens=True)
    if len(toks) < budget:
        raise RuntimeError(f"calib has only {len(toks)} tokens, need {budget}")
    return toks[:budget]


def rss_mb():
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_model_path", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--calib_path", default="llm_quant_bench/calib_data/calib_data.txt")
    p.add_argument("--ctx_list", default="1024,2048,3500")
    p.add_argument("--n_gen", type=int, default=32)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max_context_len", type=int, default=4096)
    p.add_argument("--output_json", default="llm_quant_bench/results/raw/pld_only_sweep.json")
    args = p.parse_args()

    ctx_list = [int(x) for x in args.ctx_list.split(",")]
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"[{rss_mb()} MB] loading target (RKLLM, no DraftEngine)...")
    target = TargetVerifier(args.target_model_path, max_context_len=args.max_context_len)
    print(f"[{rss_mb()} MB] target loaded\n")

    out = {"args": vars(args), "runs": []}

    try:
        for ctx in ctx_list:
            prompt_tokens = build_prompt(tok, args.calib_path, ctx, args.n_gen)
            print(f"\n{'='*60}\n# target_ctx={ctx}  prompt={len(prompt_tokens)}\n{'='*60}")
            print(f"[{rss_mb()} MB] running run_pld k={args.k}, n_gen={args.n_gen}")
            r = run_pld(target, prompt_tokens, args.n_gen, args.k,
                        n_gram_min=2, n_gram_max=3)
            print(f"[{rss_mb()} MB] tok/s={r['tok_s']:.3f}  ms/tok={r['ms_per_token']:.1f}  "
                  f"rounds={r['rounds']}  accept/round={r['mean_accepted_per_round']:.2f}  "
                  f"ngram_hit={r['ngram_hit_rate']:.2f}")
            clean = {k: v for k, v in r.items() if k not in ("accepted_tokens", "trace")}
            out["runs"].append({
                "target_ctx": ctx,
                "n_prompt_tokens": len(prompt_tokens),
                "results": {"pld": clean},
            })
            os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
            json.dump(out, open(args.output_json, "w"), indent=2)
            print(f"[{rss_mb()} MB] checkpoint saved")
    finally:
        target.close()

    print(f"\n{'='*60}\nPLD-only (no DraftEngine) summary\n{'='*60}")
    print(f"{'ctx':>6} | {'tok/s':>8} | {'ms/tok':>8} | accept/round | ngram_hit")
    for r in out["runs"]:
        pr = r["results"]["pld"]
        print(f"{r['target_ctx']:>6} | {pr['tok_s']:>8.3f} | {pr['ms_per_token']:>8.1f} | "
              f"{pr['mean_accepted_per_round']:>12.2f} | {pr['ngram_hit_rate']:>9.2f}")


if __name__ == "__main__":
    main()
