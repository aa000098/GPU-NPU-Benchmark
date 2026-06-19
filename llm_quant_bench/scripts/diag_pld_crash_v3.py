"""Diag v3: isolate which component triggers the PLD@ctx=3500 crash.

Three separate sub-tests, each fresh Python process via CLI flag:
  --test A : run_pld only, no DraftEngine loaded
  --test B : run_pld only, DraftEngine loaded (not used)
  --test C : npu_only + sequential first, then run_pld (mimics main order)

If A dies  -> run_pld itself broken at ctx=3500
If A OK, B dies -> DraftEngine presence triggers it
If A,B OK, C dies -> cumulative state (despite v2 repro not showing leak)
"""
import os, sys, time, gc, argparse, functools
from pathlib import Path

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from llm_quant_bench.benchmark.hybrid_runtime_v2 import (
    TargetVerifier, DraftEngine, run_sequential, run_pld, run_npu_only,
)


def rss_mb():
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return -1


def avail_mb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", choices=["A", "B", "C"], required=True)
    p.add_argument("--target_ctx", type=int, default=3500)
    p.add_argument("--n_gen", type=int, default=32)
    p.add_argument("--k", type=int, default=4)
    args = p.parse_args()

    model_path = "llm_quant_bench/rkllm_models/Llama-3.2-1B-Instruct_W8A8_RK3588.rkllm"
    draft_path = "llm_quant_bench/draft_models/Llama-3.2-1B-Instruct-Q4_0.gguf"
    tok_path   = "llm_quant_bench/models/Llama-3.2-1B-Instruct"
    calib_path = "llm_quant_bench/calib_data/calib_data.txt"

    print(f"=== TEST {args.test}  target_ctx={args.target_ctx}  n_gen={args.n_gen}  k={args.k} ===")
    print(f"[{rss_mb()} MB, avail {avail_mb()} MB] tokenize")
    tok = AutoTokenizer.from_pretrained(tok_path)
    text = open(calib_path).read()
    budget = args.target_ctx - args.n_gen - 16
    prompt_tokens = tok.encode(text, add_special_tokens=True)[:budget]
    print(f"[{rss_mb()} MB] prompt={len(prompt_tokens)} tokens")

    print(f"[{rss_mb()} MB] loading target (RKLLM)...")
    target = TargetVerifier(model_path, max_context_len=4096)
    print(f"[{rss_mb()} MB, avail {avail_mb()} MB] target loaded")

    draft = None
    if args.test in ("B", "C"):
        print(f"[{rss_mb()} MB] loading draft (llama.cpp)...")
        draft = DraftEngine(draft_path, n_threads=4, n_ctx=4096)
        print(f"[{rss_mb()} MB, avail {avail_mb()} MB] draft loaded")

    try:
        if args.test == "C":
            print(f"\n[{rss_mb()} MB] >>> Running npu_only (to simulate main warmup)")
            r = run_npu_only(target, prompt_tokens, args.n_gen)
            print(f"[{rss_mb()} MB, avail {avail_mb()} MB] npu_only done: {r['tok_s']:.3f} tok/s")

            print(f"\n[{rss_mb()} MB] >>> Running sequential k={args.k}")
            r = run_sequential(draft, target, prompt_tokens, args.n_gen, args.k)
            print(f"[{rss_mb()} MB, avail {avail_mb()} MB] sequential done: {r['tok_s']:.3f} tok/s")

        print(f"\n[{rss_mb()} MB, avail {avail_mb()} MB] >>> Running run_pld k={args.k}")
        r = run_pld(target, prompt_tokens, args.n_gen, args.k,
                    n_gram_min=2, n_gram_max=3, log_per_round=True)
        print(f"[{rss_mb()} MB, avail {avail_mb()} MB] PLD done: {r['tok_s']:.3f} tok/s")

    finally:
        target.close()
    print("test done")


if __name__ == "__main__":
    main()
