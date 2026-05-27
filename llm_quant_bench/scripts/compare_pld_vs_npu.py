"""Compare PLD vs NPU-only token sequences on the same prompts.

Under greedy accept, PLD output tokens should be bitwise identical to NPU-only
decoding if the target runtime is deterministic. This script empirically checks
that invariant.
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from llm_quant_bench.benchmark.hybrid_runtime_v2 import (
    TargetVerifier, run_pld, run_npu_only,
)


PROMPTS = {
    "copy": "Copy the following text exactly: 'The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.' The exact copy is:",
    "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(",
    "text": "In this short paper, we present a novel approach to edge LLM inference. Our method achieves significant speedup by",
}


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_model_path", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--n_gen", type=int, default=48)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--output_json", default="llm_quant_bench/results/raw/pld_vs_npu_diff.json")
    p.add_argument("--max_context_len", type=int, default=4096)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path)
    target = TargetVerifier(args.target_model_path, max_context_len=args.max_context_len)

    report = {"n_gen": args.n_gen, "k": args.k, "cases": {}}

    try:
        for name, prompt in PROMPTS.items():
            print(f"\n{'='*60}\n[{name}] prompt: {prompt[:60]}...\n{'='*60}")
            prompt_tokens = tok.encode(prompt, add_special_tokens=True)

            print(f"[{name}] npu_only...")
            res_npu = run_npu_only(target, prompt_tokens, args.n_gen)
            tokens_npu = res_npu["accepted_tokens"]

            print(f"[{name}] pld k={args.k}...")
            res_pld = run_pld(target, prompt_tokens, args.n_gen, args.k,
                              n_gram_min=2, n_gram_max=3)
            tokens_pld = res_pld["accepted_tokens"]

            div = first_divergence(tokens_npu, tokens_pld)
            match = (div == -1)

            case = {
                "prompt": prompt,
                "n_prompt_tokens": len(prompt_tokens),
                "npu_only_tok_s": res_npu["tok_s"],
                "pld_tok_s": res_pld["tok_s"],
                "speedup": res_pld["tok_s"] / res_npu["tok_s"],
                "len_npu": len(tokens_npu),
                "len_pld": len(tokens_pld),
                "identical": match,
                "first_divergence_idx": div,
                "tokens_npu": tokens_npu,
                "tokens_pld": tokens_pld,
                "text_npu": tok.decode(tokens_npu),
                "text_pld": tok.decode(tokens_pld),
            }
            report["cases"][name] = case

            print(f"  lengths: npu={len(tokens_npu)}, pld={len(tokens_pld)}")
            print(f"  identical: {match}")
            if not match:
                i = div
                ctx_npu = tokens_npu[max(0, i-3):i+4]
                ctx_pld = tokens_pld[max(0, i-3):i+4]
                print(f"  first divergence at position {i}")
                print(f"    npu[{i-3}:{i+4}] = {ctx_npu}")
                print(f"    pld[{i-3}:{i+4}] = {ctx_pld}")
                print(f"    npu context: ...{repr(tok.decode(tokens_npu[max(0,i-5):i+5]))}")
                print(f"    pld context: ...{repr(tok.decode(tokens_pld[max(0,i-5):i+5]))}")
            print(f"  speedup: {case['speedup']:.2f}×")

    finally:
        target.close()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output_json}")

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for name, c in report["cases"].items():
        status = "IDENTICAL" if c["identical"] else f"DIVERGE@{c['first_divergence_idx']}"
        print(f"  {name:6s}: {status}  (speedup {c['speedup']:.2f}×)")


if __name__ == "__main__":
    main()
