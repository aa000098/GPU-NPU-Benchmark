"""LongBench small-subset evaluation with rotated KV quantization.

Uses HuggingFace `THUDM/LongBench` dataset. To stay tractable on RK3588
(Llama-3.2-1B, max_ctx=4096), runs only short tasks: trec, samsum, triviaqa.

Compares: fp16 baseline vs rotated scalar quant at 4/3/2 bits.
Reports task-specific score (classification accuracy or ROUGE-L).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "llm_quant_bench/eval"))
os.sched_setaffinity(0, {4, 5, 6, 7})

from ppl_rotated_quant import (
    make_rotation, lloyd_max_gaussian,
    patch_attention_with_rotated_quant, restore_attention,
    HEAD_DIM,
)

HF_MODEL = ROOT / "llm_quant_bench/models/Llama-3.2-1B-Instruct"

# Short LongBench tasks (max_length safe for Llama-3.2-1B 4k context)
TASK_PROMPT_FORMATS = {
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passages.\n\n{context}\n\nQuestion: {input}\nAnswer:",
}
TASK_MAX_GEN = {"trec": 16, "samsum": 64, "triviaqa": 32}


def score_classification(pred, gold_list):
    p = pred.strip().lower().split("\n")[0]
    return float(any(g.lower() in p or p in g.lower() for g in gold_list))


def score_rouge(pred, gold_list):
    """Lightweight ROUGE-L F1 against best of gold answers."""
    def lcs(a, b):
        n, m = len(a), len(b)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n):
            for j in range(m):
                dp[i+1][j+1] = dp[i][j]+1 if a[i]==b[j] else max(dp[i+1][j], dp[i][j+1])
        return dp[n][m]
    p = pred.strip().split()
    best = 0.0
    for g in gold_list:
        gw = g.strip().split()
        if not p or not gw: continue
        L = lcs(p, gw)
        if L == 0: continue
        prec = L/len(p); rec = L/len(gw)
        f1 = 2*prec*rec/(prec+rec)
        best = max(best, f1)
    return best


SCORERS = {"trec": score_classification, "samsum": score_rouge, "triviaqa": score_classification}


def run_task(model, tok, task, n_samples, max_input_len=2048):
    import torch
    from datasets import load_dataset
    ds = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
    fmt = TASK_PROMPT_FORMATS[task]
    max_gen = TASK_MAX_GEN[task]
    scorer = SCORERS[task]
    scores = []
    for i, ex in enumerate(ds):
        if i >= n_samples: break
        prompt = fmt.format(context=ex.get("context", ""), input=ex.get("input", ""))
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=max_input_len)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_gen, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        gen_ids = out[0][inp["input_ids"].shape[1]:].cpu().numpy().tolist()
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        gold = ex.get("answers", [])
        if isinstance(gold, str): gold = [gold]
        s = scorer(gen_text, gold)
        scores.append(s)
    return float(np.mean(scores)) if scores else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=str, default="4,3,2")
    ap.add_argument("--tasks", type=str, default="trec,samsum,triviaqa")
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--max_input_len", type=int, default=2048)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("[load] Llama-3.2-1B...")
    tok = AutoTokenizer.from_pretrained(str(HF_MODEL))
    model = AutoModelForCausalLM.from_pretrained(str(HF_MODEL), torch_dtype=torch.float16)
    model.eval()

    R = make_rotation(HEAD_DIM, seed=42)
    R_t = torch.from_numpy(R)
    bit_rates = [int(b) for b in args.bits.split(",")]
    codebooks_t = {b: torch.from_numpy(lloyd_max_gaussian(b) / math.sqrt(HEAD_DIM)) for b in bit_rates}

    tasks = args.tasks.split(",")

    print(f"\n=== LongBench subset (tasks={tasks}, n={args.n_samples}) ===\n")
    print(f"{'task':>12} {'fp16':>8} " + " ".join(f"{b}-bit".rjust(8) for b in bit_rates))

    results = {}
    for task in tasks:
        row = {}
        t0 = time.perf_counter()
        s_fp = run_task(model, tok, task, args.n_samples, args.max_input_len)
        row["fp16"] = s_fp
        for b in bit_rates:
            cb_t = codebooks_t[b]
            handles = patch_attention_with_rotated_quant(model, R_t, cb_t)
            try:
                s_q = run_task(model, tok, task, args.n_samples, args.max_input_len)
            finally:
                restore_attention(handles)
            row[f"{b}bit"] = s_q
        line = f"{task:>12} {row['fp16']:>8.3f} "
        for b in bit_rates:
            line += f"{row[f'{b}bit']:>8.3f} "
        line += f"  ({time.perf_counter()-t0:.0f}s)"
        print(line)
        results[task] = row

    print("\n=== JSON results ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
