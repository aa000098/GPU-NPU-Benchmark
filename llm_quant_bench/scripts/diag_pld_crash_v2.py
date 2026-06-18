"""Diag v2: 50 identical-ctx verify calls to match the real PLD workload.

If memory grows unboundedly across identical calls, confirms leak and predicts
where OOM hits. If stable, rules out leak hypothesis and we look elsewhere.
"""
import os, sys, time, gc, functools
from pathlib import Path

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from llm_quant_bench.eval.ppl_rkllm import init_model, get_logits, clear_kv_cache, destroy_model


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
    model_path = "llm_quant_bench/rkllm_models/Llama-3.2-1B-Instruct_W8A8_RK3588.rkllm"
    tok_path   = "llm_quant_bench/models/Llama-3.2-1B-Instruct"
    calib_path = "llm_quant_bench/calib_data/calib_data.txt"

    tok = AutoTokenizer.from_pretrained(tok_path)
    text = open(calib_path).read()
    all_tokens = tok.encode(text, add_special_tokens=True)
    toks = all_tokens[:3488]
    print(f"Using ctx={len(toks)}")

    print(f"[{rss_mb()} MB, avail {avail_mb()} MB] init RKLLM...")
    handle = init_model(model_path, max_context_len=4096)
    print(f"[{rss_mb()} MB, avail {avail_mb()} MB] loaded.")

    try:
        N = 50
        for i in range(N):
            clear_kv_cache(handle)
            t0 = time.perf_counter()
            logits = get_logits(handle, toks, keep_history=1)
            dt = time.perf_counter() - t0
            if logits is None:
                print(f"[iter {i}] FAIL: get_logits returned None")
                break
            del logits
            if i < 5 or i % 5 == 0 or i == N-1:
                gc.collect()
                print(f"[iter {i:3d}] dt={dt:5.2f}s  RSS={rss_mb()} MB  avail={avail_mb()} MB")
    finally:
        print(f"[final RSS={rss_mb()} MB] destroying.")
        destroy_model(handle)
    print("done")


if __name__ == "__main__":
    main()
