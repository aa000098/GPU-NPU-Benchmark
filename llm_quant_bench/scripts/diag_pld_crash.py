"""Minimal reproducer for PLD@ctx=3500 silent crash.

Fresh RKLLM init, then a single verify-style call. If this alone dies,
the issue is in the verify path itself, not in PLD/state-accumulation.

Runs small probes first (ctx=512, 2048) to prove the reproducer works,
then the failing size (3452), all in one process.
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


def main():
    model_path = "llm_quant_bench/rkllm_models/Llama-3.2-1B-Instruct_W8A8_RK3588.rkllm"
    tok_path   = "llm_quant_bench/models/Llama-3.2-1B-Instruct"
    calib_path = "llm_quant_bench/calib_data/calib_data.txt"

    print(f"[{rss_mb()} MB] tokenizing...")
    tok = AutoTokenizer.from_pretrained(tok_path)
    text = open(calib_path).read()
    all_tokens = tok.encode(text, add_special_tokens=True)
    print(f"[{rss_mb()} MB] calib has {len(all_tokens)} tokens")

    print(f"[{rss_mb()} MB] init RKLLM (max_context_len=4096)...")
    handle = init_model(model_path, max_context_len=4096)
    print(f"[{rss_mb()} MB] RKLLM loaded.")

    try:
        for target_len in [512, 2048, 3000, 3200, 3400, 3452, 3488]:
            toks = all_tokens[:target_len]
            print(f"\n[{rss_mb()} MB] >>> verify @ len={len(toks)} ...")
            t0 = time.perf_counter()
            clear_kv_cache(handle)
            logits = get_logits(handle, toks, keep_history=1)
            dt = time.perf_counter() - t0
            if logits is None:
                print(f"[{rss_mb()} MB] FAIL: get_logits returned None at len={len(toks)}")
                break
            print(f"[{rss_mb()} MB] OK len={len(toks)}  shape={logits.shape}  dt={dt:.2f}s")
            del logits
            gc.collect()
            print(f"[{rss_mb()} MB] after gc")
    finally:
        print(f"[{rss_mb()} MB] destroying model")
        destroy_model(handle)
    print(f"[{rss_mb()} MB] done")


if __name__ == "__main__":
    main()
