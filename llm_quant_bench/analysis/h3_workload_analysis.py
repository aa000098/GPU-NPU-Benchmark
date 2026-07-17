"""H3: workload-aware optimal config decision boundaries.

For each (prompt_len, gen_len) workload, compute total inference time per config:
  total = prompt_len / prefill_tok_s + gen_len / decode_tok_s

Determines which config wins at each workload point, producing a routing table.
"""
import csv, os, sys
from pathlib import Path

DATA = Path("/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/paper/v9/v9_data")

def load_csv(p):
    if not p.exists(): return []
    with open(p) as f:
        return list(csv.DictReader(f))

def parse_float(v, default=None):
    try:
        return float(v)
    except:
        return default

def main():
    h1 = load_csv(DATA / "h1_baseline.csv")
    h2 = load_csv(DATA / "h2_extend.csv")
    npu = load_csv(DATA / "h2_npu.csv")

    # Combine into a single config list: (config_id, ctx, prefill, decode)
    configs = {}  # config_id -> {ctx: (prefill, decode)}
    for row in h1 + h2:
        cid = row.get('config_id', '')
        ctx = int(row.get('ctx', 0))
        pf = parse_float(row.get('prefill', ''))
        dc = parse_float(row.get('decode', ''))
        if pf is None or dc is None: continue
        configs.setdefault(cid, {})[ctx] = (pf, dc)

    # FRESH NPU measurements (h2_npu sweep, this run)
    # Format: ctx -> (prefill_tok_s, decode_tok_s)
    NPU_DATA = {
        15:   (15/0.100,  15.92),    # 150 prefill, 15.92 decode
        512:  (512/2.034, 13.82),    # 252 prefill, 13.82 decode
        1024: (1024/4.435, 12.11),   # 231 prefill, 12.11 decode
        2048: (2048/11.256, 10.21),  # 182 prefill, 10.21 decode
        3500: (3500/25.116, 8.85),   # 139 prefill, 8.85 decode
    }
    configs['npu_rkllm_run'] = NPU_DATA

    # Print per-ctx winners
    print("=== Per-ctx winners ===")
    print(f"{'ctx':>5} {'best_decode_config':<20} {'decode_tok/s':>12} {'best_prefill_config':<20} {'prefill_tok/s':>12}")
    test_ctxs = [64, 512, 1024, 2048, 3500]
    for ctx in test_ctxs:
        best_dec = None; best_dec_cid = None
        best_pf  = None; best_pf_cid  = None
        for cid, ctxmap in configs.items():
            if ctx in ctxmap:
                pf, dc = ctxmap[ctx]
                if dc and (best_dec is None or dc > best_dec):
                    best_dec = dc; best_dec_cid = cid
                if pf and (best_pf is None or pf > best_pf):
                    best_pf = pf; best_pf_cid = cid
        if best_dec_cid:
            print(f"{ctx:>5} {best_dec_cid:<20} {best_dec:>12.2f} {best_pf_cid:<20} {best_pf:>12.1f}")

    # Workload table: (prompt_len, gen_len) -> winning config
    print("\n=== End-to-end winners by workload (total_time = prompt/prefill + gen/decode) ===")
    workloads = [
        (64, 32), (64, 100), (64, 500),
        (512, 32), (512, 100), (512, 500),
        (1024, 32), (1024, 100), (1024, 500),
        (2048, 32), (2048, 100), (2048, 500),
    ]
    print(f"{'prompt':>7} {'gen':>5} {'best_config':<25} {'total_s':>10} {'tok/s_avg':>10}")
    routing_table = []
    for prompt, gen in workloads:
        best = None; best_cid = None
        for cid, ctxmap in configs.items():
            # Find closest ctx data
            ctxs_avail = sorted(ctxmap.keys())
            close_ctx = min(ctxs_avail, key=lambda c: abs(c - prompt))
            if abs(close_ctx - prompt) > prompt * 0.5:
                continue  # too far, skip
            pf, dc = ctxmap[close_ctx]
            if not pf or not dc: continue
            total = prompt / pf + gen / dc
            if best is None or total < best:
                best = total; best_cid = cid
        if best:
            tok_s = (prompt + gen) / best
            print(f"{prompt:>7} {gen:>5} {best_cid:<25} {best:>10.3f} {tok_s:>10.2f}")
            routing_table.append((prompt, gen, best_cid, best, tok_s))

    # Save routing table
    out = DATA / "h3_routing_table.csv"
    with open(out, 'w') as f:
        w = csv.writer(f)
        w.writerow(['prompt_len', 'gen_len', 'best_config', 'total_seconds', 'avg_tok_s'])
        for r in routing_table:
            w.writerow(r)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
