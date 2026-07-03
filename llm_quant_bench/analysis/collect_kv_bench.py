"""
Phase 2-2 결과 수집: quant_qkv별 CPU decode 속도 비교

kv_bench tmux 로그(results/raw/kv_bench_full.log)에서
qatten=0/1/2 × prompt=32,128,512,1024,2048,4096 결과 파싱.

Usage:
    python analysis/collect_kv_bench.py
"""
import os
import re
import csv
import json
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "results", "raw", "kv_bench_full.log")
OUTPUT_CSV = os.path.join(BASE_DIR, "results", "raw", "kv_bench_summary.csv")


def parse_log(log_path):
    """Parse tmux log into structured results.

    Log format:
        === qatten=0 (quant_qkv=8) ===
        --- prompt=32, quant_qkv=8 ---
        | M9_w8a8... | pp32 | 139.06 ± 2.62 |
        | M9_w8a8... | tg64 | 16.07 ± 0.17 |
    """
    results = []
    current_qkv = None
    current_prompt = None

    with open(log_path) as f:
        lines = f.read().splitlines()

    for line in lines:
        m = re.search(r"quant_qkv=(\d+)", line)
        if m and line.startswith("==="):
            current_qkv = int(m.group(1))
            continue
        m = re.search(r"prompt=(\d+), quant_qkv=(\d+)", line)
        if m:
            current_prompt = int(m.group(1))
            current_qkv = int(m.group(2))
            continue

        # Parse result lines
        pp_match = re.search(r"pp(\d+)\s*\|\s*([\d.]+)\s*±\s*([\d.]+)", line)
        tg_match = re.search(r"tg(\d+)\s*\|\s*([\d.]+)\s*±\s*([\d.]+)", line)

        if pp_match and current_qkv is not None:
            prompt_len = int(pp_match.group(1))
            value = float(pp_match.group(2))
            std = float(pp_match.group(3))
            results.append({
                "quant_qkv": current_qkv,
                "prompt_len": prompt_len,
                "metric": "prefill",
                "tok_s": value,
                "std": std,
            })
        elif tg_match and current_qkv is not None and current_prompt is not None:
            value = float(tg_match.group(2))
            std = float(tg_match.group(3))
            results.append({
                "quant_qkv": current_qkv,
                "prompt_len": current_prompt,
                "metric": "decode",
                "tok_s": value,
                "std": std,
            })

    return results


def make_summary_table(results):
    """Pivot into table: rows=prompt_len, cols=(qkv_8, qkv_9, qkv_10)×(prefill, decode)"""
    table = {}
    for r in results:
        p = r["prompt_len"]
        table.setdefault(p, {})
        key = f"qkv{r['quant_qkv']}_{r['metric']}"
        table[p][key] = r["tok_s"]
    return table


def print_comparison(table):
    print("\n=== MNN CPU KV Compression Comparison (M9 W8A8, Llama 3.2 1B) ===\n")
    print(f"{'Context':>8}", end="")
    for qkv in [8, 9, 10]:
        print(f"  {'qkv='+str(qkv)+' prefill':>16}  {'qkv='+str(qkv)+' decode':>16}", end="")
    print()
    print("-" * 115)
    for p in sorted(table.keys()):
        print(f"{p:>8}", end="")
        for qkv in [8, 9, 10]:
            pp = table[p].get(f"qkv{qkv}_prefill")
            tg = table[p].get(f"qkv{qkv}_decode")
            pp_str = f"{pp:>14.2f}" if pp else "N/A".rjust(14)
            tg_str = f"{tg:>14.2f}" if tg else "N/A".rjust(14)
            print(f"  {pp_str}    {tg_str}    ", end="")
        print()
    print()

    # Relative comparison
    print("\n=== Relative Change vs qkv=8 baseline ===\n")
    print(f"{'Context':>8}", end="")
    for qkv in [9, 10]:
        print(f"  {f'qkv={qkv} prefill Δ':>20}  {f'qkv={qkv} decode Δ':>18}", end="")
    print()
    print("-" * 90)
    for p in sorted(table.keys()):
        print(f"{p:>8}", end="")
        base_pp = table[p].get("qkv8_prefill")
        base_tg = table[p].get("qkv8_decode")
        for qkv in [9, 10]:
            pp = table[p].get(f"qkv{qkv}_prefill")
            tg = table[p].get(f"qkv{qkv}_decode")
            pp_delta = (pp - base_pp) / base_pp * 100 if pp and base_pp else None
            tg_delta = (tg - base_tg) / base_tg * 100 if tg and base_tg else None
            pp_str = f"{pp_delta:+18.1f}%" if pp_delta is not None else "N/A".rjust(20)
            tg_str = f"{tg_delta:+16.1f}%" if tg_delta is not None else "N/A".rjust(18)
            print(f"  {pp_str}    {tg_str}  ", end="")
        print()


def save_csv(results, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quant_qkv", "prompt_len", "metric", "tok_s", "std"])
        w.writeheader()
        w.writerows(results)
    print(f"\nCSV saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default=LOG_PATH)
    parser.add_argument("--csv", type=str, default=OUTPUT_CSV)
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"Log not found: {args.log}")
        return

    results = parse_log(args.log)
    if not results:
        print("No results parsed. Check log format.")
        return

    print(f"Parsed {len(results)} measurements")
    table = make_summary_table(results)
    print_comparison(table)
    save_csv(results, args.csv)


if __name__ == "__main__":
    main()
