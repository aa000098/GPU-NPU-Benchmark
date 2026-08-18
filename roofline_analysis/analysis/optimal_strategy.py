"""
Optimal Backend Strategy Map
(시퀀스 길이 × 양자화 설정 × phase)에서 최적 백엔드 선택.

Usage:
    python analysis/optimal_strategy.py
"""
import os
import csv
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def build_optimal_map(prefill_data, decode_data):
    """Build optimal backend selection map."""
    results = []

    # Group prefill by (seq_len, config_id)
    prefill_grid = {}
    for row in prefill_data:
        key = (int(row["seq_len"]), row["config_id"])
        backend = row["backend"]
        tps = float(row.get("prefill_tok_s", 0))
        if key not in prefill_grid:
            prefill_grid[key] = []
        prefill_grid[key].append({"backend": backend, "throughput": tps})

    # Group decode by (context_len, config_id)
    decode_grid = {}
    for row in decode_data:
        key = (int(row.get("context_len", row.get("seq_len", 0))), row["config_id"])
        backend = row["backend"]
        tps = float(row.get("decode_tok_s", 0))
        if key not in decode_grid:
            decode_grid[key] = []
        decode_grid[key].append({"backend": backend, "throughput": tps})

    # Find optimal for each cell
    for phase, grid in [("prefill", prefill_grid), ("decode", decode_grid)]:
        for (seq_len, config_id), entries in sorted(grid.items()):
            if not entries:
                continue

            entries_sorted = sorted(entries, key=lambda e: e["throughput"], reverse=True)
            best = entries_sorted[0]
            second = entries_sorted[1] if len(entries_sorted) > 1 else None

            gap_pct = 0
            if second and second["throughput"] > 0:
                gap_pct = (best["throughput"] - second["throughput"]) / second["throughput"] * 100

            results.append({
                "phase": phase,
                "seq_len": seq_len,
                "config_id": config_id,
                "best_backend": best["backend"],
                "best_throughput": round(best["throughput"], 2),
                "second_backend": second["backend"] if second else "",
                "second_throughput": round(second["throughput"], 2) if second else 0,
                "gap_pct": round(gap_pct, 1),
                "is_tossup": "yes" if gap_pct < 10 else "no",
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Optimal Strategy Map")
    parser.add_argument("--raw_dir", type=str,
                        default=os.path.join(PROJECT_DIR, "results", "raw"))
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.raw_dir
    os.makedirs(output_dir, exist_ok=True)

    prefill_data = load_csv(os.path.join(args.raw_dir, "prefill_sweep.csv"))
    decode_data = load_csv(os.path.join(args.raw_dir, "decode_sweep.csv"))

    print(f"{'='*70}")
    print("Optimal Backend Strategy Map")
    print(f"{'='*70}")

    results = build_optimal_map(prefill_data, decode_data)

    if results:
        # Print summary
        for phase in ["prefill", "decode"]:
            phase_results = [r for r in results if r["phase"] == phase]
            if not phase_results:
                continue

            print(f"\n  {phase.upper()} Phase:")
            print(f"  {'SeqLen':>7s} {'Config':>8s} {'Best':>12s} {'Throughput':>10s} "
                  f"{'2nd':>12s} {'Gap%':>6s} {'Tossup':>6s}")

            for r in phase_results:
                print(f"  {r['seq_len']:>7d} {r['config_id']:>8s} "
                      f"{r['best_backend']:>12s} {r['best_throughput']:>10.1f} "
                      f"{r['second_backend']:>12s} {r['gap_pct']:>6.1f} "
                      f"{r['is_tossup']:>6s}")
    else:
        print("\n  No data available (run profiling first)")

    # Save
    csv_path = os.path.join(output_dir, "optimal_backend_map.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    json_path = os.path.join(output_dir, "optimal_backend_map.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {json_path}")


if __name__ == "__main__":
    main()
