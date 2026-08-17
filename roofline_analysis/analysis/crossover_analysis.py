"""
3-Way Backend Crossover Analysis
시퀀스 길이에 따라 어느 백엔드가 더 빠른지 교차점 탐색.

Usage:
    python analysis/crossover_analysis.py
"""
import os
import csv
import json
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def find_crossover(x1, y1, x2, y2):
    """
    Find crossover point between two performance curves.
    Uses linear interpolation on shared x range.
    Returns (crossover_x, crossover_y) or None.
    """
    if len(x1) < 2 or len(x2) < 2:
        return None

    # Common x range
    x_min = max(min(x1), min(x2))
    x_max = min(max(x1), max(x2))
    if x_min >= x_max:
        return None

    # Interpolate to common grid
    x_common = np.linspace(x_min, x_max, 200)
    y1_interp = np.interp(x_common, x1, y1)
    y2_interp = np.interp(x_common, x2, y2)

    # Find sign changes in (y1 - y2)
    diff = y1_interp - y2_interp
    crossings = []
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] < 0:
            # Linear interpolation for precise crossing
            frac = abs(diff[i]) / (abs(diff[i]) + abs(diff[i + 1]))
            cx = x_common[i] + frac * (x_common[i + 1] - x_common[i])
            cy = y1_interp[i] + frac * (y1_interp[i + 1] - y1_interp[i])
            crossings.append((round(float(cx), 1), round(float(cy), 2)))

    return crossings if crossings else None


def analyze_crossovers(data, phase="prefill"):
    """Analyze 3-way crossovers for a given phase."""
    # Group by (backend, config_id)
    groups = {}
    for row in data:
        key = (row.get("backend", ""), row.get("config_id", ""))
        if key not in groups:
            groups[key] = {"x": [], "y": []}

        if phase == "prefill":
            x = float(row.get("seq_len", 0))
            y = float(row.get("prefill_tok_s", 0))
        else:
            x = float(row.get("context_len", row.get("seq_len", 0)))
            y = float(row.get("decode_tok_s", 0))

        if y > 0:
            groups[key]["x"].append(x)
            groups[key]["y"].append(y)

    # Sort each group by x
    for key in groups:
        pairs = sorted(zip(groups[key]["x"], groups[key]["y"]))
        if pairs:
            groups[key]["x"] = [p[0] for p in pairs]
            groups[key]["y"] = [p[1] for p in pairs]

    # Find crossovers between all pairs
    crossovers = []
    keys = list(groups.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            g1, g2 = groups[k1], groups[k2]

            if not g1["x"] or not g2["x"]:
                continue

            result = find_crossover(
                np.array(g1["x"]), np.array(g1["y"]),
                np.array(g2["x"]), np.array(g2["y"])
            )

            if result:
                for cx, cy in result:
                    crossovers.append({
                        "phase": phase,
                        "backend_1": f"{k1[0]}/{k1[1]}",
                        "backend_2": f"{k2[0]}/{k2[1]}",
                        "crossover_seq_len": cx,
                        "crossover_throughput": cy,
                    })

    return crossovers


def main():
    parser = argparse.ArgumentParser(description="3-Way Crossover Analysis")
    parser.add_argument("--raw_dir", type=str,
                        default=os.path.join(PROJECT_DIR, "results", "raw"))
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.raw_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    prefill_data = load_csv(os.path.join(args.raw_dir, "prefill_sweep.csv"))
    decode_data = load_csv(os.path.join(args.raw_dir, "decode_sweep.csv"))

    all_crossovers = []

    print(f"{'='*70}")
    print("3-Way Backend Crossover Analysis")
    print(f"{'='*70}")

    # Prefill crossovers
    if prefill_data:
        print(f"\nPrefill Crossovers ({len(prefill_data)} data points):")
        prefill_cross = analyze_crossovers(prefill_data, "prefill")
        all_crossovers.extend(prefill_cross)
        for c in prefill_cross:
            print(f"  {c['backend_1']} vs {c['backend_2']}: "
                  f"seq_len={c['crossover_seq_len']:.0f}, "
                  f"throughput={c['crossover_throughput']:.1f} tok/s")

    # Decode crossovers
    if decode_data:
        print(f"\nDecode Crossovers ({len(decode_data)} data points):")
        decode_cross = analyze_crossovers(decode_data, "decode")
        all_crossovers.extend(decode_cross)
        for c in decode_cross:
            print(f"  {c['backend_1']} vs {c['backend_2']}: "
                  f"context_len={c['crossover_seq_len']:.0f}, "
                  f"throughput={c['crossover_throughput']:.1f} tok/s")

    if not all_crossovers:
        print("\n  No crossover points found (need profiling data first)")

    # Save
    csv_path = os.path.join(output_dir, "crossover_points.csv")
    if all_crossovers:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_crossovers[0].keys())
            writer.writeheader()
            writer.writerows(all_crossovers)

    json_path = os.path.join(output_dir, "crossover_points.json")
    with open(json_path, "w") as f:
        json.dump(all_crossovers, f, indent=2, default=str)
    print(f"\nSaved to {json_path}")


if __name__ == "__main__":
    main()
