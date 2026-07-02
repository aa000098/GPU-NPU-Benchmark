"""
Analyze probe v2 results: verify efficiency + Path A/B/C decision.

Reads: results/raw/npu_verify_v2_ctx*.json produced by npu_verify_probe_v2.py
Produces:
  - Table of T_verify(k, c) medians (Mode C)
  - verify_efficiency(k, c) = k * T_verify(k=1, c) / T_verify(k, c)
  - Path A/B/C verdict
  - PDF plots:
      * fig_verify_latency_vs_k.pdf: T_verify curves per ctx
      * fig_verify_efficiency.pdf: efficiency heatmap/lines
      * fig_mode_comparison.pdf: A vs B vs C bar chart (small selected cells)
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "results", "figures_verify_v2")


def load(path):
    with open(path) as f:
        return json.load(f)


def pivot(rows):
    """Build dict[ctx][k] = {'A': ms, 'B': ms, 'C': ms, 'C_shape': (k, vocab)}."""
    table = {}
    for r in rows:
        c, k = int(r["ctx"]), int(r["k"])
        table.setdefault(c, {})[k] = {
            "A": r["A_fresh_k_only"]["median_ms"],
            "B": r["B_ctx_plus_k"]["median_ms"],
            "C": r["C_staged_verify"]["median_ms"],
            "C_shape": r["C_staged_verify"]["shape"],
            "prefill_shape": r["C_staged_verify"]["prefill_shape"],
        }
    return table


def compute_efficiency(table):
    """efficiency = k * T(k=1, c) / T(k, c)"""
    eff = {}
    for c, kmap in table.items():
        if 1 not in kmap:
            continue
        t1 = kmap[1]["C"]
        eff[c] = {k: (k * t1 / v["C"]) for k, v in kmap.items() if v.get("C")}
    return eff


def fit_verify_model(table):
    """Fit T_verify(k, c) = a(c) + b(c) * k for each ctx."""
    fits = {}
    for c, kmap in table.items():
        ks = sorted(kmap.keys())
        ts = [kmap[k]["C"] for k in ks]
        if len(ks) < 2:
            continue
        slope, inter = np.polyfit(ks, ts, 1)
        fits[c] = {"slope_ms_per_k": float(slope), "fixed_ms": float(inter),
                    "ks": ks, "ts": ts}
    return fits


def decide_path(table, eff):
    """Decide Path A/B/C based on largest ctx, k=1 realism + efficiency."""
    if not table:
        return "UNKNOWN", "no data"
    max_ctx = max(table.keys())
    kmap = table[max_ctx]
    c1 = kmap.get(1, {}).get("C")
    a1 = kmap.get(1, {}).get("A")
    if c1 is None:
        return "UNKNOWN", "no k=1 data"
    # Path B signal: C ≈ A (keep_history ignored)
    if a1 and abs(c1 - a1) / max(a1, 1e-6) < 0.2:
        return "PATH_B", (f"At ctx={max_ctx}, Mode C (staged) ≈ Mode A (fresh); "
                          f"keep_history appears to be ignored. Pivot to negative result.")
    # Path A/C criterion: verify efficiency at k=8
    if max_ctx in eff and 8 in eff[max_ctx]:
        e8 = eff[max_ctx][8]
        if e8 >= 1.5:
            return "PATH_A", f"At ctx={max_ctx}, verify_eff(k=8)={e8:.2f}× > 1.5×; proceed with hybrid plan."
        else:
            return "PATH_B_SOFT", (f"At ctx={max_ctx}, verify_eff(k=8)={e8:.2f}× < 1.5×; "
                                    f"batching gives modest gain. Continue as 'modest speedup' narrative.")
    return "UNCLEAR", "insufficient k=8 data"


def print_tables(table, eff, fits):
    print("\n=== T_verify (Mode C, ms) ===")
    ks_all = sorted({k for c in table.values() for k in c.keys()})
    print(f"{'ctx':>6} | " + " ".join(f"{'k='+str(k):>10}" for k in ks_all))
    print("-" * (10 + 10 * len(ks_all)))
    for c in sorted(table.keys()):
        cells = []
        for k in ks_all:
            v = table[c].get(k, {}).get("C")
            cells.append(f"{v:>10.2f}" if v else " " * 10)
        print(f"{c:>6} | " + " ".join(cells))

    print("\n=== verify_efficiency = k × T(k=1) / T(k) ===")
    print("(>1 means batching amortizes work; max = k if perfectly parallel)")
    print(f"{'ctx':>6} | " + " ".join(f"{'k='+str(k):>8}" for k in ks_all))
    print("-" * (10 + 8 * len(ks_all)))
    for c in sorted(eff.keys()):
        cells = []
        for k in ks_all:
            v = eff[c].get(k)
            cells.append(f"{v:>8.2f}" if v else " " * 8)
        print(f"{c:>6} | " + " ".join(cells))

    print("\n=== Linear fit T_verify(k, c) = a + b × k (Mode C) ===")
    print(f"{'ctx':>6} {'fixed(ms)':>12} {'slope(ms/k)':>14}")
    for c in sorted(fits.keys()):
        f = fits[c]
        print(f"{c:>6} {f['fixed_ms']:>12.2f} {f['slope_ms_per_k']:>14.3f}")


def plot_latency_vs_k(table, fits, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(table)))
    for (c, kmap), color in zip(sorted(table.items()), colors):
        ks = sorted(kmap.keys())
        ts = [kmap[k]["C"] for k in ks]
        ax.plot(ks, ts, "o-", label=f"ctx={c}", color=color, markersize=8, linewidth=2)
        if c in fits:
            f = fits[c]
            x = np.array([min(ks), max(ks)])
            y = f["fixed_ms"] + f["slope_ms_per_k"] * x
            ax.plot(x, y, "--", color=color, alpha=0.4)
    ax.set_xlabel("Draft length k")
    ax.set_ylabel("T_verify (ms, Mode C staged verify)")
    ax.set_title("NPU verify latency vs draft length, per context\n(dashed = linear fit)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(out_dir, "fig_verify_latency_vs_k.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_efficiency(eff, out_dir):
    ctxs = sorted(eff.keys())
    ks = sorted({k for c in eff.values() for k in c.keys()})
    # Heatmap
    grid = np.full((len(ctxs), len(ks)), np.nan)
    for i, c in enumerate(ctxs):
        for j, k in enumerate(ks):
            if k in eff[c]:
                grid[i, j] = eff[c][k]
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=1, vmax=max(ks))
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_yticks(range(len(ctxs)))
    ax.set_yticklabels([f"ctx={c}" for c in ctxs])
    for i in range(len(ctxs)):
        for j in range(len(ks)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i,j]:.2f}×", ha="center", va="center",
                         color="black" if grid[i, j] < 4 else "white", fontsize=10)
    plt.colorbar(im, ax=ax, label="verify_efficiency (× NPU-baseline)")
    ax.set_title("Verify batching efficiency\n"
                  "(k=1 is baseline; higher is better; equal to k means perfect parallelism)")
    path = os.path.join(out_dir, "fig_verify_efficiency.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_mode_comparison(table, out_dir):
    """Side-by-side bar chart: A vs B vs C for selected cells."""
    # Pick 2 ctx × 3 k = 6 cells
    ctxs_pick = [c for c in sorted(table) if c in (256, 4079) or c >= 2000][:2]
    ks_pick = [1, 4, 16]
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = []
    A_vals, B_vals, C_vals = [], [], []
    for c in ctxs_pick:
        for k in ks_pick:
            if k in table[c]:
                labels.append(f"ctx={c}\nk={k}")
                A_vals.append(table[c][k]["A"])
                B_vals.append(table[c][k]["B"])
                C_vals.append(table[c][k]["C"])
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, A_vals, w, label="A: fresh k-only (v1 bug)", color="#95A5A6")
    ax.bar(x,     B_vals, w, label="B: combined ctx+k (no KV reuse)", color="#F39C12")
    ax.bar(x + w, C_vals, w, label="C: staged verify (real)", color="#E74C3C")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_title("Three measurement modes of 'verify' — Mode C is the real one")
    ax.legend()
    ax.grid(True, axis="y", which="both", alpha=0.3)
    path = os.path.join(out_dir, "fig_mode_comparison.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None,
                    help="Path to probe v2 JSON. Defaults to newest npu_verify_v2*.json.")
    p.add_argument("--output_dir", default=OUT_DIR)
    args = p.parse_args()

    if args.input is None:
        raw_dir = os.path.join(BASE, "results", "raw")
        candidates = [f for f in os.listdir(raw_dir) if f.startswith("npu_verify_v2_") and f.endswith(".json")]
        if not candidates:
            print("No probe v2 JSON found.", file=sys.stderr)
            sys.exit(1)
        args.input = os.path.join(raw_dir, sorted(candidates)[-1])

    print(f"Loading: {args.input}")
    rows = load(args.input)
    table = pivot(rows)
    eff = compute_efficiency(table)
    fits = fit_verify_model(table)

    print_tables(table, eff, fits)

    path_decision, reason = decide_path(table, eff)
    print(f"\n=== PATH DECISION: {path_decision} ===")
    print(f"  {reason}")

    os.makedirs(args.output_dir, exist_ok=True)
    plot_latency_vs_k(table, fits, args.output_dir)
    plot_efficiency(eff, args.output_dir)
    plot_mode_comparison(table, args.output_dir)

    # Save analysis summary
    summary = {
        "input_json": args.input,
        "path_decision": path_decision,
        "reason": reason,
        "linear_fits": fits,
        "verify_efficiency": eff,
    }
    summary_path = os.path.join(args.output_dir, "probe_v2_analysis.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
