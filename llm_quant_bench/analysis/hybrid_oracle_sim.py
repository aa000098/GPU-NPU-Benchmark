"""
Oracle simulator for CPU-draft / NPU-verify hybrid speculative decoding.

Purpose:
- Predict when hybrid speculative decoding beats CPU-only or NPU-only decode
- Sweep context length, verify length k, and acceptance rate
- Plug in measured NPU verify latencies once npu_verify_probe.py results are available

Usage:
    python analysis/hybrid_oracle_sim.py

    python analysis/hybrid_oracle_sim.py \
        --verify_json ./results/raw/npu_verify_probe.json
"""
import argparse
import csv
import json
import math
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(BASE, "results", "figures_hybrid")


# From current characterization
CONTEXTS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
CPU_DECODE_MS = {
    32: 64.6,
    64: 60.4,
    128: 63.5,
    256: 59.9,
    512: 60.2,
    1024: 61.1,
    2048: 60.8,
    4096: 63.4,
}
NPU_DECODE_MS = {
    32: 50.0,
    64: 50.0,
    128: 51.0,
    256: 54.9,
    512: 55.2,
    1024: 58.8,
    2048: 68.5,
    4096: 88.5,
}

# Conservative draft-side placeholder.
# This is intentionally simple and can be replaced with measured draft numbers later.
DRAFT_MS_PER_TOKEN = {
    32: 3.0,
    64: 3.0,
    128: 3.1,
    256: 3.2,
    512: 3.3,
    1024: 3.5,
    2048: 3.7,
    4096: 4.0,
}

# Placeholder verify table until probe results are supplied.
# The pattern encodes the hoped-for property:
# batched verify grows sublinearly relative to k * NPU decode.
DEFAULT_VERIFY_MS = {
    32:   {1: 50.0, 2: 54.0, 4: 60.0, 8: 73.0, 16: 98.0},
    64:   {1: 50.0, 2: 54.0, 4: 60.0, 8: 73.0, 16: 98.0},
    128:  {1: 51.0, 2: 55.0, 4: 61.0, 8: 74.0, 16: 100.0},
    256:  {1: 54.9, 2: 59.0, 4: 66.0, 8: 81.0, 16: 110.0},
    512:  {1: 55.2, 2: 60.0, 4: 68.0, 8: 85.0, 16: 116.0},
    1024: {1: 58.8, 2: 64.0, 4: 74.0, 8: 93.0, 16: 128.0},
    2048: {1: 68.5, 2: 74.0, 4: 86.0, 8: 110.0, 16: 153.0},
    4096: {1: 88.5, 2: 95.0, 4: 112.0, 8: 145.0, 16: 204.0},
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_verify_json(path):
    """
    Expected schema:
    {
      "32": {"1": 50.0, "2": 54.0, "4": 60.0},
      "256": {"1": 54.9, "2": 59.0, "4": 66.0}
    }
    """
    with open(path) as f:
        data = json.load(f)

    verify = {}
    for ctx_key, inner in data.items():
        ctx = int(ctx_key)
        verify[ctx] = {int(k): float(v) for k, v in inner.items()}
    return verify


def acceptance_model(context, k, mode):
    """
    Simple oracle acceptance priors for paper planning.

    mode:
    - optimistic
    - moderate
    - pessimistic
    """
    if mode == "optimistic":
        base = 0.92
        ctx_penalty = 0.02 * math.log2(max(context, 32) / 32)
        k_penalty = 0.012 * max(k - 1, 0)
    elif mode == "pessimistic":
        base = 0.78
        ctx_penalty = 0.035 * math.log2(max(context, 32) / 32)
        k_penalty = 0.020 * max(k - 1, 0)
    else:
        base = 0.86
        ctx_penalty = 0.028 * math.log2(max(context, 32) / 32)
        k_penalty = 0.016 * max(k - 1, 0)

    per_token_accept = max(0.35, min(0.98, base - ctx_penalty - k_penalty))
    expected_accepted = sum(per_token_accept ** i for i in range(1, k + 1))
    return per_token_accept, expected_accepted


def correction_overhead_ms(context):
    """
    Expected extra target work after a rejection.
    We approximate this as a fraction of one NPU decode step.
    """
    return 0.25 * NPU_DECODE_MS[context]


def simulate_context(context, verify_table, k_values, acceptance_mode):
    rows = []
    cpu_ms = CPU_DECODE_MS[context]
    npu_ms = NPU_DECODE_MS[context]
    draft_ms_per_token = DRAFT_MS_PER_TOKEN[context]

    for k in k_values:
        if k not in verify_table[context]:
            continue

        verify_ms = verify_table[context][k]
        per_token_accept, expected_accepted = acceptance_model(context, k, acceptance_mode)
        expected_accepted = max(expected_accepted, 1e-6)

        draft_total_ms = draft_ms_per_token * k
        corr_ms = correction_overhead_ms(context) * (1.0 - per_token_accept)
        hybrid_ms_per_token = (draft_total_ms + verify_ms + corr_ms) / expected_accepted

        verify_efficiency = (k * npu_ms) / verify_ms
        speedup_vs_cpu = cpu_ms / hybrid_ms_per_token
        speedup_vs_npu = npu_ms / hybrid_ms_per_token

        rows.append({
            "context": context,
            "k": k,
            "cpu_ms_per_token": cpu_ms,
            "npu_ms_per_token": npu_ms,
            "draft_total_ms": draft_total_ms,
            "verify_ms": verify_ms,
            "corr_ms": corr_ms,
            "per_token_accept": per_token_accept,
            "expected_accepted": expected_accepted,
            "hybrid_ms_per_token": hybrid_ms_per_token,
            "verify_efficiency": verify_efficiency,
            "speedup_vs_cpu": speedup_vs_cpu,
            "speedup_vs_npu": speedup_vs_npu,
        })

    return rows


def pick_best(rows):
    if not rows:
        return None
    return min(rows, key=lambda r: r["hybrid_ms_per_token"])


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_best_speedup(best_rows, output_dir, acceptance_mode):
    if not HAS_MATPLOTLIB:
        return
    contexts = [r["context"] for r in best_rows]
    cpu_speedup = [r["speedup_vs_cpu"] for r in best_rows]
    npu_speedup = [r["speedup_vs_npu"] for r in best_rows]
    best_k = [r["k"] for r in best_rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(contexts, cpu_speedup, 'o-', linewidth=2, markersize=8,
            label="Hybrid vs CPU-only", color="#1f77b4")
    ax.plot(contexts, npu_speedup, 's-', linewidth=2, markersize=8,
            label="Hybrid vs NPU-only", color="#d62728")
    ax.axhline(1.0, linestyle="--", color="gray", alpha=0.8)

    for ctx, y, k in zip(contexts, npu_speedup, best_k):
        ax.annotate(f"k={k}", (ctx, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9)

    ax.set_xscale("log")
    ax.set_xticks(contexts)
    ax.set_xticklabels([str(c) for c in contexts])
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Best Hybrid Speedup by Context ({acceptance_mode})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, f"oracle_best_speedup_{acceptance_mode}.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)


def plot_optimal_k(best_rows, output_dir, acceptance_mode):
    if not HAS_MATPLOTLIB:
        return
    contexts = [r["context"] for r in best_rows]
    best_k = [r["k"] for r in best_rows]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(contexts, best_k, 'o-', linewidth=2, markersize=8, color="#2ca02c")
    ax.set_xscale("log")
    ax.set_xticks(contexts)
    ax.set_xticklabels([str(c) for c in contexts])
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Best Draft Length k")
    ax.set_title(f"Context-Aware Optimal Draft Length ({acceptance_mode})")
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, f"oracle_optimal_k_{acceptance_mode}.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(rows, output_dir, acceptance_mode):
    if not HAS_MATPLOTLIB:
        return
    contexts = sorted({r["context"] for r in rows})
    ks = sorted({r["k"] for r in rows})
    speedup = np.full((len(ks), len(contexts)), np.nan)

    for r in rows:
        ki = ks.index(r["k"])
        ci = contexts.index(r["context"])
        speedup[ki, ci] = r["speedup_vs_npu"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    im = ax.imshow(speedup, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels([str(c) for c in contexts])
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([str(k) for k in ks])
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Draft Length k")
    ax.set_title(f"Hybrid Speedup vs NPU-only Heatmap ({acceptance_mode})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Speedup vs NPU-only")

    path = os.path.join(output_dir, f"oracle_heatmap_vs_npu_{acceptance_mode}.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)


def summarize(best_rows, acceptance_mode):
    print(f"\n=== Best Hybrid Policy Summary ({acceptance_mode}) ===")
    print(" ctx | best_k | hybrid_ms/tok | vs_cpu | vs_npu | verify_eff | exp_acc")
    for r in best_rows:
        print(
            f"{r['context']:>4} | "
            f"{r['k']:>6} | "
            f"{r['hybrid_ms_per_token']:>13.2f} | "
            f"{r['speedup_vs_cpu']:>6.2f} | "
            f"{r['speedup_vs_npu']:>6.2f} | "
            f"{r['verify_efficiency']:>10.2f} | "
            f"{r['expected_accepted']:>7.2f}"
        )


def extract_dispatch_policy(best_rows):
    policy_rows = []
    for r in best_rows:
        cpu_ms = r["cpu_ms_per_token"]
        npu_ms = r["npu_ms_per_token"]
        hybrid_ms = r["hybrid_ms_per_token"]

        candidates = [
            ("CPU-only", cpu_ms, None),
            ("NPU-only", npu_ms, None),
            ("Hybrid", hybrid_ms, r["k"]),
        ]
        best_mode, best_ms, best_k = min(candidates, key=lambda x: x[1])

        policy_rows.append({
            "context": r["context"],
            "best_mode": best_mode,
            "best_k": best_k,
            "cpu_ms_per_token": cpu_ms,
            "npu_ms_per_token": npu_ms,
            "hybrid_ms_per_token": hybrid_ms,
            "hybrid_speedup_vs_cpu": r["speedup_vs_cpu"],
            "hybrid_speedup_vs_npu": r["speedup_vs_npu"],
        })
    return policy_rows


def print_dispatch_policy(policy_rows, acceptance_mode):
    print(f"\n=== Dynamic Dispatch Policy ({acceptance_mode}) ===")
    print(" ctx | best_mode | best_k | cpu_ms | npu_ms | hybrid_ms")
    for r in policy_rows:
        best_k = "-" if r["best_k"] is None else str(r["best_k"])
        print(
            f"{r['context']:>4} | "
            f"{r['best_mode']:<9} | "
            f"{best_k:>6} | "
            f"{r['cpu_ms_per_token']:>6.1f} | "
            f"{r['npu_ms_per_token']:>6.1f} | "
            f"{r['hybrid_ms_per_token']:>9.1f}"
        )


def write_policy_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Oracle simulator for hybrid speculative decoding")
    parser.add_argument("--verify_json", type=str, default=None,
                        help="Optional JSON file with measured verify latencies")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--acceptance_mode", choices=["optimistic", "moderate", "pessimistic", "all"],
                        default="all")
    parser.add_argument("--k_values", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    verify_table = load_verify_json(args.verify_json) if args.verify_json else DEFAULT_VERIFY_MS

    modes = [args.acceptance_mode] if args.acceptance_mode != "all" else [
        "optimistic", "moderate", "pessimistic"
    ]

    for mode in modes:
        rows = []
        best_rows = []

        for context in CONTEXTS:
            sim_rows = simulate_context(
                context=context,
                verify_table=verify_table,
                k_values=args.k_values,
                acceptance_mode=mode,
            )
            rows.extend(sim_rows)
            best = pick_best(sim_rows)
            if best:
                best_rows.append(best)

        csv_path = os.path.join(args.output_dir, f"oracle_sim_{mode}.csv")
        write_csv(csv_path, rows)
        summarize(best_rows, mode)
        policy_rows = extract_dispatch_policy(best_rows)
        print_dispatch_policy(policy_rows, mode)
        plot_best_speedup(best_rows, args.output_dir, mode)
        plot_optimal_k(best_rows, args.output_dir, mode)
        plot_heatmap(rows, args.output_dir, mode)

        best_json = os.path.join(args.output_dir, f"oracle_best_policy_{mode}.json")
        dispatch_csv = os.path.join(args.output_dir, f"oracle_dispatch_policy_{mode}.csv")
        with open(best_json, "w") as f:
            json.dump(best_rows, f, indent=2)
        write_policy_csv(dispatch_csv, policy_rows)
        print(f"Saved: {csv_path}")
        print(f"Saved: {best_json}")
        print(f"Saved: {dispatch_csv}")

    if not HAS_MATPLOTLIB:
        print("\n[WARN] matplotlib not installed; skipped figure generation and wrote CSV/JSON only.")


if __name__ == "__main__":
    main()
