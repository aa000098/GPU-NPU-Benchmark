"""
Oracle simulator with MEASURED data (Day 1-4 inputs).

Inputs (all measured):
- T_verify(k, ctx) from probe v2 Mode C → results/raw/npu_verify_v2_ctx4096.json
- T_draft(k, draft) = k × draft_ms_per_tok from Day 3 draft_bench
- α(draft) from Day 4 acceptance traces

Baselines (measured):
- T_cpu_decode(ctx) from Phase 1 (nearly context-invariant ~60 ms)
- T_npu_decode(ctx) from Phase 1 (context-linear, up to 88.5 ms at ctx=4096)

Output:
- `hybrid_ms_per_token(ctx, k, draft)` = (T_draft + T_verify) / E[accept]
- Speedup vs best single-backend baseline
- Figures: heatmap, profitable region, best (k, draft) per ctx
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "results", "raw")
OUT_DIR = os.path.join(BASE, "results", "figures_hybrid_measured")


# ── Measured baselines (Phase 1) ──────────────────────────────────────
NPU_DECODE_MS = {
    32: 50.0, 64: 50.0, 128: 51.0, 256: 54.9,
    512: 55.2, 1024: 58.8, 2048: 68.5, 4096: 88.5,
}
CPU_DECODE_MS = {
    32: 64.6, 64: 60.4, 128: 63.5, 256: 59.9,
    512: 60.2, 1024: 61.1, 2048: 60.8, 4096: 63.4,
}


# ── Draft models (Day 3) ──────────────────────────────────────────────
DRAFTS = {
    "Llama_Q4_0": {
        "ms_per_tok": 41.6,   # Day 3 llama-bench, CPU 4 threads
        "alpha": 0.88,         # Day 4 acceptance trace (aggregate)
        "tokenizer_compat": True,
    },
    "Qwen_0.5B_Q4_0": {
        "ms_per_tok": 19.5,
        "alpha": 0.11,         # ablation — cross-tokenizer, underestimate
        "tokenizer_compat": False,
    },
}


def load_measured_verify():
    """Load probe v2 Mode C times."""
    path = os.path.join(RAW, "npu_verify_v2_ctx4096.json")
    with open(path) as f:
        rows = json.load(f)
    # Build table: verify[ctx][k] = ms (Mode C median)
    v = {}
    for r in rows:
        ctx = int(r["ctx"])
        k = int(r["k"])
        v.setdefault(ctx, {})[k] = r["C_staged_verify"]["median_ms"]
    return v


def expected_accept(alpha, k):
    """E[accepted tokens | α, k] under geometric acceptance.

    Standard speculative decoding: for each drafted position i=0..k-1,
    accept with probability α, stop at first reject.
    E[accepted] = sum_{i=1}^{k} α^i = α(1-α^k)/(1-α)
    Minus 1 because the k+1-th token (correction) is not drafted.
    """
    if alpha >= 1.0:
        return float(k)
    return alpha * (1 - alpha ** k) / (1 - alpha)


def simulate(verify_table, ctx, k, draft_info, mode="serial"):
    """Compute hybrid ms/token given measured inputs.

    mode:
      - 'serial' : T_draft + T_verify per round (naive)
      - 'async'  : max(T_draft, T_verify) per round (overlapped)
                   Assumption: round N+1 draft runs on CPU while round N
                   verify runs on NPU. First round has a T_verify cold-start.
    """
    if ctx not in verify_table or k not in verify_table[ctx]:
        return None

    T_verify = verify_table[ctx][k]
    T_draft = draft_info["ms_per_tok"] * k
    alpha = draft_info["alpha"]
    E_accept = expected_accept(alpha, k)
    if E_accept <= 0:
        return None

    emitted_per_round = E_accept + 1

    if mode == "serial":
        round_ms = T_draft + T_verify
    elif mode == "async":
        # steady-state amortization: round throughput limited by slower of {draft, verify}
        round_ms = max(T_draft, T_verify)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    hybrid_ms_per_token = round_ms / emitted_per_round

    return {
        "ctx": ctx,
        "k": k,
        "mode": mode,
        "T_draft": T_draft,
        "T_verify": T_verify,
        "alpha": alpha,
        "E_accept": E_accept,
        "emitted_per_round": emitted_per_round,
        "round_ms": round_ms,
        "hybrid_ms_per_token": hybrid_ms_per_token,
        "speedup_vs_cpu": CPU_DECODE_MS.get(ctx, 60) / hybrid_ms_per_token,
        "speedup_vs_npu": NPU_DECODE_MS.get(ctx, 60) / hybrid_ms_per_token,
        "speedup_vs_best_single": min(CPU_DECODE_MS.get(ctx, 60), NPU_DECODE_MS.get(ctx, 60)) / hybrid_ms_per_token,
    }


def build_grid(verify_table, draft_name, mode="serial"):
    """Compute full (ctx, k) grid for one draft."""
    di = DRAFTS[draft_name]
    grid = {}
    for ctx, kmap in verify_table.items():
        grid[ctx] = {}
        for k in kmap.keys():
            r = simulate(verify_table, ctx, k, di, mode=mode)
            if r:
                grid[ctx][k] = r
    return grid


def print_grid(grid, draft_name, metric):
    ctxs = sorted(grid.keys())
    ks = sorted(set(k for c in grid.values() for k in c.keys()))
    print(f"\n=== {draft_name}: {metric} ===")
    print(f"{'ctx':>6} | " + " ".join(f"{'k='+str(k):>8}" for k in ks))
    print("-" * (8 + 9 * len(ks)))
    for ctx in ctxs:
        row = f"{ctx:>6} | "
        for k in ks:
            v = grid[ctx].get(k, {}).get(metric)
            if v is None:
                row += " " * 9
            else:
                row += f"{v:>8.3f} "
        print(row)


def find_best(grid, ctx):
    if ctx not in grid:
        return None
    best = min(grid[ctx].values(), key=lambda r: r["hybrid_ms_per_token"])
    return best


def plot_heatmap_speedup(grid, draft_name, output_dir, metric="speedup_vs_best_single"):
    ctxs = sorted(grid.keys())
    ks = sorted(set(k for c in grid.values() for k in c.keys()))
    mat = np.full((len(ctxs), len(ks)), np.nan)
    for i, ctx in enumerate(ctxs):
        for j, k in enumerate(ks):
            r = grid[ctx].get(k)
            if r is not None:
                mat[i, j] = r[metric]

    fig, ax = plt.subplots(figsize=(10, 6))
    vmax = max(1.2, np.nanmax(mat))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=vmax)
    for i in range(len(ctxs)):
        for j in range(len(ks)):
            if not np.isnan(mat[i, j]):
                col = "white" if mat[i, j] < 0.9 else "black"
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                         color=col, fontsize=9)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_yticks(range(len(ctxs)))
    ax.set_yticklabels([f"ctx={c}" for c in ctxs])
    plt.colorbar(im, ax=ax, label=metric)
    ax.set_title(f"Hybrid speedup vs best single backend — {draft_name}\n"
                  f"(green = hybrid wins; >1 means profitable)")

    # profitability contour at 1.0
    ax.contour(mat, levels=[1.0], colors="black", linewidths=2)

    path = os.path.join(output_dir, f"fig_speedup_heatmap_{draft_name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_hybrid_vs_baselines(grid, draft_name, output_dir):
    """Line plot: tok/s for CPU, NPU, best hybrid at each context."""
    ctxs = sorted(grid.keys())
    cpu = [1000 / CPU_DECODE_MS.get(c, 60) for c in ctxs]
    npu = [1000 / NPU_DECODE_MS.get(c, 60) for c in ctxs]
    best_hyb = []
    best_k = []
    for c in ctxs:
        best = find_best(grid, c)
        if best:
            best_hyb.append(1000 / best["hybrid_ms_per_token"])
            best_k.append(best["k"])
        else:
            best_hyb.append(0)
            best_k.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ctxs, cpu, "s-", color="#3498DB", linewidth=2, markersize=9, label="CPU decode")
    ax.plot(ctxs, npu, "^-", color="#E74C3C", linewidth=2, markersize=9, label="NPU decode")
    ax.plot(ctxs, best_hyb, "o-", color="#27AE60", linewidth=2.5, markersize=10,
             label=f"Hybrid (best k per ctx, {draft_name})")

    for i, (c, bk) in enumerate(zip(ctxs, best_k)):
        if bk:
            ax.annotate(f"k={bk}", (c, best_hyb[i]),
                         xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_xscale("log")
    ax.set_xticks(ctxs)
    ax.set_xticklabels([str(c) for c in ctxs])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Output throughput (tok/s)")
    ax.set_title(f"Hybrid throughput vs baselines — {draft_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, f"fig_hybrid_vs_baselines_{draft_name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=OUT_DIR)
    args = p.parse_args()

    print("Loading measured T_verify from probe v2...")
    verify_table = load_measured_verify()
    print(f"  contexts: {sorted(verify_table.keys())}")

    os.makedirs(args.output_dir, exist_ok=True)

    summary = {}
    for draft_name in DRAFTS:
        di = DRAFTS[draft_name]
        for mode in ["serial", "async"]:
            print(f"\n\n{'='*75}")
            print(f"DRAFT: {draft_name} | MODE: {mode}  (ms/tok={di['ms_per_tok']}, "
                  f"α={di['alpha']}, tok_compat={di['tokenizer_compat']})")
            print('='*75)

            grid = build_grid(verify_table, draft_name, mode=mode)
            print_grid(grid, f"{draft_name}-{mode}", "hybrid_ms_per_token")
            print_grid(grid, f"{draft_name}-{mode}", "speedup_vs_best_single")

            print(f"\n=== Best (k) per ctx for {draft_name} [{mode}] ===")
            print(f"{'ctx':>6} {'best_k':>7} {'hybrid_ms':>12} {'vs_cpu':>8} {'vs_npu':>8} {'vs_best':>8}")
            for ctx in sorted(grid.keys()):
                best = find_best(grid, ctx)
                if best:
                    print(f"{ctx:>6} {best['k']:>7} {best['hybrid_ms_per_token']:>12.2f} "
                          f"{best['speedup_vs_cpu']:>7.2f}× {best['speedup_vs_npu']:>7.2f}× "
                          f"{best['speedup_vs_best_single']:>7.2f}×")

            plot_heatmap_speedup(grid, f"{draft_name}_{mode}", args.output_dir)
            plot_hybrid_vs_baselines(grid, f"{draft_name}_{mode}", args.output_dir)

            summary[f"{draft_name}_{mode}"] = {
                "draft_info": di,
                "mode": mode,
                "grid": {str(c): {str(k): r for k, r in m.items()} for c, m in grid.items()},
            }

    summary_path = os.path.join(args.output_dir, "hybrid_oracle_measured.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
