"""
Analyze acceptance traces and produce:
- Figures: acceptance vs position, heatmap, hit-rate curve
- Distilled data for Oracle simulator: a(c, k) — expected acceptance rate at context c, draft length k

Inputs:
- results/raw/acceptance_llama_draft.json (Llama Q4_0 → Llama W8A8)
- results/raw/acceptance_qwen_draft.json  (Qwen 0.5B → Llama W8A8)

Outputs:
- results/figures_acceptance/fig_acceptance_vs_position.pdf
- results/figures_acceptance/fig_acceptance_heatmap.pdf
- results/figures_acceptance/fig_alpha_vs_hit.pdf
- results/raw/acceptance_distilled_for_oracle.json
"""

import argparse
import json
import os
import statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "results", "raw")
OUT_DIR = os.path.join(BASE, "results", "figures_acceptance")


def load_trace(path):
    with open(path) as f:
        return json.load(f)


def per_position_aggregate(trace):
    """Group by position, return {position: {alpha_mean, alpha_std, hit_rate}}"""
    by_pos = {}
    for prompt in trace["per_prompt"]:
        for r in prompt["per_position"]:
            pos = r["position"]
            by_pos.setdefault(pos, {"alpha": [], "hit": []})
            by_pos[pos]["alpha"].append(r["alpha_spec"])
            by_pos[pos]["hit"].append(r["hit_greedy"])

    result = {}
    for pos, v in by_pos.items():
        result[pos] = {
            "alpha_mean": statistics.mean(v["alpha"]),
            "alpha_std": statistics.stdev(v["alpha"]) if len(v["alpha"]) > 1 else 0.0,
            "hit_rate": statistics.mean(v["hit"]),
            "n": len(v["alpha"]),
        }
    return result


def distill_for_oracle(llama_trace, qwen_trace, k_max=16):
    """
    Produce a table a(c, k) that the oracle sim can consume.
    a(c, k) = E[accepted tokens | verify round with context c, draft length k]

    For independent per-position accept with rate α:
      E[accepted | k] = α(1-α^k)/(1-α) if α < 1 else k

    But accept is stopped at first reject, so actually:
      E[accepted | k] = sum_{i=0}^{k-1} α^(i+1) * (if i<k-1) + α^k * (hit through)
      Simplified: E[accepted] = (1 - α^k) / (1 - α) * α  [geometric]

    We collapse per-position α into one α per context bucket (mean over trace positions in that bucket).
    """
    buckets = [64, 256, 1024, 4096]
    ks = list(range(1, k_max + 1))

    def expected_accepted(alpha, k):
        if alpha >= 1.0:
            return float(k)
        return (1 - alpha ** k) / (1 - alpha) - 1  # subtract 1 because last is unaccepted (correction)

    out = {}
    for name, trace in [("llama_draft", llama_trace), ("qwen_draft", qwen_trace)]:
        by_bucket = {str(b): {} for b in buckets}
        agg = trace.get("aggregate_by_context_bucket", {})
        for b, s in agg.items():
            alpha = s["alpha_mean"]
            for k in ks:
                by_bucket[b][str(k)] = {
                    "alpha_per_pos": alpha,
                    "expected_accepted": expected_accepted(alpha, k),
                }
        out[name] = by_bucket
    return out


def fig_acceptance_vs_position(traces, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"llama_draft": "#E74C3C", "qwen_draft": "#3498DB"}
    labels = {"llama_draft": "Llama Q4_0 → Llama W8A8", "qwen_draft": "Qwen 0.5B → Llama W8A8"}

    for name, trace in traces.items():
        by_pos = per_position_aggregate(trace)
        positions = sorted(by_pos.keys())
        alpha = [by_pos[p]["alpha_mean"] for p in positions]
        std = [by_pos[p]["alpha_std"] for p in positions]
        ax.errorbar(positions, alpha, yerr=std, fmt="o-", color=colors[name],
                     label=labels[name], alpha=0.75, markersize=5, linewidth=1.5)

    ax.set_xlabel("Generated token position")
    ax.set_ylabel("Acceptance rate α = Σ min(p, q)")
    ax.set_title("Per-position speculative acceptance rate\n"
                  "(higher = draft more aligned with target)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = os.path.join(output_dir, "fig_acceptance_vs_position.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_alpha_distribution(traces, output_dir):
    """Compare distribution of α values — histogram per draft."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"llama_draft": "#E74C3C", "qwen_draft": "#3498DB"}
    labels = {"llama_draft": "Llama Q4_0 (same tokenizer)",
              "qwen_draft": "Qwen 0.5B (different tokenizer)"}

    for name, trace in traces.items():
        alphas = []
        for prompt in trace["per_prompt"]:
            for r in prompt["per_position"]:
                alphas.append(r["alpha_spec"])
        ax.hist(alphas, bins=30, color=colors[name], label=labels[name],
                 alpha=0.5, density=True)
        mean = statistics.mean(alphas)
        ax.axvline(mean, color=colors[name], linestyle="--",
                    linewidth=2, label=f"{labels[name]} mean={mean:.3f}")

    ax.set_xlabel("Acceptance rate α per position")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of per-position acceptance rates")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "fig_alpha_distribution.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def fig_expected_accepted(distilled, output_dir):
    """Plot E[accepted tokens | k] for each draft & context bucket."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "llama_draft": {"64": "#E74C3C", "256": "#C0392B"},
        "qwen_draft":  {"64": "#3498DB", "256": "#2874A6"},
    }
    labels = {"llama_draft": "Llama Q4_0", "qwen_draft": "Qwen 0.5B"}

    for name, by_bucket in distilled.items():
        for bucket, k_map in by_bucket.items():
            if not k_map:
                continue
            ks = sorted(int(k) for k in k_map.keys())
            ea = [k_map[str(k)]["expected_accepted"] for k in ks]
            alpha = k_map[str(ks[0])]["alpha_per_pos"]
            color = colors.get(name, {}).get(bucket, "#95A5A6")
            ax.plot(ks, ea, "o-", color=color,
                     label=f"{labels[name]} ctx≤{bucket} (α={alpha:.2f})", linewidth=2)

    # Ideal reference lines (α = 1)
    ks_all = list(range(1, 17))
    ax.plot(ks_all, ks_all, "k--", alpha=0.3, label="Ideal (α=1)")

    ax.set_xlabel("Draft length k")
    ax.set_ylabel("E[accepted tokens per round]")
    ax.set_title("Expected accepted tokens per speculative round\n"
                  "(based on measured α; assumes i.i.d. per-position)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "fig_expected_accepted_vs_k.pdf")
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_summary(traces, distilled):
    print("=" * 80)
    print("ACCEPTANCE TRACE SUMMARY")
    print("=" * 80)

    for name, trace in traces.items():
        print(f"\n=== {name} ===")
        print(f"  target: {trace['target_model'].split('/')[-1]}")
        print(f"  draft:  {trace['draft_model'].split('/')[-1]}")
        print(f"  {trace['n_prompts']} prompts × {trace['gen_tokens']} gen tokens")
        for b, s in sorted(trace["aggregate_by_context_bucket"].items(),
                            key=lambda x: int(x[0])):
            print(f"  ctx≤{b}: α_mean={s['alpha_mean']:.3f}, hit={s['hit_rate']:.3f} (n={s['n']})")

    print("\n=== Oracle input: Expected accepted tokens per round ===")
    print(f"{'Draft':<15} {'Ctx':<6} " + " ".join(f"{'k='+str(k):>6}" for k in [1, 2, 4, 8, 16]))
    print("-" * 85)
    for name, by_bucket in distilled.items():
        for b, k_map in sorted(by_bucket.items(), key=lambda x: int(x[0])):
            if not k_map:
                continue
            row = f"{name:<15} {b:<6} "
            for k in [1, 2, 4, 8, 16]:
                if str(k) in k_map:
                    row += f"{k_map[str(k)]['expected_accepted']:>6.2f} "
            print(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=OUT_DIR)
    args = p.parse_args()

    traces = {
        "llama_draft": load_trace(os.path.join(RAW, "acceptance_llama_draft.json")),
        "qwen_draft": load_trace(os.path.join(RAW, "acceptance_qwen_draft.json")),
    }

    distilled = distill_for_oracle(traces["llama_draft"], traces["qwen_draft"])

    print_summary(traces, distilled)

    os.makedirs(args.output_dir, exist_ok=True)
    fig_acceptance_vs_position(traces, args.output_dir)
    fig_alpha_distribution(traces, args.output_dir)
    fig_expected_accepted(distilled, args.output_dir)

    # Save distilled for oracle
    distilled_path = os.path.join(RAW, "acceptance_distilled_for_oracle.json")
    with open(distilled_path, "w") as f:
        json.dump(distilled, f, indent=2)
    print(f"\nSaved oracle input: {distilled_path}")


if __name__ == "__main__":
    main()
