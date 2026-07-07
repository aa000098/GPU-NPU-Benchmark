"""Generate fig_A_regimespec from v3 RegimeSpec measurements.

Loads regimespec_{mixed,long_only,short_only}_v3.json, draws a grouped
bar chart of speedup vs NPU-only across the three workloads × five modes.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "llm_quant_bench" / "results" / "raw"
OUT_DIR = ROOT / "llm_quant_bench" / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORKLOADS = [
    ("mixed", "mixed\n(5 nat + 5 code + 5 wiki long)"),
    ("long_only", "long_only\n(wiki natural, ctx 1024–3000)"),
    ("short_only", "short_only\n(5 nat + 5 code)"),
]
MODES = [
    ("npu_only",     "NPU-only",            "#7f7f7f"),
    ("always_seq",   "Always-Seq",          "#9467bd"),
    ("always_async", "Always-Async",        "#2ca02c"),
    ("always_pld",   "Always-PLD",          "#1f77b4"),
    ("regimespec",   "RegimeSpec(PLD+Async)", "#d62728"),
    ("oracle",       "Oracle",              "#ff7f0e"),
]

import math
def _is_num(x):
    return isinstance(x, (int, float)) and not math.isnan(x)

THRESHOLD = 1024  # ctx<1024 → PLD, else Async

speedups = {}
abs_tok_s = {}
for wkey, _ in WORKLOADS:
    with open(RAW / f"regimespec_{wkey}_v3.json") as f:
        d = json.load(f)
    avgs = dict(d["averages_tok_s"])

    # 새 RegimeSpec(PLD if ctx<1024 else Async) 시뮬레이션
    pld_d = d["details"]["pld"]
    async_d = d["details"].get("async", [])
    n = min(len(pld_d), len(async_d))
    sim = []
    for i in range(n):
        ctx = pld_d[i]["ctx_len"]
        if ctx < THRESHOLD:
            ts = pld_d[i].get("tok_s")
        else:
            ts = async_d[i].get("tok_s")
        if _is_num(ts):
            sim.append(ts)
    avgs["regimespec"] = (sum(sim) / len(sim)) if sim else float("nan")

    base = avgs["npu_only"]
    speedups[wkey] = {m: avgs[m] / base for m, _, _ in MODES}
    abs_tok_s[wkey] = avgs

fig, ax = plt.subplots(figsize=(13, 5.5))
n_workloads = len(WORKLOADS)
n_modes = len(MODES)
bar_w = 0.13
x = np.arange(n_workloads)

for i, (mkey, mlabel, color) in enumerate(MODES):
    vals = [speedups[w][mkey] for w, _ in WORKLOADS]
    pos = x + (i - (n_modes - 1) / 2) * bar_w
    bars = ax.bar(pos, vals, bar_w, label=mlabel, color=color,
                  edgecolor="black", linewidth=0.5)
    for b, v, w in zip(bars, vals, [w for w, _ in WORKLOADS]):
        tok = abs_tok_s[w][mkey]
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04,
                f"{v:.2f}\n({tok:.2f})",
                ha="center", va="bottom", fontsize=8)

ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels([label for _, label in WORKLOADS], fontsize=10)
ax.set_ylabel("Speedup vs NPU-only\n(label format: speedup / abs tok/s)", fontsize=10)
ax.set_title("RegimeSpec(PLD+Async) Speedup Across Workloads (v3 + Async)\n"
             "Llama-3.2-1B W8A8 on RK3588, n_gen=32, k=4 — RegimeSpec is simulated from PLD/Async per-prompt data",
             fontsize=10)
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.10), fontsize=9)
ax.grid(axis="y", alpha=0.3)
ymax = max(max(speedups[w].values()) for w, _ in WORKLOADS) * 1.20
ax.set_ylim(0, ymax)

plt.tight_layout()
pdf_path = OUT_DIR / "fig_A_regimespec.pdf"
png_path = OUT_DIR / "fig_A_regimespec.png"
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, bbox_inches="tight", dpi=150)
print(f"Saved {pdf_path}")
print(f"Saved {png_path}")

print("\n--- Summary table ---")
print(f"{'workload':<12s} {'NPU':>6s} {'Seq':>6s} {'PLD':>6s} {'RS':>6s} {'Oracle':>7s}  RS/PLD  RS/Oracle")
for w, _ in WORKLOADS:
    a = abs_tok_s[w]
    rs_pld = a["regimespec"] / a["always_pld"] if a["always_pld"] else 0
    rs_or = a["regimespec"] / a["oracle"] if a["oracle"] else 0
    print(f"{w:<12s} {a['npu_only']:>6.2f} {a['always_seq']:>6.2f} "
          f"{a['always_pld']:>6.2f} {a['regimespec']:>6.2f} "
          f"{a['oracle']:>7.2f}  {rs_pld:>6.2f}  {rs_or:>9.2f}")
