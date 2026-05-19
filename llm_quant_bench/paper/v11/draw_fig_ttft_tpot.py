#!/usr/bin/env python3
"""Draw §4.1 figures: TTFT and TPOT across backends/ctx as separate figures."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
DATA = ROOT / "v11_data" / "per_backend_summary.csv"
OUT = ROOT / "v11_figures"
OUT.mkdir(exist_ok=True)

rows = list(csv.DictReader(open(DATA)))
backends = ["cpu_t4_c47", "cpu_t2_c45", "gpu_low", "npu_rkllm"]
labels = {
    "cpu_t4_c47":  "CPU MNN (t=4, A76 x4)",
    "cpu_t2_c45":  "CPU MNN (t=2, A76 cluster pin)",
    "gpu_low":     "GPU MNN OpenCL (Mali-G610)",
    "npu_rkllm":   "NPU RKLLM (rkllm_run)",
}
colors = {
    "cpu_t4_c47":  "#1f77b4",
    "cpu_t2_c45":  "#17becf",
    "gpu_low":     "#2ca02c",
    "npu_rkllm":   "#d62728",
}
markers = {
    "cpu_t4_c47":  "o",
    "cpu_t2_c45":  "s",
    "gpu_low":     "^",
    "npu_rkllm":   "D",
}


def collect(metric):
    out = {}
    for b in backends:
        ctxs, vals = [], []
        for r in rows:
            if r["backend"] == b:
                ctxs.append(int(r["ctx"]))
                vals.append(float(r[metric]))
        out[b] = (ctxs, vals)
    return out


# --- Figure 1: TTFT ---
fig, ax = plt.subplots(figsize=(6.0, 4.4))
ttft = collect("TTFT_s")
for b in backends:
    ctxs, vals = ttft[b]
    ax.plot(ctxs, vals, marker=markers[b], color=colors[b],
            label=labels[b], linewidth=1.7, markersize=7)
ax.set_xlabel("Prompt length n_p (tokens)")
ax.set_ylabel("TTFT (sec, lower is better)")
ax.set_title("Time-To-First-Token (Prefill latency) vs n_p")
ax.set_yscale("log")
ax.set_xscale("log")
ax.grid(True, which="both", linestyle=":", alpha=0.5)
ax.legend(fontsize=9, loc="upper left")
ax.set_xticks([64, 256, 512, 1024, 2048, 3500])
ax.set_xticklabels(["64", "256", "512", "1024", "2048", "3500"])
ax.minorticks_off()
plt.tight_layout()
fig.savefig(OUT / "fig1_ttft.png", dpi=170, bbox_inches="tight")
fig.savefig(OUT / "fig1_ttft.pdf", bbox_inches="tight")
plt.close(fig)
print(f"saved: {OUT / 'fig1_ttft.png'}")
print(f"saved: {OUT / 'fig1_ttft.pdf'}")

# --- Figure 2: TPOT ---
fig, ax = plt.subplots(figsize=(6.0, 4.4))
tpot = collect("TPOT_ms")
for b in backends:
    ctxs, vals = tpot[b]
    ax.plot(ctxs, vals, marker=markers[b], color=colors[b],
            label=labels[b], linewidth=1.7, markersize=7)
# LPDDR5 fill 27.31 GB/s -> tok/s_max = 22.0 -> TPOT_min = 1000/22.0 = 45.5 ms
ax.axhline(45.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
ax.text(70, 47.0, "LPDDR5 roofline lower bound (45.5 ms/token)",
        fontsize=8.5, color="gray")
ax.set_xlabel("Context length ctx (tokens, prefill state)")
ax.set_ylabel("TPOT (ms/token, lower is better)")
ax.set_title("Time-Per-Output-Token (Decode latency) vs ctx")
ax.set_xscale("log")
ax.grid(True, which="both", linestyle=":", alpha=0.5)
ax.legend(fontsize=9, loc="upper left")
ax.set_xticks([64, 256, 512, 1024, 2048, 3500])
ax.set_xticklabels(["64", "256", "512", "1024", "2048", "3500"])
ax.set_ylim(40, 130)
ax.minorticks_off()
plt.tight_layout()
fig.savefig(OUT / "fig2_tpot.png", dpi=170, bbox_inches="tight")
fig.savefig(OUT / "fig2_tpot.pdf", bbox_inches="tight")
plt.close(fig)
print(f"saved: {OUT / 'fig2_tpot.png'}")
print(f"saved: {OUT / 'fig2_tpot.pdf'}")
