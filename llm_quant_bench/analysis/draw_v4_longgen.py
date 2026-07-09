"""Generate fig_longgen.pdf — backend cross-over at ctx=3500 with varying n_gen."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "llm_quant_bench/results/raw"
OUT = ROOT / "llm_quant_bench/paper/figures"
OUT.mkdir(parents=True, exist_ok=True)

with open(RAW / "long_generation_v4.json") as f:
    d = json.load(f)

n_gens = [r["n_gen"] for r in d["npu"]]
npu_mean = [r["tok_s_overall_mean"] for r in d["npu"]]
npu_std = [r["tok_s_overall_std"] for r in d["npu"]]
cpu_vals = [r["tok_s_overall"] for r in d["cpu_mnn"]]
gpu_vals = [r["tok_s_overall"] for r in d.get("gpu_mnn_opencl", [])]

fig, ax = plt.subplots(figsize=(7.0, 4.4))
xs = np.array(n_gens)

ax.errorbar(xs, npu_mean, yerr=npu_std, fmt='o-', color="#d62728",
            label="NPU (RKLLM)", linewidth=2, markersize=8, capsize=4)
ax.plot(xs, cpu_vals, 's-', color="#1f77b4",
        label="CPU MNN (4×A76)", linewidth=2, markersize=8)
if gpu_vals:
    ax.plot(xs, gpu_vals, '^-', color="#2ca02c",
            label="Mali GPU (MNN OpenCL)", linewidth=2, markersize=8)

# Mark cross-over points
ax.axvline(128, color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax.text(140, 0.4, "GPU > NPU\nat n_gen ≥ 128", fontsize=9, color="#2ca02c")
ax.axvline(512, color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax.text(420, 0.9, "CPU > NPU\nat n_gen = 512", fontsize=9, color="#1f77b4")

ax.set_xscale("log")
ax.set_xticks(xs)
ax.set_xticklabels(xs)
ax.set_xlabel("n_gen (generated tokens)", fontsize=11)
ax.set_ylabel("Overall throughput (tok/s)", fontsize=11)
ax.set_title("Fig 3. Backend cross-over at ctx=3500 with varying n_gen\n"
             "Llama-3.2-1B W8A8, Rock 5B+ (3 repeats, 5-min cooldown between backends)",
             fontsize=10)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3, which="both")
ax.set_ylim(0, max(max(gpu_vals or [0]), max(npu_mean)) * 1.15)

plt.tight_layout()
plt.savefig(OUT / "fig_longgen.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_longgen.png", bbox_inches="tight", dpi=150)
print(f"saved {OUT/'fig_longgen.pdf'}")

print("\nSummary:")
print(f"{'n_gen':>6} {'NPU':>14} {'CPU':>10} {'GPU':>10}")
for i, n in enumerate(n_gens):
    g = gpu_vals[i] if gpu_vals else float('nan')
    print(f"{n:>6} {npu_mean[i]:>8.3f}±{npu_std[i]:.3f}  {cpu_vals[i]:>9.3f}  {g:>9.3f}")
