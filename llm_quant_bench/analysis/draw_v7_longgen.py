"""Generate fig_longgen for v7 — 6 points: n_gen 32, 128, 256, 512, 1024, 2048.

Combines:
  - cooldown run logs (32, 128, 256, 512)
  - extension run JSON (1024, 2048)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "llm_quant_bench/results/raw"
OUT = ROOT / "llm_quant_bench/paper/v7/v7_figures"
OUT.mkdir(parents=True, exist_ok=True)

# Manual aggregation from logs (cooldown run) and JSON (extension run)
data = {
    "npu": {  # n_gen → (mean, std)
        32: (1.146, 0.026), 128: (2.996, 0.006),
        256: (4.063, 0.002), 512: (4.912, 0.008),
        1024: (5.179, 0.257), 2048: (5.519, 0.196),
    },
    "cpu": {
        32: (0.788, 0), 128: (2.648, 0), 256: (4.385, 0), 512: (6.519, 0),
        1024: (8.659, 0), 2048: (10.084, 0),
    },
    "gpu": {
        32: (1.031, 0), 128: (3.323, 0), 256: (5.117, 0), 512: (7.154, 0),
        1024: (8.701, 0), 2048: (9.228, 0),
    },
}
ns = sorted(data["npu"].keys())

fig, ax = plt.subplots(figsize=(7.4, 4.5))

def plot_one(d, marker, color, label):
    means = [d[n][0] for n in ns]
    stds  = [d[n][1] for n in ns]
    ax.errorbar(ns, means, yerr=stds, fmt=marker+'-', color=color,
                label=label, linewidth=2, markersize=8, capsize=4)

plot_one(data["npu"], 'o', "#d62728", "NPU (RKLLM)")
plot_one(data["cpu"], 's', "#1f77b4", "CPU MNN (4×A76)")
plot_one(data["gpu"], '^', "#2ca02c", "Mali GPU (MNN OpenCL)")

# Cross-over markers (3개)
for n_cross, label, color, ypos in [
    (128,  "GPU > NPU\nat n_gen ≥ 128", "#2ca02c", 0.5),
    (512,  "CPU > NPU\nat n_gen ≥ 512", "#1f77b4", 1.0),
    (1024, "GPU ≈ CPU\n(872 estimated)", "black",  1.5),
]:
    ax.axvline(n_cross, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(n_cross * 1.05, ypos, label, fontsize=8, color=color)

ax.set_xscale("log")
ax.set_xticks(ns)
ax.set_xticklabels(ns)
ax.set_xlabel("n_gen (generated tokens)", fontsize=11)
ax.set_ylabel("Overall throughput (tok/s)", fontsize=11)
ax.set_title("Backend cross-over at ctx=3500, varying n_gen\n"
             "Llama-3.2-1B W8A8, Rock 5B+ (3 repeats)",
             fontsize=10)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3, which="both")
all_vals = [v[0] for d_ in data.values() for v in d_.values()]
ax.set_ylim(0, max(all_vals) * 1.15)

plt.tight_layout()
plt.savefig(OUT / "fig_longgen.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_longgen.png", bbox_inches="tight", dpi=150)
print(f"saved {OUT/'fig_longgen.pdf'}")

# Also write the CSV
csv_path = ROOT / "llm_quant_bench/paper/v7/v7_data/v7_longgen_extended.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)
with open(csv_path, "w") as f:
    f.write("n_gen,backend,tok_s_mean,tok_s_std\n")
    for backend, dd in [("NPU_RKLLM", data["npu"]),
                          ("CPU_MNN_W8A8", data["cpu"]),
                          ("GPU_OpenCL_W8A8", data["gpu"])]:
        for n in ns:
            m, s = dd[n]
            f.write(f"{n},{backend},{m:.3f},{s:.3f}\n")
print(f"saved {csv_path}")
