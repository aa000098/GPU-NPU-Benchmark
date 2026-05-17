#!/usr/bin/env python3
"""Draw fig: hetero vs single-backend overall throughput at ctx=3500."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / "v11_figures"
OUT.mkdir(exist_ok=True)

# Table 6 values (tok/s)
n_g     = np.array([32, 128, 256, 512, 1024, 2048])
npu     = np.array([1.15, 3.30, 4.80, 6.21, 7.27, 7.96])
cpu_t4  = np.array([0.78, 2.64, 4.38, 6.55, 8.69, 10.40])
gpu_low = np.array([1.03, 3.24, 5.05, 7.02, 8.72, 9.91])
hetero  = np.array([1.20, 3.79, 5.91, 8.20, 10.17, 11.56])

# Densely sampled n_g for smooth curves using closed-form eq (2.1)
n_dense = np.logspace(np.log10(20), np.log10(3000), 200)
def overall(ttft, tpot_ms, n):
    return n / (ttft + n * tpot_ms / 1000.0)

# ctx=3500 measurements (from §4.1)
ttft_npu, tpot_npu = 24.17, 113.9
ttft_cpu, tpot_cpu = 38.64, 77.3
ttft_gpu, tpot_gpu = 28.41, 87.0
tpot_cpu_t2 = 74.7  # for hetero decode

npu_d  = overall(ttft_npu, tpot_npu, n_dense)
cpu_d  = overall(ttft_cpu, tpot_cpu, n_dense)
gpu_d  = overall(ttft_gpu, tpot_gpu, n_dense)
het_d  = overall(ttft_npu, tpot_cpu_t2, n_dense)

fig, ax = plt.subplots(figsize=(7.0, 4.6))

# Smooth curves (background)
ax.plot(n_dense, npu_d,  color="#d62728", linewidth=1.2, alpha=0.55)
ax.plot(n_dense, cpu_d,  color="#1f77b4", linewidth=1.2, alpha=0.55)
ax.plot(n_dense, gpu_d,  color="#2ca02c", linewidth=1.2, alpha=0.55)
ax.plot(n_dense, het_d,  color="#9467bd", linewidth=2.4, alpha=0.95)

# Markers at measurement points
ax.plot(n_g, npu,     "D", color="#d62728", markersize=6.5, label="NPU only", linewidth=0)
ax.plot(n_g, cpu_t4,  "o", color="#1f77b4", markersize=6.5, label="cpu_t4 only", linewidth=0)
ax.plot(n_g, gpu_low, "^", color="#2ca02c", markersize=6.5, label="gpu_low only", linewidth=0)
ax.plot(n_g, hetero,  "s", color="#9467bd", markersize=8.0,
        label="Hetero (NPU prefill + cpu_t2 decode)", linewidth=0)

# Asymptote: cpu_t2 TPOT 74.7 ms -> 13.4 tok/s
ax.axhline(13.4, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
ax.text(2200, 13.55, "cpu_t2 decode limit (13.4 tok/s)",
        fontsize=8, color="gray", ha="right")

# Annotate hetero gain at each point
for i, n in enumerate(n_g):
    best = max(npu[i], cpu_t4[i], gpu_low[i])
    gain = hetero[i] / best
    ax.annotate(f"{gain:.2f}x",
                xy=(n, hetero[i]), xytext=(0, 7),
                textcoords="offset points",
                fontsize=7.5, color="#5e3a82", ha="center", fontweight="bold")

ax.set_xscale("log")
ax.set_xlabel("Generation length n_g (tokens)")
ax.set_ylabel("Overall throughput (tok/s)")
ax.set_title("ctx=3500: hetero vs single-backend overall throughput\n"
             "(tok/s = n_g / (TTFT + n_g x TPOT))")
ax.set_xticks([32, 128, 256, 512, 1024, 2048])
ax.set_xticklabels(["32", "128", "256", "512", "1024", "2048"])
ax.minorticks_off()
ax.grid(True, which="both", linestyle=":", alpha=0.5)
ax.set_ylim(0, 14.5)
ax.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.95)

plt.tight_layout()
fig.savefig(OUT / "fig4_hetero.png", dpi=170, bbox_inches="tight")
fig.savefig(OUT / "fig4_hetero.pdf", bbox_inches="tight")
plt.close(fig)
print(f"saved: {OUT / 'fig4_hetero.png'}")
print(f"saved: {OUT / 'fig4_hetero.pdf'}")
