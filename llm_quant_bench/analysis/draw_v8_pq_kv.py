"""Generate §4.7 figures: PQ NEON CPU speedup + GPU comparison."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "llm_quant_bench/paper/v8/v8_figures"
OUT.mkdir(parents=True, exist_ok=True)

# Microbench data (ARM NEON, single attention step, 5-iter avg)
ctx = [512, 1024, 2048, 3500]
fp16_neon  = [2.41, 5.03, 20.20, 37.81]   # ms
pq_neon    = [0.53, 0.89, 1.74, 2.74]
pq_naive_c = [2.94, 5.79, 11.59, 20.71]   # rough estimate from speedup × pq_neon

# Mali-G610 OpenCL (only ctx=3500 measured)
fp16_cl_3500 = 2.00
pq_cl_3500   = 10.56

# Figure: ms vs ctx (log-log)
fig, ax = plt.subplots(figsize=(7.0, 4.5))
ax.plot(ctx, fp16_neon, 'o-', color="#1f77b4", label="fp16 attention (NEON)", linewidth=2, markersize=8)
ax.plot(ctx, pq_neon, 's-', color="#d62728", label="PQ attention (NEON, our work)", linewidth=2, markersize=8)
ax.scatter([3500], [pq_cl_3500], color="#9467bd", marker='^', s=120,
           label=f"PQ attention (Mali GPU OpenCL): {pq_cl_3500:.1f} ms — slower")
ax.scatter([3500], [fp16_cl_3500], color="#2ca02c", marker='v', s=120,
           label=f"fp16 attention (Mali GPU OpenCL): {fp16_cl_3500:.1f} ms")
ax.annotate(f"{fp16_neon[-1]/pq_neon[-1]:.1f}× speedup\n(NEON CPU)",
            xy=(3500, pq_neon[-1]), xytext=(1500, 0.7),
            fontsize=10, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728"))
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(ctx)
ax.set_xticklabels(ctx)
ax.set_xlabel("Context length (tokens)", fontsize=11)
ax.set_ylabel("Attention wall time per step (ms, log)", fontsize=11)
ax.set_title("Fig 4. ARM NEON PQ codebook vs fp16 baseline\n"
             "(microbench, 1 attention step, 8 heads × 64 dim)", fontsize=11)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(OUT / "fig4_pq_neon_speedup.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig4_pq_neon_speedup.png", bbox_inches="tight", dpi=150)
print(f"saved {OUT/'fig4_pq_neon_speedup.pdf'}")
