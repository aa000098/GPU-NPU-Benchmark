"""§4.7 Table 9 figure: MNN built-in quant_qkv variants vs our PQ NEON."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "llm_quant_bench/paper/v8/v8_figures"
OUT.mkdir(parents=True, exist_ok=True)

ctx = [15, 512, 1024, 2048, 3500]
q0 = [16.58, 14.98, 15.37, 13.93, 12.84]   # no KV quant (baseline)
q1 = [15.37, 14.79, 14.41, 13.70, 12.59]   # int8 K
q2 = [15.17, 13.11, 14.30, 12.75, 8.87]    # fp8 V
q3 = [9.74, 13.21, 14.27, 8.10, 7.93]      # int8 K + V
q4 = [9.60, 9.37, 8.46, 8.26, 7.75]        # int8 K + V + GEMM

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(ctx, q0, 'o-', color="#1f77b4", label="q0: no KV quant (fp16, baseline)", linewidth=2.5, markersize=8)
ax.plot(ctx, q1, 's-', color="#ff7f0e", label="q1: int8 K", linewidth=1.5)
ax.plot(ctx, q2, '^-', color="#2ca02c", label="q2: fp8 V", linewidth=1.5)
ax.plot(ctx, q3, 'v-', color="#d62728", label="q3: int8 K + int8 V", linewidth=1.5)
ax.plot(ctx, q4, 'D-', color="#9467bd", label="q4: int8 K + V + int8 GEMM (SDOT)", linewidth=1.5)

ax.set_xscale("log")
ax.set_xticks(ctx)
ax.set_xticklabels(ctx)
ax.set_xlabel("Context length (tokens)", fontsize=11)
ax.set_ylabel("Decode throughput (tok/s)", fontsize=11)
ax.set_title("Fig 6. MNN built-in quant_qkv variants on ARM Cortex-A76\n"
             "(baseline q0 wins every cell — i8mm absent → dequant overhead dominates)",
             fontsize=10.5)
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(7, 18)
plt.tight_layout()
plt.savefig(OUT / "fig6_quant_qkv_sweep.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig6_quant_qkv_sweep.png", bbox_inches="tight", dpi=150)
print(f"saved {OUT/'fig6_quant_qkv_sweep.pdf'}")
