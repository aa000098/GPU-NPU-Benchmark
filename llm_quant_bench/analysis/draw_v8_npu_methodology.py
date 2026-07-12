"""§4.6 figure: NPU Python loop vs rkllm_run methodology comparison."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "llm_quant_bench/paper/v8/v8_figures"
OUT.mkdir(parents=True, exist_ok=True)

ctx = [15, 512, 1024, 2048, 3500]
v7_python = [20.80, 15.09, 12.20, 8.46, 6.12]   # tok/s — v7 reported
v8_native = [20.60, 19.13, 17.28, 14.72, 12.11]  # tok/s — rkllm_run

fig, ax = plt.subplots(figsize=(7.0, 4.5))
ax.plot(ctx, v7_python, 'o-', color="#9467bd", label="v7: Python ctypes loop", linewidth=2, markersize=8)
ax.plot(ctx, v8_native, 's-', color="#d62728", label="v8: rkllm_run native", linewidth=2, markersize=8)
ax.axhline(22.0, color="black", linestyle="--", linewidth=1.5, alpha=0.6, label="LPDDR5 roofline 22.0 tok/s")

# Annotate the underestimation
for i, (c, p, n) in enumerate(zip(ctx, v7_python, v8_native)):
    if c >= 1024:
        ax.annotate(f"{n/p:.2f}×", xy=(c, n), xytext=(c, n+1.5),
                    fontsize=9, ha="center", color="#d62728")

ax.set_xscale("log")
ax.set_xticks(ctx)
ax.set_xticklabels(ctx)
ax.set_xlabel("Context length (tokens)", fontsize=11)
ax.set_ylabel("NPU decode throughput (tok/s)", fontsize=11)
ax.set_title("Fig 5. NPU measurement methodology: Python integration overhead\n"
             "(same NPU compute, different host invocation)", fontsize=11)
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0, 24)
plt.tight_layout()
plt.savefig(OUT / "fig5_npu_methodology.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig5_npu_methodology.png", bbox_inches="tight", dpi=150)
print(f"saved {OUT/'fig5_npu_methodology.pdf'}")
