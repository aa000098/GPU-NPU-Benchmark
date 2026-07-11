"""Generate fig_prefill_ttft for v7 — standalone TTFT figure.

Splits the old combined fig_backends (TTFT + decode) so that:
  - Fig 1 (this script): TTFT only — for the prefill discussion
  - Fig 2 (existing fig_roofline): decode + LPDDR5 roofline — for the decode discussion
  - Fig 3 (fig_longgen, separate): long-generation cross-over

Data: matches Table 1 of v7_paper.md.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "llm_quant_bench/paper/v7/v7_figures"
OUT.mkdir(parents=True, exist_ok=True)

CTX = [15, 512, 1024, 2048, 3500]
TTFT = {  # seconds — Table 1 of v7_paper.md
    "NPU": [0.128, 1.95, 4.42, 10.85, 23.06],
    "CPU": [0.201, 3.55, 8.10, 21.74, 38.42],
    "GPU": [0.269, 2.87, 5.99, 14.34, 28.83],
}

fig, ax = plt.subplots(figsize=(7.0, 4.2))
xs = np.array(CTX)

ax.plot(xs, TTFT["NPU"], 'o-', color="#d62728",
        label="NPU (RKLLM)", linewidth=2, markersize=7)
ax.plot(xs, TTFT["CPU"], 's-', color="#1f77b4",
        label="CPU MNN (4×A76)", linewidth=2, markersize=7)
ax.plot(xs, TTFT["GPU"], '^-', color="#2ca02c",
        label="Mali GPU (MNN OpenCL)", linewidth=2, markersize=7)

ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
           label="Interactive SLO 500 ms")

# annotate NPU advantage at largest ctx
ax.annotate(f"NPU vs CPU = {TTFT['CPU'][-1]/TTFT['NPU'][-1]:.2f}×",
            xy=(3500, TTFT["NPU"][-1]), xytext=(700, 28),
            fontsize=9, color="#d62728")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(CTX)
ax.set_xticklabels(CTX)
ax.set_xlabel("Context length (tokens)", fontsize=11)
ax.set_ylabel("TTFT (sec, log scale)", fontsize=11)
ax.set_title("Fig 1. Prefill TTFT by context length (Llama-3.2-1B W8A8)",
             fontsize=11)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3, which="both")

plt.tight_layout()
plt.savefig(OUT / "fig_prefill_ttft.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_prefill_ttft.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"saved {OUT/'fig_prefill_ttft.pdf'}")

# Also print numerical summary for paper writing
print("\nTTFT (sec) by ctx:")
print(f"{'ctx':>5} | {'NPU':>7} {'CPU':>7} {'GPU':>7} | NPU advantage vs CPU")
for i, c in enumerate(CTX):
    n, cp, g = TTFT["NPU"][i], TTFT["CPU"][i], TTFT["GPU"][i]
    print(f"{c:>5} | {n:>7.3f} {cp:>7.3f} {g:>7.3f} | {cp/n:.2f}×")
