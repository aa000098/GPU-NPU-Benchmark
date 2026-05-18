#!/usr/bin/env python3
"""Draw fig: ctx=1024 decode TPOT op-level breakdown across backends."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / "v11_figures"
OUT.mkdir(exist_ok=True)

# Backends and op categories (5 grouped categories for readability)
backends = ["cpu_t4", "cpu_t2", "gpu_low", "NPU est"]

# (op_category, [cpu_t4, cpu_t2, gpu_low, NPU])
# Values in ms/token, summed for 16 layers (or 1 for LM head)
op_data = [
    # proj2k = QKV proj + O proj (×16 each)
    ("Proj (QKV+O)",     [6.6 + 6.6,  6.4 + 6.4,  13.1 + 13.1,  10 + 7]),
    # FFN = gate + up + down (×16 each)
    ("FFN (gate+up+down)", [13.0 * 3,   12.6 * 3,   18.4 * 3,    13 * 3]),
    ("LM head",          [12.6,         12.3,        13.5,         10]),
    ("Attention",        [0.8,          0.8,         6.0,          2]),
    ("Norm/RoPE/Other",  [0.2,          0.2,         6.0,          0]),
]

totals = [65.9, 63.9, 107.0, 68.0]

colors = {
    "Proj (QKV+O)":      "#4c72b0",
    "FFN (gate+up+down)": "#dd8452",
    "LM head":           "#55a467",
    "Attention":         "#c44e52",
    "Norm/RoPE/Other":   "#8172b3",
}

fig, ax = plt.subplots(figsize=(7.0, 4.6))
x = np.arange(len(backends))
bottom = np.zeros(len(backends))
width = 0.55

for cat, vals in op_data:
    vals = np.array(vals)
    bars = ax.bar(x, vals, width, bottom=bottom, label=cat, color=colors[cat],
                  edgecolor="white", linewidth=0.6)
    # Annotate segment when ≥ 5% of largest total
    for i, v in enumerate(vals):
        if v >= 5.0:
            ax.text(x[i], bottom[i] + v / 2, f"{v:.1f}",
                    ha="center", va="center", fontsize=8.0,
                    color="white", fontweight="bold")
    bottom += vals

# Total annotations on top
for i, t in enumerate(totals):
    ax.text(x[i], t + 1.5, f"{t:.1f} ms",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(backends)
ax.set_ylabel("TPOT (ms/token), 16-layer cumulative")
ax.set_title("ctx=1024 decode TPOT decomposition by op category\n"
             "(3 large matmuls account for ~95% of decode time)")
ax.set_ylim(0, max(totals) * 1.15)
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
ax.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.95,
          handlelength=1.2, handletextpad=0.5, borderpad=0.4, labelspacing=0.3)

plt.tight_layout()
fig.savefig(OUT / "fig3_op_breakdown.png", dpi=170, bbox_inches="tight")
fig.savefig(OUT / "fig3_op_breakdown.pdf", bbox_inches="tight")
plt.close(fig)
print(f"saved: {OUT / 'fig3_op_breakdown.png'}")
print(f"saved: {OUT / 'fig3_op_breakdown.pdf'}")
