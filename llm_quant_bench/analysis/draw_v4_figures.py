"""Generate v4 figures: Roofline + Multi-backend ctx comparison."""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "llm_quant_bench/results/raw"
OUT = ROOT / "llm_quant_bench/paper/figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Load measurements ──────────────────────────────────────────────────
with open(RAW / "npu_only_kvreuse_v3.json") as f:
    npu_results = json.load(f)
npu = {r["ctx_len"]: r for r in npu_results}

def parse_mnn(p):
    m = re.search(r"\| ([\d.]+) ± [\d.]+<br>([\d.]+) ± [\d.]+ \|\s*$",
                  p.read_text().strip())
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)

CTX = [15, 512, 1024, 2048, 3500]
mnn_cpu = {c: parse_mnn(RAW / f"mnn_cpu_ctx{c}.txt") for c in CTX}
mnn_ocl = {c: parse_mnn(RAW / f"mnn_opencl_ctx{c}.txt") for c in CTX}

# Roofline constants — use fill (write only) as conservative single-direction
# read bandwidth proxy. tinymembench measures: NEON copy 12.91 GB/s (read+write
# concurrent), fill 27.31 GB/s (write only).
BW_FILL = 27.31     # GB/s, fill (single-direction) measured
WEIGHT_GB = 1.24    # Llama-3.2-1B W8A8 weight (1.24B params × 1 byte)
ROOFLINE = BW_FILL / WEIGHT_GB   # ≈ 22.02 tok/s

# Decode tok/s
def decode_tok_s(record):
    """For NPU records, derive from decode_ms_per_tok_mean."""
    return 1000.0 / record["decode_ms_per_tok_mean"]

npu_dec = [decode_tok_s(npu[c]) for c in CTX]
cpu_dec = [mnn_cpu[c][1] for c in CTX]
ocl_dec = [mnn_ocl[c][1] for c in CTX]

# TTFT (s) per backend
npu_ttft = [npu[c]["prefill_ms"] / 1000 for c in CTX]
cpu_ttft = [c / mnn_cpu[c][0] for c in CTX]
ocl_ttft = [c / mnn_ocl[c][0] for c in CTX]

# ── Figure 1: Roofline ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.0, 4.2))
xs = np.array(CTX)
ax.plot(xs, npu_dec, 'o-', color="#d62728", label="NPU (RKLLM)", linewidth=2, markersize=7)
ax.plot(xs, cpu_dec, 's-', color="#1f77b4", label="CPU MNN (4×A76)", linewidth=2, markersize=7)
ax.plot(xs, ocl_dec, '^-', color="#2ca02c", label="Mali GPU (MNN OpenCL)", linewidth=2, markersize=7)
ax.axhline(ROOFLINE, color="black", linestyle="--", linewidth=1.5,
           label=f"Memory-BW roofline = {ROOFLINE:.1f} tok/s")
ax.fill_between([10, 4000], 0, ROOFLINE, alpha=0.04, color="green")

# annotate NPU efficiency at extreme ctx
ax.annotate(f"{npu_dec[0]/ROOFLINE*100:.0f}% of roofline",
            xy=(15, npu_dec[0]), xytext=(80, npu_dec[0]+1.5),
            fontsize=9, color="#d62728")
ax.annotate(f"{npu_dec[-1]/ROOFLINE*100:.0f}% of roofline",
            xy=(3500, npu_dec[-1]), xytext=(900, npu_dec[-1]-2.2),
            fontsize=9, color="#d62728")

ax.set_xscale("log")
ax.set_xticks(CTX)
ax.set_xticklabels(CTX)
ax.set_xlabel("Context length (tokens)", fontsize=11)
ax.set_ylabel("Decode throughput (tok/s)", fontsize=11)
ax.set_title("Fig 1. RK3588 LPDDR5 Roofline + Multi-backend Decode\n"
             "Llama-3.2-1B W8A8, Rock 5B+, n_gen=32",
             fontsize=11)
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0, ROOFLINE * 1.15)
plt.tight_layout()
plt.savefig(OUT / "fig_roofline.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_roofline.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"saved {OUT/'fig_roofline.pdf'}")

# ── Figure 2: Backend comparison (prefill + decode) ────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) TTFT (Time To First Token)
ax1.plot(xs, npu_ttft, 'o-', color="#d62728", label="NPU", linewidth=2, markersize=7)
ax1.plot(xs, cpu_ttft, 's-', color="#1f77b4", label="CPU MNN", linewidth=2, markersize=7)
ax1.plot(xs, ocl_ttft, '^-', color="#2ca02c", label="Mali GPU", linewidth=2, markersize=7)
ax1.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
            label="Interactive SLO 500 ms")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xticks(CTX)
ax1.set_xticklabels(CTX)
ax1.set_xlabel("Context length", fontsize=11)
ax1.set_ylabel("TTFT (sec, log scale)", fontsize=11)
ax1.set_title("(a) Time To First Token", fontsize=11)
ax1.legend(loc="best", fontsize=9)
ax1.grid(alpha=0.3, which="both")

# (b) Decode (linear, with crossover annotation)
ax2.plot(xs, npu_dec, 'o-', color="#d62728", label="NPU", linewidth=2, markersize=7)
ax2.plot(xs, cpu_dec, 's-', color="#1f77b4", label="CPU MNN", linewidth=2, markersize=7)
ax2.plot(xs, ocl_dec, '^-', color="#2ca02c", label="Mali GPU", linewidth=2, markersize=7)
ax2.axhline(ROOFLINE, color="black", linestyle="--", linewidth=1.0, alpha=0.5,
            label=f"Roofline {ROOFLINE:.1f}")
# crossover annotation
ax2.axvline(1024, color="gray", linestyle=":", linewidth=1, alpha=0.7)
ax2.annotate("crossover\nctx≈1024",
             xy=(1024, 13.5), xytext=(1300, 17),
             fontsize=9, color="gray",
             arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7))
ax2.set_xscale("log")
ax2.set_xticks(CTX)
ax2.set_xticklabels(CTX)
ax2.set_xlabel("Context length", fontsize=11)
ax2.set_ylabel("Decode throughput (tok/s)", fontsize=11)
ax2.set_title("(b) Decode  —  NPU loses to CPU at ctx≥1024", fontsize=11)
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 22)

plt.suptitle("Fig 2. Backend-by-Backend Throughput Decomposition (Llama-3.2-1B W8A8)",
             fontsize=12)
plt.tight_layout()
plt.savefig(OUT / "fig_backends.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_backends.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"saved {OUT/'fig_backends.pdf'}")

# ── Print summary numbers ──────────────────────────────────────────────
print("\nSummary:")
print(f"  Roofline = {ROOFLINE:.2f} tok/s (read BW {BW_READ:.1f} GB/s, weight {WEIGHT_GB} GB)")
print(f"\n  ctx | NPU dec | CPU dec | GPU dec | NPU/limit")
for c, n, cp, g in zip(CTX, npu_dec, cpu_dec, ocl_dec):
    print(f"  {c:>4} | {n:>7.2f} | {cp:>7.2f} | {g:>7.2f} | {n/ROOFLINE*100:>6.1f}%")
