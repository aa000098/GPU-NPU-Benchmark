"""§4 figure: NPU/GPU/CPU baseline + CPU+PQ projected for long-ctx decode."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "llm_quant_bench/paper/v9/v9_figures"
OUT.mkdir(parents=True, exist_ok=True)

ctx = [15, 512, 1024, 2048, 3500]
# Measured baselines (RKLLM rkllm_run for NPU; MNN llm_bench for CPU/GPU)
npu = [20.60, 19.13, 17.28, 14.72, 12.11]
cpu = [16.18, 15.01, 14.80, 13.84, 13.05]
gpu = [14.93, 13.76, 13.04, 12.68, 11.41]

# CPU+PQ projection: assume attention is 30% of decode at ctx=3500, scaling with ctx
# Attention fraction: 5% at ctx=15, 15% at ctx=512, 22% at ctx=1024, 28% at ctx=2048, 33% at ctx=3500
# (rough estimates based on Q*K^T cost growing linearly with ctx vs FFN constant)
att_frac = [0.05, 0.15, 0.22, 0.28, 0.33]
neon_speedup_attn = 13.8  # measured at ctx=3500; conservative same factor at other ctx
cpu_pq = []
for c, f in zip(cpu, att_frac):
    new_decode_time = (1 - f) + f / neon_speedup_attn  # normalized
    cpu_pq.append(c / new_decode_time)

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(ctx, npu, 'o-', color="#d62728", label="NPU (RKLLM)", linewidth=2, markersize=8)
ax.plot(ctx, cpu, 's-', color="#1f77b4", label="CPU MNN (4×A76, fp16 KV baseline)", linewidth=2, markersize=8)
ax.plot(ctx, gpu, '^-', color="#2ca02c", label="Mali GPU (MNN OpenCL)", linewidth=2, markersize=8)
ax.plot(ctx, cpu_pq, 'D-', color="#9467bd", label="CPU + PQ NEON (projected from microbench)",
        linewidth=2.5, markersize=9)
ax.axhline(22.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="LPDDR5 roofline 22 tok/s")

# Annotate
ax.annotate(f"{cpu_pq[-1]:.1f}", xy=(3500, cpu_pq[-1]), xytext=(2200, cpu_pq[-1]+1.5),
            fontsize=9, color="#9467bd", fontweight="bold")
ax.annotate(f"{npu[-1]:.1f} (NPU)", xy=(3500, npu[-1]), xytext=(2200, npu[-1]-1.5),
            fontsize=9, color="#d62728")

ax.set_xscale("log")
ax.set_xticks(ctx)
ax.set_xticklabels(ctx)
ax.set_xlabel("Context length (tokens)", fontsize=11)
ax.set_ylabel("Decode throughput (tok/s)", fontsize=11)
ax.set_title("Fig 1. RK3588 Three-backend baseline + CPU+PQ projection\n"
             "(Llama-3.2-1B W8A8, n_gen=32 decode rate)", fontsize=10.5)
ax.legend(loc="lower left", fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(8, 24)
plt.tight_layout()
plt.savefig(OUT / "fig1_backend_comparison.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig1_backend_comparison.png", bbox_inches="tight", dpi=150)
print(f"saved {OUT/'fig1_backend_comparison.pdf'}")
print()
print("Projection numbers:")
print(f"{'ctx':>5} {'NPU':>6} {'CPU':>6} {'GPU':>6} {'CPU+PQ':>7} {'PQ vs NPU':>10}")
for i, c in enumerate(ctx):
    print(f"{c:>5} {npu[i]:>6.2f} {cpu[i]:>6.2f} {gpu[i]:>6.2f} {cpu_pq[i]:>7.2f} {cpu_pq[i]/npu[i]:>9.2f}x")
