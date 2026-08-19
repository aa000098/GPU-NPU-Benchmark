"""
TurboQuant Analysis — KV 캐시 양자화가 Decode 성능에 미치는 영향 분석

RK3588에서 실제 가능한 W8A8 조합:
  - CPU W8A8 (M9): weight=1B, kv=2B (MNN FP16 KV cache)
  - GPU W8A8 (M9): weight=1B, kv=2B (MNN FP16 KV cache)
  - NPU W8A8 (R1): weight=1B, kv=1B (RKNN INT8 KV cache)

핵심 분석:
1. Decode에서 가중치 로딩 vs KV 캐시 로딩 바이트 분해
2. 시퀀스 길이에 따른 KV 캐시 지배 전환점
3. CPU(FP16 KV) vs NPU(INT8 KV): KV 크기 차이가 decode에 미치는 영향
4. 실측 decode tok/s와 이론적 메모리 읽기량의 상관관계

Usage:
    python analysis/turboquant_analysis.py
"""
import os
import csv
import json
import argparse
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, "config", "roofline_config.yaml")


def load_config(config_path=None):
    path = config_path or DEFAULT_CONFIG
    with open(path) as f:
        return yaml.safe_load(f)


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


REAL_CONFIGS = {
    "CPU-W8A8 (M9)": {
        "bytes_per_weight": 1,
        "bytes_per_kv": 1,       # MNN with -qatten 1: INT8 KV cache
        "bytes_per_activation": 1,
        "backend": "cpu_mnn",
        "config_id": "M9",
        "color": "#3498DB",
        "linestyle": "-",
        "marker": "o",
    },
    "GPU-W8A8 (M9)": {
        "bytes_per_weight": 1,
        "bytes_per_kv": 1,       # MNN with -qatten 1: INT8 KV cache
        "bytes_per_activation": 1,
        "backend": "gpu_mnn",
        "config_id": "M9",
        "color": "#2ECC71",
        "linestyle": "--",
        "marker": "s",
    },
    "NPU-W8A8 (R1)": {
        "bytes_per_weight": 1,
        "bytes_per_kv": 1,       # RKNN INT8 KV cache
        "bytes_per_activation": 1,
        "backend": "npu_rknn",
        "config_id": "R1",
        "color": "#E74C3C",
        "linestyle": "-",
        "marker": "^",
    },
}


def decode_bandwidth_decomposition(seq_lengths, arch):
    """Decode 1토큰 생성 시 읽는 바이트 분해."""
    d = arch["hidden_dim"]
    h = arch["num_heads"]
    hd = arch["head_dim"]
    kv_h = arch["num_kv_heads"]
    L = arch["num_layers"]
    ff = arch["intermediate_size"]

    results = {}
    for name, cfg in REAL_CONFIGS.items():
        bw = cfg["bytes_per_weight"]
        bkv = cfg["bytes_per_kv"]
        ba = cfg["bytes_per_activation"]

        qkv_out_dim = d + 2 * kv_h * hd
        weight_bytes_per_layer = (
            d * qkv_out_dim * bw + d * d * bw + 3 * d * ff * bw
        )
        total_weight_bytes = weight_bytes_per_layer * L

        results[name] = {}
        for S in seq_lengths:
            kv_bytes_per_layer = 2 * S * kv_h * hd * bkv
            total_kv_bytes = kv_bytes_per_layer * L

            total_bytes = total_weight_bytes + total_kv_bytes
            kv_ratio = total_kv_bytes / total_bytes if total_bytes > 0 else 0

            flops_per_layer = (
                2 * 1 * d * qkv_out_dim +
                2 * h * 1 * S * hd * 2 +
                2 * 1 * d * d +
                2 * 1 * d * ff * 3
            )
            total_flops = flops_per_layer * L

            results[name][S] = {
                "weight_bytes": total_weight_bytes,
                "kv_bytes": total_kv_bytes,
                "total_bytes": total_bytes,
                "kv_ratio": kv_ratio,
                "total_flops": total_flops,
                "intensity": total_flops / total_bytes if total_bytes > 0 else 0,
            }
    return results


def plot_bandwidth_decomposition(decomp, seq_lengths, output_dir):
    """가중치 vs KV 캐시 바이트 스택 바."""
    names = list(decomp.keys())
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    for idx, name in enumerate(names):
        ax = axes[idx]
        cfg = REAL_CONFIGS[name]
        seq_data = decomp[name]
        sls = sorted(seq_data.keys())

        weight_mb = [seq_data[s]["weight_bytes"] / 1e6 for s in sls]
        kv_mb = [seq_data[s]["kv_bytes"] / 1e6 for s in sls]

        x = np.arange(len(sls))
        w = 0.6

        ax.bar(x, weight_mb, w, label=f'Weights ({cfg["bytes_per_weight"]}B)',
               color='#3498DB', alpha=0.85)
        ax.bar(x, kv_mb, w, bottom=weight_mb,
               label=f'KV Cache ({cfg["bytes_per_kv"]}B)',
               color='#E74C3C', alpha=0.85)

        for i, s in enumerate(sls):
            ratio = seq_data[s]["kv_ratio"]
            total = weight_mb[i] + kv_mb[i]
            ax.text(i, total + 2, f'{ratio:.0%}', ha='center', fontsize=7,
                    color='#E74C3C', fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(sls, fontsize=8, rotation=45)
        ax.set_xlabel('Sequence Length', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Bytes Read per Decode Step (MB)', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.2)

    fig.suptitle('TurboQuant: Decode Bandwidth Decomposition\n'
                 'Red % = KV cache share of total read',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir, f"turboquant_bandwidth_decomp.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved turboquant_bandwidth_decomp.pdf/png")


def plot_kv_dominance(decomp, seq_lengths, output_dir):
    """KV 캐시 비율 vs 시퀀스 길이."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, seq_data in decomp.items():
        cfg = REAL_CONFIGS[name]
        sls = sorted(seq_data.keys())
        kv_pct = [seq_data[s]["kv_ratio"] * 100 for s in sls]
        ax.plot(sls, kv_pct, color=cfg["color"], linestyle=cfg["linestyle"],
                marker=cfg["marker"], linewidth=2, markersize=6, label=name)

    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.7)
    ax.text(seq_lengths[0] * 1.2, 52, 'KV = 50% (dominance threshold)',
            fontsize=9, color='gray')

    ax.set_xscale('log', base=2)
    ax.set_xlabel('Sequence Length', fontsize=12)
    ax.set_ylabel('KV Cache % of Total Read', fontsize=12)
    ax.set_title('TurboQuant: KV Cache Dominance\n'
                 'NPU(1B KV) reaches dominance later than CPU/GPU(2B KV)',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=10)
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir, f"turboquant_kv_dominance.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved turboquant_kv_dominance.pdf/png")


def plot_bytes_vs_speed(decomp, seq_lengths, output_dir):
    """총 읽기 바이트 vs 실측 decode tok/s."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sls = sorted(seq_lengths)

    # Left: Total bytes
    for name, seq_data in decomp.items():
        cfg = REAL_CONFIGS[name]
        total_mb = [seq_data[s]["total_bytes"] / 1e6 for s in sls]
        ax1.plot(sls, total_mb, color=cfg["color"], linestyle=cfg["linestyle"],
                 marker=cfg["marker"], linewidth=2, markersize=6, label=name)

    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Sequence Length', fontsize=11)
    ax1.set_ylabel('Total Bytes per Decode Step (MB)', fontsize=11)
    ax1.set_title('Theoretical Memory Read per Token', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # Right: Measured decode tok/s
    decode_data = load_csv(os.path.join(PROJECT_DIR, "results", "raw", "decode_sweep.csv"))
    name_to_key = {
        "CPU-W8A8 (M9)": "cpu_mnn/M9",
        "GPU-W8A8 (M9)": "gpu_mnn/M9",
        "NPU-W8A8 (R1)": "npu_rknn/R1",
    }

    measured = {}
    for row in decode_data:
        key = row["backend"] + "/" + row["config_id"]
        if key not in measured:
            measured[key] = {"x": [], "y": []}
        measured[key]["x"].append(int(row.get("context_len", 0)))
        measured[key]["y"].append(float(row.get("decode_tok_s", 0)))

    for name, cfg in REAL_CONFIGS.items():
        key = name_to_key.get(name)
        if key and key in measured:
            m = measured[key]
            pairs = sorted(zip(m["x"], m["y"]))
            ax2.plot([p[0] for p in pairs], [p[1] for p in pairs],
                     color=cfg["color"], linestyle=cfg["linestyle"],
                     marker=cfg["marker"], linewidth=2, markersize=6, label=name)

    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Sequence Length', fontsize=11)
    ax2.set_ylabel('Decode Speed (tok/s)', fontsize=11)
    ax2.set_title('Measured Decode Throughput', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    fig.suptitle('TurboQuant: Memory Read vs Actual Performance\n'
                 'NPU reads less at long seq (1B KV) but decode still drops due to attention compute',
                 fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir, f"turboquant_bytes_vs_speed.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved turboquant_bytes_vs_speed.pdf/png")


def plot_roofline_shift(decomp, peaks, seq_lengths, output_dir):
    """Decode 산술 밀도 이동 on roofline."""
    fig, ax = plt.subplots(figsize=(12, 8))

    hw_colors = {"cpu": "#3498DB", "gpu": "#2ECC71", "npu": "#E74C3C"}
    backend_labels = {"cpu": "CPU (A76)", "gpu": "GPU (Mali)", "npu": "NPU (RKNN)"}

    for backend, hw in peaks.items():
        pg = hw.get("peak_gflops", 0)
        pb = hw.get("peak_bandwidth_gb_s", 0)
        if pg <= 0 or pb <= 0:
            continue
        x = np.logspace(-2, 4, 500)
        y = np.minimum(pg, pb * x)
        ridge = pg / pb
        ax.loglog(x, y, color=hw_colors.get(backend, "#333"), linewidth=2, alpha=0.7,
                  label=f'{backend_labels.get(backend)} (ridge={ridge:.1f})')

    for name, seq_data in decomp.items():
        cfg = REAL_CONFIGS[name]
        sls = sorted(seq_data.keys())

        if "NPU" in name:
            ref_peak = peaks.get("npu", {})
        else:
            ref_peak = peaks.get("cpu", {})

        pg = ref_peak.get("peak_gflops", 30)
        pb = ref_peak.get("peak_bandwidth_gb_s", 15)

        intensities = []
        achievables = []
        for s in sls:
            intensity = seq_data[s]["intensity"]
            if intensity <= 0:
                continue
            achievable = min(pg, pb * intensity)
            intensities.append(intensity)
            achievables.append(achievable)

        ax.plot(intensities, achievables, color=cfg["color"],
                linestyle=cfg["linestyle"], marker=cfg["marker"],
                linewidth=1.5, markersize=5, alpha=0.8, label=name)

        if intensities:
            ax.annotate(f's={sls[0]}', (intensities[0], achievables[0]),
                        fontsize=7, xytext=(5, -10), textcoords='offset points',
                        color=cfg["color"])
            ax.annotate(f's={sls[-1]}', (intensities[-1], achievables[-1]),
                        fontsize=7, xytext=(5, 5), textcoords='offset points',
                        color=cfg["color"])

    ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=12)
    ax.set_ylabel('Achievable Performance (GFLOPS)', fontsize=12)
    ax.set_title('TurboQuant: Decode Intensity Shift on Roofline\n'
                 'Longer sequences → lower intensity → more memory-bound',
                 fontsize=13, fontweight='bold')
    ax.set_xlim(0.1, 1000)
    ax.grid(True, which="both", ls="-", alpha=0.15)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()

    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(output_dir, f"turboquant_roofline_shift.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved turboquant_roofline_shift.pdf/png")


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Analysis")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--raw_dir", type=str,
                        default=os.path.join(PROJECT_DIR, "results", "raw"))
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    arch = cfg["model_arch"]
    seq_lengths = cfg["llm_profiling"]["seq_lengths"]

    peaks = load_json(os.path.join(args.raw_dir, "hardware_peaks.json"))
    if not peaks:
        peaks = {
            "cpu": {"peak_gflops": 56, "peak_bandwidth_gb_s": 26},
            "gpu": {"peak_gflops": 15, "peak_bandwidth_gb_s": 16},
            "npu": {"peak_gflops": 40, "peak_bandwidth_gb_s": 8},
        }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join(PROJECT_DIR, "results", f"turboquant_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f"TurboQuant Analysis — {arch['name']}")
    print(f"  CPU-W8A8: weight=1B, kv=1B (INT8, -qatten 1)")
    print(f"  GPU-W8A8: weight=1B, kv=1B (INT8, -qatten 1)")
    print(f"  NPU-W8A8: weight=1B, kv=1B (INT8)")
    print(f"{'='*70}")

    decomp = decode_bandwidth_decomposition(seq_lengths, arch)

    for name in REAL_CONFIGS:
        print(f"\n  {name}:")
        print(f"  {'SeqLen':>7s} {'Weight(MB)':>10s} {'KV(MB)':>10s} {'Total(MB)':>10s} "
              f"{'KV%':>6s} {'Intensity':>10s}")
        for S in seq_lengths:
            d = decomp[name][S]
            print(f"  {S:>7d} {d['weight_bytes']/1e6:>10.1f} {d['kv_bytes']/1e6:>10.1f} "
                  f"{d['total_bytes']/1e6:>10.1f} {d['kv_ratio']*100:>5.1f}% "
                  f"{d['intensity']:>10.3f}")

    print(f"\n  Generating plots...")
    plot_bandwidth_decomposition(decomp, seq_lengths, output_dir)
    plot_kv_dominance(decomp, seq_lengths, output_dir)
    plot_bytes_vs_speed(decomp, seq_lengths, output_dir)
    plot_roofline_shift(decomp, peaks, seq_lengths, output_dir)

    # Save CSV
    csv_path = os.path.join(args.raw_dir, "turboquant_decomposition.csv")
    rows = []
    for name in REAL_CONFIGS:
        for S in seq_lengths:
            d = decomp[name][S]
            rows.append({
                "config": name, "seq_len": S,
                "weight_bytes": d["weight_bytes"], "kv_bytes": d["kv_bytes"],
                "total_bytes": d["total_bytes"],
                "kv_ratio": round(d["kv_ratio"], 4),
                "intensity": round(d["intensity"], 4),
            })
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV: {csv_path}")
    print(f"  Figures: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
