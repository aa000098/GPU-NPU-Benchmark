"""
RK3588 Power/Energy Analysis (간접 측정)

직접 전류 측정 안 됨 (USB-PD current_now=0).
대안:
1. NPU/CPU/GPU thermal sensor 변화로 활성 추적
2. RK3588 SoC 사양 + datasheet 기반 모델링 (NPU active 1.5W, big core ~1W/core)
3. tok/J 추정값 비교

Thermal:
- /sys/class/thermal/thermal_zone6/temp = npu-thermal (mC)
- /sys/class/thermal/thermal_zone1,2 = bigcore0,1
- /sys/class/thermal/thermal_zone5 = gpu-thermal
"""
import subprocess
import time
import os
import json
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/analysis/mechanism_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

THERMAL_ZONES = {
    "soc": "/sys/class/thermal/thermal_zone0/temp",
    "bigcore0": "/sys/class/thermal/thermal_zone1/temp",
    "bigcore1": "/sys/class/thermal/thermal_zone2/temp",
    "gpu": "/sys/class/thermal/thermal_zone5/temp",
    "npu": "/sys/class/thermal/thermal_zone6/temp",
}

# RK3588 datasheet 기반 추정 전력 (W)
POWER_MODEL = {
    "idle": 1.0,            # SoC base power
    "cpu_a76_per_core": 1.0,  # active big core
    "cpu_a55_per_core": 0.2,  # active little core
    "gpu_active": 2.5,      # Mali-G610 full load
    "npu_per_core": 0.5,    # 3 cores * 0.5 = 1.5W
    "lpddr": 1.5,           # DRAM under load
}


def read_thermal():
    """Read all thermal zones in mC."""
    result = {}
    for name, path in THERMAL_ZONES.items():
        try:
            with open(path) as f:
                result[name] = int(f.read().strip()) / 1000  # mC → C
        except:
            result[name] = None
    return result


def estimate_power(npu_load_pct, cpu_a76_count_active, gpu_load_pct=0):
    """추정 전력 (W) — datasheet 기반."""
    p = POWER_MODEL["idle"]
    p += POWER_MODEL["npu_per_core"] * 3 * (npu_load_pct / 100)
    p += POWER_MODEL["cpu_a76_per_core"] * cpu_a76_count_active
    p += POWER_MODEL["gpu_active"] * (gpu_load_pct / 100)
    p += POWER_MODEL["lpddr"]  # DRAM (LLM = high BW)
    return p


def compute_energy_per_token():
    """각 구성의 추정 에너지/토큰 (J/tok) 계산."""

    # Throughput 측정값 (논문/실측)
    configs = {
        "NPU-only @ ctx=1024":    {"tok_s": 0.241, "npu_load": 75, "cpu_a76": 1, "gpu": 0},
        "NPU-only @ ctx=2048":    {"tok_s": 0.097, "npu_load": 75, "cpu_a76": 1, "gpu": 0},
        "Sequential @ ctx=1024":  {"tok_s": 0.529, "npu_load": 60, "cpu_a76": 4, "gpu": 0},
        "Sequential @ ctx=2048":  {"tok_s": 0.236, "npu_load": 60, "cpu_a76": 4, "gpu": 0},
        "PLD @ ctx=1024":         {"tok_s": 0.256, "npu_load": 75, "cpu_a76": 1, "gpu": 0},
        "PLD @ ctx=2048":         {"tok_s": 0.130, "npu_load": 75, "cpu_a76": 1, "gpu": 0},
        # CPU-only baseline
        "CPU-only (MNN W8A8)":    {"tok_s": 16.8,  "npu_load": 0,  "cpu_a76": 4, "gpu": 0},
        "GPU-only (MNN OpenCL)":  {"tok_s": 13.9,  "npu_load": 0,  "cpu_a76": 1, "gpu": 95},
    }

    print("=" * 70)
    print("Estimated Energy per Token (RK3588 model-based)")
    print("=" * 70)
    print(f"{'Config':<28s} {'tok/s':>8s} {'Power(W)':>10s} {'J/tok':>10s} {'tok/J':>10s}")

    results = []
    for name, c in configs.items():
        p = estimate_power(c["npu_load"], c["cpu_a76"], c["gpu"])
        j_per_tok = p / c["tok_s"]
        tok_per_j = c["tok_s"] / p
        results.append({"name": name, "tok_s": c["tok_s"], "power_w": p,
                         "j_per_tok": j_per_tok, "tok_per_j": tok_per_j})
        print(f"{name:<28s} {c['tok_s']:>8.3f} {p:>10.2f} {j_per_tok:>10.2f} {tok_per_j:>10.4f}")

    return results


def fig_energy_comparison(results):
    """tok/J vs tok/s 산점도."""
    fig, ax = plt.subplots(figsize=(11, 7))

    colors_map = {
        "NPU-only": "#E74C3C",
        "Sequential": "#9B59B6",
        "PLD": "#3498DB",
        "CPU-only": "#2ECC71",
        "GPU-only": "#F39C12",
    }
    markers_map = {1024: "o", 2048: "s", "default": "^"}

    for r in results:
        # Pick color by family
        color = "#888888"
        for k, v in colors_map.items():
            if k in r["name"]:
                color = v
                break
        marker = "o"
        if "ctx=1024" in r["name"]: marker = "o"
        elif "ctx=2048" in r["name"]: marker = "s"
        elif "CPU-only" in r["name"] or "GPU-only" in r["name"]: marker = "*"

        ax.scatter(r["tok_s"], r["tok_per_j"], c=color, marker=marker,
                   s=200, edgecolors='black', linewidth=1, zorder=5)
        ax.annotate(r["name"], (r["tok_s"], r["tok_per_j"]),
                    fontsize=8, xytext=(8, 5), textcoords='offset points')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Throughput (tokens/sec)', fontsize=12)
    ax.set_ylabel('Energy Efficiency (tokens/Joule)', fontsize=12)
    ax.set_title('RK3588 Energy Efficiency by Configuration\n'
                 '(Power model: NPU 1.5W + CPU 1W/core + GPU 2.5W + DRAM 1.5W + idle 1W)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, which="both", ls="-", alpha=0.15)

    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(OUTPUT_DIR, f"fig_energy.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved fig_energy.pdf/png")


def measure_temperature_trace(duration_s=10):
    """동작 중 온도 시계열 측정 (다른 프로세스 동시 실행 필요)."""
    print(f"\n=== Temperature trace ({duration_s}s) ===")
    samples = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        t = time.perf_counter() - t0
        therm = read_thermal()
        therm["t"] = t
        samples.append(therm)
        time.sleep(0.5)

    # Print summary
    for name in ["soc", "bigcore0", "gpu", "npu"]:
        vals = [s[name] for s in samples if s[name] is not None]
        if vals:
            print(f"  {name:10s}: {min(vals):.1f} → {max(vals):.1f} C (Δ {max(vals)-min(vals):.1f})")
    return samples


def main():
    print("=" * 70)
    print("RK3588 Power/Energy Analysis")
    print("=" * 70)

    # 현재 온도 확인
    print(f"\nCurrent thermal:")
    therm = read_thermal()
    for name, t in therm.items():
        print(f"  {name:10s}: {t:.1f} C")

    # USB-PD current 시도
    try:
        with open("/sys/class/power_supply/tcpm-source-psy-4-0022/current_now") as f:
            current_ua = int(f.read().strip())
            print(f"\nUSB-PD current: {current_ua/1000:.0f} mA (note: 0 means not measuring)")
    except:
        pass

    # 모델 기반 에너지 비교
    results = compute_energy_per_token()
    fig_energy_comparison(results)

    # Save
    summary = {
        "power_model_W": POWER_MODEL,
        "configurations": results,
        "note": "Direct current measurement unavailable (USB-PD reports 0). Estimated from RK3588 datasheet typical values.",
    }
    with open(os.path.join(OUTPUT_DIR, "energy_estimate.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
