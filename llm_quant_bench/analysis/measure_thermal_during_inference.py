"""
실제 추론 중 thermal sensor 시계열 측정.
보드 식힌 후 각 구성을 30초씩 실행하며 온도 추적.

직접 power 측정 불가 → 온도 변화 + 모델 추정으로 검증.
"""
import os
import sys
import json
import time
import threading
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = "/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/analysis/mechanism_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

THERMAL_ZONES = {
    "soc": "/sys/class/thermal/thermal_zone0/temp",
    "bigcore0": "/sys/class/thermal/thermal_zone1/temp",
    "bigcore1": "/sys/class/thermal/thermal_zone2/temp",
    "gpu": "/sys/class/thermal/thermal_zone5/temp",
    "npu": "/sys/class/thermal/thermal_zone6/temp",
}


class ThermalMonitor:
    def __init__(self, sample_interval_s=0.5):
        self.interval = sample_interval_s
        self.samples = []
        self.running = False
        self.thread = None
        self.t0 = None

    def _read_all(self):
        result = {"t": time.perf_counter() - self.t0}
        for name, path in THERMAL_ZONES.items():
            try:
                with open(path) as f:
                    result[name] = int(f.read().strip()) / 1000
            except:
                result[name] = None
        return result

    def _loop(self):
        while self.running:
            self.samples.append(self._read_all())
            time.sleep(self.interval)

    def start(self):
        self.t0 = time.perf_counter()
        self.running = True
        self.samples = []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)

    def summary(self, t_start=None, t_end=None):
        """주어진 구간의 thermal 통계."""
        s = self.samples
        if t_start is not None:
            s = [x for x in s if x["t"] >= t_start]
        if t_end is not None:
            s = [x for x in s if x["t"] <= t_end]
        if not s:
            return {}
        out = {}
        for name in THERMAL_ZONES:
            vals = [x[name] for x in s if x[name] is not None]
            if vals:
                out[name] = {"min": min(vals), "max": max(vals),
                             "mean": sum(vals) / len(vals), "delta": max(vals) - min(vals)}
        return out


def cool_down_to(target_C=40, max_wait_s=60):
    """SoC 온도가 target 이하가 될 때까지 대기."""
    print(f"[Cool down] waiting until SoC < {target_C}°C...")
    start = time.perf_counter()
    while time.perf_counter() - start < max_wait_s:
        with open(THERMAL_ZONES["soc"]) as f:
            t = int(f.read().strip()) / 1000
        if t < target_C:
            print(f"  SoC = {t:.1f}°C (cool enough)")
            return t
        print(f"  SoC = {t:.1f}°C, waiting...")
        time.sleep(5)
    return t


def run_workload(workload_name, cmd_or_func, duration_s=30):
    """Run a workload and return thermal trace."""
    print(f"\n{'='*70}")
    print(f"Workload: {workload_name} ({duration_s}s)")
    print(f"{'='*70}")

    # Cool down
    initial_temp = cool_down_to(40)

    # Start monitor
    mon = ThermalMonitor(sample_interval_s=0.3)
    mon.start()
    time.sleep(2)  # baseline

    t_start = time.perf_counter() - mon.t0
    if callable(cmd_or_func):
        cmd_or_func(duration_s)
    else:
        # subprocess command
        try:
            subprocess.run(cmd_or_func, shell=True, timeout=duration_s + 10,
                            capture_output=True)
        except subprocess.TimeoutExpired:
            pass
    t_end = time.perf_counter() - mon.t0

    time.sleep(2)  # cool tail
    mon.stop()

    summary = mon.summary(t_start, t_end)
    print(f"\nThermal during workload (t={t_start:.1f}s - {t_end:.1f}s):")
    for name, s in summary.items():
        print(f"  {name:10s}: {s['min']:.1f} → {s['max']:.1f}°C (Δ {s['delta']:.1f}, mean {s['mean']:.1f})")

    return {
        "workload": workload_name,
        "duration_s": duration_s,
        "initial_temp": initial_temp,
        "samples": mon.samples,
        "active_period": (t_start, t_end),
        "summary": summary,
    }


# ════════════════════════════════════════════════════════════════
# Workloads
# ════════════════════════════════════════════════════════════════

def workload_idle(duration_s):
    """Idle baseline."""
    time.sleep(duration_s)


def workload_npu_only(duration_s):
    """NPU-only inference loop."""
    from llm_quant_bench.benchmark.hybrid_runtime_v2 import TargetVerifier, run_npu_only
    target_path = "/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/rkllm_models/Llama-3.2-1B-Instruct_W8A8_RK3588.rkllm"
    target = TargetVerifier(target_path)
    prompt = list(range(100))  # 100 dummy tokens

    end = time.perf_counter() + duration_s
    while time.perf_counter() < end:
        try:
            run_npu_only(target, prompt, n_gen=8)
        except Exception as e:
            print(f"  npu_only err: {e}")
            break
    target.close()


def workload_cpu_mnn(duration_s):
    """CPU MNN inference."""
    cmd = ("/home/hyunho.son/install_files/MNN/build/llm_bench "
           "-m /home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/mnn_models/M9_w8a8_ch_direct/config.json "
           "-a cpu -t 4 -c 2 -qatten 1 -p 64 -n 64 -rep 5")
    subprocess.run(cmd, shell=True, capture_output=True, timeout=duration_s + 10)


def workload_gpu_mnn(duration_s):
    """GPU MNN inference."""
    cmd = ("/home/hyunho.son/install_files/MNN/build/llm_bench "
           "-m /home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/mnn_models/M9_w8a8_ch_direct/config.json "
           "-a opencl -t 4 -c 2 -qatten 1 -p 64 -n 64 -rep 5")
    subprocess.run(cmd, shell=True, capture_output=True, timeout=duration_s + 10)


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    workloads = [
        ("idle", workload_idle, 20),
        ("CPU-only (MNN W8A8)", workload_cpu_mnn, 30),
        ("GPU-only (MNN OpenCL)", workload_gpu_mnn, 30),
        ("NPU-only (RKLLM W8A8)", workload_npu_only, 30),
    ]

    all_results = []
    for name, fn, dur in workloads:
        try:
            r = run_workload(name, fn, dur)
            all_results.append(r)
        except Exception as e:
            print(f"[ERR] {name}: {e}")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "thermal_traces.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved {out_path}")

    # Summary table
    print(f"\n{'='*70}\nThermal Comparison (Δ from idle baseline)\n{'='*70}")
    if all_results and all_results[0]["workload"] == "idle":
        idle_means = {n: s["mean"] for n, s in all_results[0]["summary"].items()}
        print(f"{'Workload':<25s} {'NPU':>8s} {'CPU':>8s} {'GPU':>8s} {'SoC':>8s}")
        for r in all_results[1:]:
            s = r["summary"]
            npu_d = s.get("npu", {}).get("mean", 0) - idle_means.get("npu", 0)
            cpu_d = s.get("bigcore0", {}).get("mean", 0) - idle_means.get("bigcore0", 0)
            gpu_d = s.get("gpu", {}).get("mean", 0) - idle_means.get("gpu", 0)
            soc_d = s.get("soc", {}).get("mean", 0) - idle_means.get("soc", 0)
            print(f"{r['workload']:<25s} {npu_d:>7.1f}°C {cpu_d:>7.1f}°C {gpu_d:>7.1f}°C {soc_d:>7.1f}°C")


if __name__ == "__main__":
    main()
