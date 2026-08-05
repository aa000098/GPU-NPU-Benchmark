"""
CPU Peak Compute Benchmark
Cortex-A76 x4 (OpenBLAS + NEON)로 FP32/FP16 피크 GFLOPS 측정.

Usage:
    python hardware/peak_compute_cpu.py
    python hardware/peak_compute_cpu.py --config config/roofline_config.yaml
"""
import numpy as np
import time
import os
import sys
import csv
import argparse
import yaml

os.environ['OPENBLAS_NUM_THREADS'] = '4'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, "config", "roofline_config.yaml")


def load_config(config_path=None):
    path = config_path or DEFAULT_CONFIG
    with open(path) as f:
        return yaml.safe_load(f)


def bench_matmul(M, N, K, dtype=np.float32, reps=10, warmup=10):
    """Measure matmul latency in ms (median of reps)."""
    A = np.random.randn(M, K).astype(dtype)
    B = np.random.randn(K, N).astype(dtype)

    for _ in range(warmup):
        np.matmul(A, B)

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        np.matmul(A, B)
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.median(times))


def compute_gflops(M, N, K, ms):
    """Compute GFLOPS from matmul dimensions and latency."""
    ops = 2.0 * M * N * K
    return ops / (ms / 1000) / 1e9


def main():
    parser = argparse.ArgumentParser(description="CPU Peak Compute Benchmark")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pc = cfg["peak_compute"]
    sizes = pc["sizes"]
    reps = pc["reps"]
    warmup = pc["warmup"]
    cpu_dtypes = pc["dtypes"]["cpu"]

    output_dir = args.output_dir or os.path.join(PROJECT_DIR, "results", "raw")
    os.makedirs(output_dir, exist_ok=True)

    dtype_map = {
        "float32": np.float32,
        "float16": np.float16,
        "int8": np.int8,
    }

    results = []

    for dtype_name in cpu_dtypes:
        np_dtype = dtype_map.get(dtype_name)
        if np_dtype is None:
            print(f"[WARN] Unsupported dtype: {dtype_name}")
            continue

        print(f"\n{'='*60}")
        print(f"CPU Peak Compute — {dtype_name} (OpenBLAS 4T, NEON)")
        print(f"Warmup: {warmup} | Measure: {reps} (median)")
        print(f"{'='*60}")

        for S in sizes:
            # int8 matmul not supported by numpy, use float32 simulation
            if dtype_name == "int8":
                A = np.random.randint(-128, 127, (S, S)).astype(np.float32)
                B = np.random.randint(-128, 127, (S, S)).astype(np.float32)
                for _ in range(warmup):
                    np.matmul(A, B)
                times = []
                for _ in range(reps):
                    t0 = time.perf_counter()
                    np.matmul(A, B)
                    times.append((time.perf_counter() - t0) * 1000)
                ms = float(np.median(times))
            else:
                ms = bench_matmul(S, S, S, np_dtype, reps, warmup)

            gflops = compute_gflops(S, S, S, ms)
            results.append({
                "backend": "cpu",
                "dtype": dtype_name,
                "size": S,
                "latency_ms": round(ms, 4),
                "gflops": round(gflops, 3),
            })
            print(f"  {S:5d} x {S:5d} : {ms:10.3f} ms  ({gflops:8.3f} GFLOPS)")

    # Find peak per dtype
    print(f"\n{'='*60}")
    print("Peak GFLOPS Summary:")
    for dtype_name in cpu_dtypes:
        dtype_results = [r for r in results if r["dtype"] == dtype_name]
        if dtype_results:
            peak = max(dtype_results, key=lambda r: r["gflops"])
            print(f"  {dtype_name:10s}: {peak['gflops']:8.3f} GFLOPS (at size {peak['size']})")
    print(f"{'='*60}")

    # Save CSV
    csv_path = os.path.join(output_dir, "peak_compute_cpu.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "dtype", "size", "latency_ms", "gflops"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
