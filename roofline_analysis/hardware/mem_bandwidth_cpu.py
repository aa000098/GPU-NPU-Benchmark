"""
CPU Memory Bandwidth Benchmark (STREAM-like)
Copy, Scale, Add, Triad 연산으로 Cortex-A76 LPDDR4x 대역폭 측정.

Usage:
    python hardware/mem_bandwidth_cpu.py
"""
import numpy as np
import time
import os
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


def stream_copy(A, B, reps):
    """B[:] = A[:] → 2 * N * sizeof reads/writes."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        np.copyto(B, A)
        times.append(time.perf_counter() - t0)
    return times


def stream_scale(A, B, scalar, reps):
    """B[:] = scalar * A[:] → 2 * N * sizeof."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        np.multiply(A, scalar, out=B)
        times.append(time.perf_counter() - t0)
    return times


def stream_add(A, B, C, reps):
    """C[:] = A[:] + B[:] → 3 * N * sizeof."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        np.add(A, B, out=C)
        times.append(time.perf_counter() - t0)
    return times


def stream_triad(A, B, C, scalar, reps):
    """C[:] = A[:] + scalar * B[:] → 3 * N * sizeof."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        # Use in-place to avoid allocation
        np.multiply(B, scalar, out=C)
        np.add(A, C, out=C)
        times.append(time.perf_counter() - t0)
    return times


def compute_bandwidth(bytes_transferred, time_s):
    """Compute bandwidth in GB/s."""
    return bytes_transferred / time_s / 1e9


def main():
    parser = argparse.ArgumentParser(description="CPU Memory Bandwidth (STREAM)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mb = cfg["mem_bandwidth"]
    sizes_mb = mb["sizes_mb"]
    reps = mb["reps"]
    warmup = mb["warmup"]

    output_dir = args.output_dir or os.path.join(PROJECT_DIR, "results", "raw")
    os.makedirs(output_dir, exist_ok=True)

    dtype = np.float64  # STREAM canonical uses double
    elem_bytes = dtype().itemsize

    results = []

    print(f"{'='*70}")
    print(f"CPU STREAM Bandwidth — Cortex-A76 x4 (LPDDR4x)")
    print(f"Warmup: {warmup} | Measure: {reps} (best)")
    print(f"{'='*70}")

    # Operations: name, bytes_factor (multiplier of N * sizeof)
    ops_info = {
        "copy":  2,  # read A + write B
        "scale": 2,  # read A + write B
        "add":   3,  # read A + read B + write C
        "triad": 3,  # read A + read B + write C
    }

    for size_mb in sizes_mb:
        n_elements = int(size_mb * 1024 * 1024 / elem_bytes)
        actual_mb = n_elements * elem_bytes / (1024 * 1024)

        A = np.random.rand(n_elements).astype(dtype)
        B = np.random.rand(n_elements).astype(dtype)
        C = np.zeros(n_elements, dtype=dtype)
        scalar = 3.0

        print(f"\n  Array size: {actual_mb:.1f} MB ({n_elements:,} elements)")

        # Warmup all
        for _ in range(warmup):
            np.copyto(B, A)
            np.multiply(A, scalar, out=B)
            np.add(A, B, out=C)

        # Copy
        times = stream_copy(A, B, reps)
        best_time = min(times)
        bw = compute_bandwidth(ops_info["copy"] * n_elements * elem_bytes, best_time)
        results.append({"backend": "cpu", "operation": "copy", "size_mb": actual_mb,
                         "best_time_ms": round(best_time * 1000, 4), "bandwidth_gb_s": round(bw, 3)})
        print(f"    Copy:  {bw:8.3f} GB/s")

        # Scale
        times = stream_scale(A, B, scalar, reps)
        best_time = min(times)
        bw = compute_bandwidth(ops_info["scale"] * n_elements * elem_bytes, best_time)
        results.append({"backend": "cpu", "operation": "scale", "size_mb": actual_mb,
                         "best_time_ms": round(best_time * 1000, 4), "bandwidth_gb_s": round(bw, 3)})
        print(f"    Scale: {bw:8.3f} GB/s")

        # Add
        times = stream_add(A, B, C, reps)
        best_time = min(times)
        bw = compute_bandwidth(ops_info["add"] * n_elements * elem_bytes, best_time)
        results.append({"backend": "cpu", "operation": "add", "size_mb": actual_mb,
                         "best_time_ms": round(best_time * 1000, 4), "bandwidth_gb_s": round(bw, 3)})
        print(f"    Add:   {bw:8.3f} GB/s")

        # Triad
        times = stream_triad(A, B, C, scalar, reps)
        best_time = min(times)
        bw = compute_bandwidth(ops_info["triad"] * n_elements * elem_bytes, best_time)
        results.append({"backend": "cpu", "operation": "triad", "size_mb": actual_mb,
                         "best_time_ms": round(best_time * 1000, 4), "bandwidth_gb_s": round(bw, 3)})
        print(f"    Triad: {bw:8.3f} GB/s")

    # Peak summary
    print(f"\n{'='*70}")
    print("Peak Bandwidth Summary:")
    for op in ["copy", "scale", "add", "triad"]:
        op_results = [r for r in results if r["operation"] == op]
        if op_results:
            peak = max(op_results, key=lambda r: r["bandwidth_gb_s"])
            print(f"  {op:6s}: {peak['bandwidth_gb_s']:8.3f} GB/s (at {peak['size_mb']:.0f} MB)")
    print(f"{'='*70}")

    # Save CSV
    csv_path = os.path.join(output_dir, "mem_bw_cpu.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "operation", "size_mb",
                                                "best_time_ms", "bandwidth_gb_s"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
