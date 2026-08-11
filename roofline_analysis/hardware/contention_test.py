"""
CPU+GPU Memory Bandwidth Contention Test
CPU STREAM + GPU OpenCL 동시 실행 시 LPDDR4x 대역폭 경합 측정.

Usage:
    python hardware/contention_test.py
"""
import numpy as np
import time
import os
import sys
import csv
import argparse
import yaml
import multiprocessing as mp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, "config", "roofline_config.yaml")

os.environ['OPENBLAS_NUM_THREADS'] = '4'


def load_config(config_path=None):
    path = config_path or DEFAULT_CONFIG
    with open(path) as f:
        return yaml.safe_load(f)


def cpu_stream_worker(size_mb, reps, result_queue):
    """Run CPU STREAM triad and report bandwidth."""
    n = int(size_mb * 1024 * 1024 / 8)  # float64
    A = np.random.rand(n)
    B = np.random.rand(n)
    C = np.zeros(n)
    scalar = 3.0

    # Warmup
    for _ in range(3):
        np.multiply(B, scalar, out=C)
        np.add(A, C, out=C)

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        np.multiply(B, scalar, out=C)
        np.add(A, C, out=C)
        times.append(time.perf_counter() - t0)

    best = min(times)
    bw = 3 * n * 8 / best / 1e9  # triad: 3 arrays
    result_queue.put(("cpu", bw))


def gpu_copy_worker(size_mb, reps, result_queue):
    """Run GPU buffer copy and report bandwidth."""
    try:
        import pyopencl as cl
    except ImportError:
        result_queue.put(("gpu", 0))
        return

    # Get context
    platforms = cl.get_platforms()
    platform = None
    for p in platforms:
        if "ARM" in p.name:
            platform = p
            break
    if platform is None:
        platform = platforms[0]

    devices = platform.get_devices(device_type=cl.device_type.GPU)
    if not devices:
        devices = platform.get_devices(device_type=cl.device_type.ALL)

    ctx = cl.Context([devices[0]])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    KERNEL = """
    __kernel void copy_k(__global const float* src, __global float* dst, int n) {
        int gid = get_global_id(0);
        if (gid < n) dst[gid] = src[gid];
    }
    """
    prog = cl.Program(ctx, KERNEL).build()

    n_floats = int(size_mb * 1024 * 1024 / 4)
    data = np.random.rand(n_floats).astype(np.float32)
    mf = cl.mem_flags
    buf_src = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
    buf_dst = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)

    # Warmup
    for _ in range(3):
        prog.copy_k(queue, (n_floats,), None, buf_src, buf_dst, np.int32(n_floats))
    queue.finish()

    times = []
    for _ in range(reps):
        evt = prog.copy_k(queue, (n_floats,), None, buf_src, buf_dst, np.int32(n_floats))
        evt.wait()
        ms = (evt.profile.end - evt.profile.start) / 1e6
        times.append(ms)

    best_ms = min(times)
    bw = 2 * n_floats * 4 / (best_ms / 1000) / 1e9
    result_queue.put(("gpu", bw))


def main():
    parser = argparse.ArgumentParser(description="CPU+GPU Contention Test")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    test_sizes = [16, 64, 128]  # MB
    reps = 10

    output_dir = args.output_dir or os.path.join(PROJECT_DIR, "results", "raw")
    os.makedirs(output_dir, exist_ok=True)

    results = []

    print(f"{'='*70}")
    print("CPU + GPU Memory Bandwidth Contention Test")
    print(f"{'='*70}")

    for size_mb in test_sizes:
        print(f"\n  Buffer size: {size_mb} MB")

        # 1. Solo CPU
        q = mp.Queue()
        p = mp.Process(target=cpu_stream_worker, args=(size_mb, reps, q))
        p.start()
        p.join()
        _, cpu_solo_bw = q.get()
        print(f"    CPU solo:      {cpu_solo_bw:.2f} GB/s")

        # 2. Solo GPU
        q = mp.Queue()
        p = mp.Process(target=gpu_copy_worker, args=(size_mb, reps, q))
        p.start()
        p.join()
        _, gpu_solo_bw = q.get()
        print(f"    GPU solo:      {gpu_solo_bw:.2f} GB/s")

        # 3. Simultaneous CPU + GPU
        q_cpu = mp.Queue()
        q_gpu = mp.Queue()
        p_cpu = mp.Process(target=cpu_stream_worker, args=(size_mb, reps, q_cpu))
        p_gpu = mp.Process(target=gpu_copy_worker, args=(size_mb, reps, q_gpu))

        p_cpu.start()
        p_gpu.start()
        p_cpu.join()
        p_gpu.join()

        _, cpu_contended_bw = q_cpu.get()
        _, gpu_contended_bw = q_gpu.get()

        cpu_contention = cpu_solo_bw / cpu_contended_bw if cpu_contended_bw > 0 else 0
        gpu_contention = gpu_solo_bw / gpu_contended_bw if gpu_contended_bw > 0 else 0

        print(f"    CPU contended: {cpu_contended_bw:.2f} GB/s (factor: {cpu_contention:.2f}x)")
        print(f"    GPU contended: {gpu_contended_bw:.2f} GB/s (factor: {gpu_contention:.2f}x)")

        results.append({
            "size_mb": size_mb,
            "cpu_solo_gb_s": round(cpu_solo_bw, 3),
            "gpu_solo_gb_s": round(gpu_solo_bw, 3),
            "cpu_contended_gb_s": round(cpu_contended_bw, 3),
            "gpu_contended_gb_s": round(gpu_contended_bw, 3),
            "cpu_contention_factor": round(cpu_contention, 3),
            "gpu_contention_factor": round(gpu_contention, 3),
            "total_solo_gb_s": round(cpu_solo_bw + gpu_solo_bw, 3),
            "total_contended_gb_s": round(cpu_contended_bw + gpu_contended_bw, 3),
        })

    # Summary
    print(f"\n{'='*70}")
    print("Contention Summary:")
    print(f"  {'Size':>6s} {'CPU Solo':>10s} {'GPU Solo':>10s} "
          f"{'CPU Cont':>10s} {'GPU Cont':>10s} {'CPU Fx':>8s} {'GPU Fx':>8s}")
    for r in results:
        print(f"  {r['size_mb']:>4d}MB {r['cpu_solo_gb_s']:>10.2f} {r['gpu_solo_gb_s']:>10.2f} "
              f"{r['cpu_contended_gb_s']:>10.2f} {r['gpu_contended_gb_s']:>10.2f} "
              f"{r['cpu_contention_factor']:>8.2f} {r['gpu_contention_factor']:>8.2f}")
    print(f"{'='*70}")

    # Save
    csv_path = os.path.join(output_dir, "contention_test.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
