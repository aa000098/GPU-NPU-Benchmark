"""
GPU Memory Bandwidth Benchmark
Mali-G610 MC4 (OpenCL) 버퍼 복사/읽기/쓰기 대역폭 측정.

Usage:
    python hardware/mem_bandwidth_gpu.py
"""
import numpy as np
import time
import os
import sys
import csv
import argparse
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, "config", "roofline_config.yaml")


def load_config(config_path=None):
    path = config_path or DEFAULT_CONFIG
    with open(path) as f:
        return yaml.safe_load(f)


def get_opencl_context():
    import pyopencl as cl
    platforms = cl.get_platforms()
    platform = None
    for p in platforms:
        if "ARM" in p.name:
            platform = p
            break
    if platform is None:
        platform = platforms[0]

    devices = platform.get_devices(device_type=cl.device_type.GPU)
    if len(devices) == 0:
        devices = platform.get_devices(device_type=cl.device_type.ALL)

    ctx = cl.Context([devices[0]])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    return ctx, queue


# Simple copy kernel for measuring effective memory bandwidth from compute kernel
COPY_KERNEL = """
__kernel void copy_kernel(__global const float* src, __global float* dst, int n) {
    int gid = get_global_id(0);
    if (gid < n) dst[gid] = src[gid];
}
"""


def main():
    parser = argparse.ArgumentParser(description="GPU Memory Bandwidth Benchmark")
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

    try:
        import pyopencl as cl
    except ImportError:
        print("[ERROR] pyopencl not installed")
        sys.exit(1)

    ctx, queue = get_opencl_context()
    print(f"GPU Device: {ctx.devices[0].name}")

    prog = cl.Program(ctx, COPY_KERNEL).build()

    results = []

    print(f"\n{'='*70}")
    print(f"GPU Memory Bandwidth — Mali-G610 MC4 (OpenCL)")
    print(f"Warmup: {warmup} | Measure: {reps} (best)")
    print(f"{'='*70}")

    for size_mb in sizes_mb:
        n_bytes = int(size_mb * 1024 * 1024)
        n_floats = n_bytes // 4
        actual_mb = n_floats * 4 / (1024 * 1024)

        data = np.random.rand(n_floats).astype(np.float32)

        mf = cl.mem_flags
        buf_src = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
        buf_dst = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)
        host_dst = np.empty_like(data)

        print(f"\n  Buffer size: {actual_mb:.1f} MB")

        # 1. Device-to-Device copy (via kernel)
        global_size = (n_floats,)
        local_size = None

        for _ in range(warmup):
            prog.copy_kernel(queue, global_size, local_size, buf_src, buf_dst, np.int32(n_floats))
        queue.finish()

        times = []
        for _ in range(reps):
            evt = prog.copy_kernel(queue, global_size, local_size, buf_src, buf_dst, np.int32(n_floats))
            evt.wait()
            ms = (evt.profile.end - evt.profile.start) / 1e6
            times.append(ms)

        best_ms = min(times)
        bw = 2 * n_bytes / (best_ms / 1000) / 1e9  # read + write
        results.append({"backend": "gpu", "operation": "d2d_copy", "size_mb": actual_mb,
                         "best_time_ms": round(best_ms, 4), "bandwidth_gb_s": round(bw, 3)})
        print(f"    D2D Copy:   {bw:8.3f} GB/s")

        # 2. Host-to-Device write
        for _ in range(warmup):
            cl.enqueue_copy(queue, buf_dst, data)
        queue.finish()

        times = []
        for _ in range(reps):
            evt = cl.enqueue_copy(queue, buf_dst, data)
            evt.wait()
            ms = (evt.profile.end - evt.profile.start) / 1e6
            times.append(ms)

        best_ms = min(times)
        bw = n_bytes / (best_ms / 1000) / 1e9
        results.append({"backend": "gpu", "operation": "h2d_write", "size_mb": actual_mb,
                         "best_time_ms": round(best_ms, 4), "bandwidth_gb_s": round(bw, 3)})
        print(f"    H2D Write:  {bw:8.3f} GB/s")

        # 3. Device-to-Host read
        for _ in range(warmup):
            cl.enqueue_copy(queue, host_dst, buf_src)
        queue.finish()

        times = []
        for _ in range(reps):
            evt = cl.enqueue_copy(queue, host_dst, buf_src)
            evt.wait()
            ms = (evt.profile.end - evt.profile.start) / 1e6
            times.append(ms)

        best_ms = min(times)
        bw = n_bytes / (best_ms / 1000) / 1e9
        results.append({"backend": "gpu", "operation": "d2h_read", "size_mb": actual_mb,
                         "best_time_ms": round(best_ms, 4), "bandwidth_gb_s": round(bw, 3)})
        print(f"    D2H Read:   {bw:8.3f} GB/s")

    # Peak summary
    print(f"\n{'='*70}")
    print("Peak Bandwidth Summary:")
    for op in ["d2d_copy", "h2d_write", "d2h_read"]:
        op_results = [r for r in results if r["operation"] == op]
        if op_results:
            peak = max(op_results, key=lambda r: r["bandwidth_gb_s"])
            print(f"  {op:12s}: {peak['bandwidth_gb_s']:8.3f} GB/s (at {peak['size_mb']:.0f} MB)")
    print(f"{'='*70}")

    # Save CSV
    csv_path = os.path.join(output_dir, "mem_bw_gpu.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "operation", "size_mb",
                                                "best_time_ms", "bandwidth_gb_s"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
