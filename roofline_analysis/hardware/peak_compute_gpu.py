"""
GPU Peak Compute Benchmark
Mali-G610 MC4 (OpenCL) FP16/FP32 피크 GFLOPS 측정.

Usage:
    python hardware/peak_compute_gpu.py
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

# OpenCL kernel paths (reuse from project)
OPENCL_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "opencl")

# Add parent project to path for opencl.util
sys.path.insert(0, os.path.dirname(PROJECT_DIR))


def load_config(config_path=None):
    path = config_path or DEFAULT_CONFIG
    with open(path) as f:
        return yaml.safe_load(f)


def get_opencl_context():
    """Get OpenCL context for Mali-G610 GPU."""
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


FP16_KERNEL = """
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

__kernel void matmul_fp16(
    int M, int N, int K,
    __global const half* restrict A, int lda,
    __global const half* restrict B, int ldb,
    __global half* restrict C, int ldc)
{
    const int wg_row = get_group_id(0);
    const int wg_col = get_group_id(1);
    const int lid_row = get_local_id(0);
    const int lid_col = get_local_id(1);
    const int global_row = wg_row * TILE_M + lid_row;
    const int global_col = wg_col * TILE_N + lid_col;

    __local half As[TILE_M * TILE_K];
    __local half Bs[TILE_K * TILE_N];

    float acc = 0.0f;
    const int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        const int k_base = t * TILE_K;
        int a_row = global_row, a_col = k_base + lid_col;
        As[lid_row * TILE_K + lid_col] = (a_row < M && a_col < K) ?
            A[a_row * lda + a_col] : (half)0.0h;

        int b_row = k_base + lid_row, b_col = global_col;
        Bs[lid_row * TILE_N + lid_col] = (b_row < K && b_col < N) ?
            B[b_row * ldb + b_col] : (half)0.0h;

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < TILE_K; ++kk) {
            acc += convert_float(As[lid_row * TILE_K + kk]) *
                   convert_float(Bs[kk * TILE_N + lid_col]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (global_row < M && global_col < N)
        C[global_row * ldc + global_col] = convert_half(acc);
}
"""

FP32_KERNEL = """
#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

__kernel void matmul_fp32(
    int M, int N, int K,
    __global const float* restrict A, int lda,
    __global const float* restrict B, int ldb,
    __global float* restrict C, int ldc)
{
    const int wg_row = get_group_id(0);
    const int wg_col = get_group_id(1);
    const int lid_row = get_local_id(0);
    const int lid_col = get_local_id(1);
    const int global_row = wg_row * TILE_M + lid_row;
    const int global_col = wg_col * TILE_N + lid_col;

    __local float As[TILE_M * TILE_K];
    __local float Bs[TILE_K * TILE_N];

    float acc = 0.0f;
    const int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        const int k_base = t * TILE_K;
        int a_row = global_row, a_col = k_base + lid_col;
        As[lid_row * TILE_K + lid_col] = (a_row < M && a_col < K) ?
            A[a_row * lda + a_col] : 0.0f;

        int b_row = k_base + lid_row, b_col = global_col;
        Bs[lid_row * TILE_N + lid_col] = (b_row < K && b_col < N) ?
            B[b_row * ldb + b_col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < TILE_K; ++kk) {
            acc += As[lid_row * TILE_K + kk] * Bs[kk * TILE_N + lid_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (global_row < M && global_col < N)
        C[global_row * ldc + global_col] = acc;
}
"""


def bench_gpu_matmul(ctx, queue, M, N, K, dtype="float16", reps=10, warmup=10):
    """Measure GPU matmul latency in ms using OpenCL profiling events."""
    import pyopencl as cl

    if dtype == "float16":
        np_dtype = np.float16
        kernel_src = FP16_KERNEL
        kernel_name = "matmul_fp16"
    else:
        np_dtype = np.float32
        kernel_src = FP32_KERNEL
        kernel_name = "matmul_fp32"

    prog = cl.Program(ctx, kernel_src).build(
        options=["-cl-fast-relaxed-math", "-cl-mad-enable"]
    )
    kernel = getattr(prog, kernel_name)

    A = np.random.rand(M, K).astype(np_dtype)
    B = np.random.rand(K, N).astype(np_dtype)
    C = np.zeros((M, N), dtype=np_dtype)

    mf = cl.mem_flags
    dA = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    dB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    dC = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

    TILE = 16
    global_M = ((M + TILE - 1) // TILE) * TILE
    global_N = ((N + TILE - 1) // TILE) * TILE
    global_size = (global_M, global_N)
    local_size = (TILE, TILE)

    # Warmup
    for _ in range(warmup):
        kernel(queue, global_size, local_size,
               np.int32(M), np.int32(N), np.int32(K),
               dA, np.int32(K), dB, np.int32(N), dC, np.int32(N))
    queue.finish()

    # Measure with profiling events
    times = []
    for _ in range(reps):
        evt = kernel(queue, global_size, local_size,
                     np.int32(M), np.int32(N), np.int32(K),
                     dA, np.int32(K), dB, np.int32(N), dC, np.int32(N))
        evt.wait()
        ms = (evt.profile.end - evt.profile.start) / 1e6
        times.append(ms)

    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description="GPU Peak Compute Benchmark")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pc = cfg["peak_compute"]
    sizes = pc["sizes"]
    reps = pc["reps"]
    warmup = pc["warmup"]
    gpu_dtypes = pc["dtypes"]["gpu"]

    output_dir = args.output_dir or os.path.join(PROJECT_DIR, "results", "raw")
    os.makedirs(output_dir, exist_ok=True)

    try:
        import pyopencl as cl
    except ImportError:
        print("[ERROR] pyopencl not installed: pip install pyopencl")
        sys.exit(1)

    ctx, queue = get_opencl_context()
    print(f"GPU Device: {ctx.devices[0].name}")

    results = []

    for dtype_name in gpu_dtypes:
        print(f"\n{'='*60}")
        print(f"GPU Peak Compute — {dtype_name} (Mali-G610, OpenCL)")
        print(f"Warmup: {warmup} | Measure: {reps} (median)")
        print(f"{'='*60}")

        for S in sizes:
            try:
                ms = bench_gpu_matmul(ctx, queue, S, S, S, dtype_name, reps, warmup)
                ops = 2.0 * S * S * S
                gflops = ops / (ms / 1000) / 1e9
            except Exception as e:
                print(f"  {S:5d} x {S:5d} : FAILED ({e})")
                continue

            results.append({
                "backend": "gpu",
                "dtype": dtype_name,
                "size": S,
                "latency_ms": round(ms, 4),
                "gflops": round(gflops, 3),
            })
            print(f"  {S:5d} x {S:5d} : {ms:10.3f} ms  ({gflops:8.3f} GFLOPS)")

    # Peak summary
    print(f"\n{'='*60}")
    print("Peak GFLOPS Summary:")
    for dtype_name in gpu_dtypes:
        dtype_results = [r for r in results if r["dtype"] == dtype_name]
        if dtype_results:
            peak = max(dtype_results, key=lambda r: r["gflops"])
            print(f"  {dtype_name:10s}: {peak['gflops']:8.3f} GFLOPS (at size {peak['size']})")
    print(f"{'='*60}")

    # Save CSV
    csv_path = os.path.join(output_dir, "peak_compute_gpu.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "dtype", "size", "latency_ms", "gflops"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
