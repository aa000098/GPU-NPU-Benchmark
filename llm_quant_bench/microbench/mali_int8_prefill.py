"""Test Mali IDP int8 GEMM at prefill batch sizes (M=64, 256, 512, 1024).

Prefill is compute-bound at high M, so IDP int8 should win significantly.
Compare against MNN's actual GPU prefill measurement.
"""
import os, sys, time
import numpy as np
import pyopencl as cl

os.environ['PYOPENCL_CTX'] = '0'

plats = cl.get_platforms()
mali_dev = None
for p in plats:
    for d in p.get_devices():
        if 'Mali' in d.name:
            mali_dev = d; break
    if mali_dev: break

ctx = cl.Context([mali_dev])
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

# Tile-based GEMM with IDP. Each work item handles 1 output element of (M × N) result
KERNEL_GEMM_IDP = """
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

__kernel void gemm_idp(
    __global const uint4* W,   // [N, K/16] uint4
    __global const uint4* X,   // [M, K/16] uint4
    __global const float* w_scale,
    float x_scale,
    __global float* Y,         // [M, N]
    int M, int N, int K_div16)
{
    int n = get_global_id(0);
    int m = get_global_id(1);
    if (n >= N || m >= M) return;
    int acc = 0;
    for (int k = 0; k < K_div16; k++) {
        uint4 w16 = W[n * K_div16 + k];
        uint4 x16 = X[m * K_div16 + k];
        acc = dot_acc_sat_4x8packed_ss_int(w16.s0, x16.s0, acc);
        acc = dot_acc_sat_4x8packed_ss_int(w16.s1, x16.s1, acc);
        acc = dot_acc_sat_4x8packed_ss_int(w16.s2, x16.s2, acc);
        acc = dot_acc_sat_4x8packed_ss_int(w16.s3, x16.s3, acc);
    }
    Y[m * N + n] = (float)acc * w_scale[n] * x_scale;
}
"""

# fp16 dequant version for comparison
KERNEL_GEMM_FP16 = """
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void gemm_fp16(
    __global const half8* W,   // [N, K/8] half8 (already dequantized)
    __global const half8* X,   // [M, K/8] half8
    __global float* Y,
    int M, int N, int K_div8)
{
    int n = get_global_id(0);
    int m = get_global_id(1);
    if (n >= N || m >= M) return;
    half16 acc = 0;
    for (int k = 0; k < K_div8; k++) {
        half8 w = W[n * K_div8 + k];
        half8 x = X[m * K_div8 + k];
        acc.s01234567 += w * x;
    }
    half sum = acc.s0 + acc.s1 + acc.s2 + acc.s3 + acc.s4 + acc.s5 + acc.s6 + acc.s7;
    Y[m * N + n] = (float)sum;
}
"""


def bench_kernel(prog, kernel_name, args, gws, lws=None, iters=10, warmup=3):
    knl = getattr(prog, kernel_name)
    for _ in range(warmup):
        knl(queue, gws, lws, *args)
    queue.finish()
    t0 = time.perf_counter()
    for _ in range(iters):
        knl(queue, gws, lws, *args)
    queue.finish()
    return (time.perf_counter() - t0) * 1000.0 / iters


def test_prefill(M, K, N, name, mnn_prefill_tok_s_ref=None):
    print(f"\n=== {name}: M={M} × K={K} → N={N} ===")
    mnn_ms = M * 1000.0 / mnn_prefill_tok_s_ref if mnn_prefill_tok_s_ref else None
    print(f"  MNN ref ({mnn_prefill_tok_s_ref} tok/s): {mnn_ms:.2f} ms" if mnn_ms else "")

    np.random.seed(0)
    W_int8 = np.random.randint(-127, 127, (N, K), dtype=np.int8)
    X_int8 = np.random.randint(-127, 127, (M, K), dtype=np.int8)
    X_fp16 = X_int8.astype(np.float16) * 0.01
    W_fp16 = W_int8.astype(np.float16) * 0.01
    w_scale = np.full(N, 0.01, dtype=np.float32)
    x_scale = np.float32(0.01)

    W_packed = W_int8.reshape(-1).view(np.uint32)
    X_packed = X_int8.reshape(-1).view(np.uint32)

    Wp_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=W_packed)
    Xp_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=X_packed)
    Wf_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=W_fp16.flatten())
    Xf_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=X_fp16.flatten())
    s_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w_scale)
    y_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=M * N * 4)

    prog_idp = cl.Program(ctx, KERNEL_GEMM_IDP).build()
    args_idp = [Wp_buf, Xp_buf, s_buf, x_scale, y_buf, np.int32(M), np.int32(N), np.int32(K // 16)]
    for lws_n in [4, 8, 16]:
        for lws_m in [4, 8, 16]:
            if lws_n * lws_m > 256: continue
            try:
                # Round up gws to multiples of lws
                gws_n = ((N + lws_n - 1) // lws_n) * lws_n
                gws_m = ((M + lws_m - 1) // lws_m) * lws_m
                ms = bench_kernel(prog_idp, 'gemm_idp', args_idp, (gws_n, gws_m), (lws_n, lws_m))
                tok_s = M * 1000.0 / ms
                print(f"  IDP  lws=({lws_n},{lws_m}): {ms:8.3f} ms  →  {tok_s:6.0f} prefill tok/s")
            except Exception as e:
                pass

    prog_fp = cl.Program(ctx, KERNEL_GEMM_FP16).build()
    args_fp = [Wf_buf, Xf_buf, y_buf, np.int32(M), np.int32(N), np.int32(K // 8)]
    for lws_n in [4, 8, 16]:
        for lws_m in [4, 8, 16]:
            if lws_n * lws_m > 256: continue
            try:
                gws_n = ((N + lws_n - 1) // lws_n) * lws_n
                gws_m = ((M + lws_m - 1) // lws_m) * lws_m
                ms = bench_kernel(prog_fp, 'gemm_fp16', args_fp, (gws_n, gws_m), (lws_n, lws_m))
                tok_s = M * 1000.0 / ms
                print(f"  fp16 lws=({lws_n},{lws_m}): {ms:8.3f} ms  →  {tok_s:6.0f} prefill tok/s")
            except Exception as e:
                pass


# Test FFN gate prefill at multiple M
test_prefill(64, 2048, 8192, "FFN gate prefill M=64", mnn_prefill_tok_s_ref=126)
test_prefill(256, 2048, 8192, "FFN gate prefill M=256")
test_prefill(512, 2048, 8192, "FFN gate prefill M=512", mnn_prefill_tok_s_ref=175)
test_prefill(1024, 2048, 8192, "FFN gate prefill M=1024", mnn_prefill_tok_s_ref=170)
