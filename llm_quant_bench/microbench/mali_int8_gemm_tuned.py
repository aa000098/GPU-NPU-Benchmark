"""Tuned Mali-G610 IDP int8 GEMM kernel with:
- char16 vectorized loads (16 int8 per load)
- Local memory for input vector (reused across rows)
- Work group cooperation
- Multiple output rows per work item

Target: beat MNN's tuned dequant fp16 kernel (~13.47ms LM head).
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

print(f"Device: {mali_dev.name}")
print(f"Max work group: {mali_dev.max_work_group_size}")

# === Tuned IDP kernel ===
# 1) Each work item handles 1 output row
# 2) Reads 16 int8 at a time via char16
# 3) Uses dot_acc_sat_4x8packed_ss_int 4 times per char16 (16 int8 = 4 × 4 int8)

KERNEL_IDP_TUNED = """
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

__kernel void gemv_idp_tuned(
    __global const uint4* W,     // int8 [N, K/16] viewed as uint4 (each uint = 4 int8)
    __global const uint4* x,     // int8 [K/16] as uint4
    __global const float* w_scale,
    float x_scale,
    __global float* y,
    int N, int K_div16)
{
    int n = get_global_id(0);
    if (n >= N) return;
    int acc = 0;
    for (int k = 0; k < K_div16; k++) {
        uint4 w16 = W[n * K_div16 + k];
        uint4 x16 = x[k];
        // 4 IDP ops, each handles 4 int8
        acc = dot_acc_sat_4x8packed_ss_int(w16.s0, x16.s0, acc);
        acc = dot_acc_sat_4x8packed_ss_int(w16.s1, x16.s1, acc);
        acc = dot_acc_sat_4x8packed_ss_int(w16.s2, x16.s2, acc);
        acc = dot_acc_sat_4x8packed_ss_int(w16.s3, x16.s3, acc);
    }
    y[n] = (float)acc * w_scale[n] * x_scale;
}
"""

# Multi-row per work item version (each WI handles 4 output rows for better latency hiding)
KERNEL_IDP_MULTI = """
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

__kernel void gemv_idp_multi(
    __global const uint4* W,
    __global const uint4* x,
    __global const float* w_scale,
    float x_scale,
    __global float* y,
    int N, int K_div16)
{
    int n_base = get_global_id(0) * 4;
    int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    int x_off = 0;
    for (int k = 0; k < K_div16; k++) {
        uint4 x16 = x[k];
        if (n_base + 0 < N) {
            uint4 w16 = W[(n_base + 0) * K_div16 + k];
            acc0 = dot_acc_sat_4x8packed_ss_int(w16.s0, x16.s0, acc0);
            acc0 = dot_acc_sat_4x8packed_ss_int(w16.s1, x16.s1, acc0);
            acc0 = dot_acc_sat_4x8packed_ss_int(w16.s2, x16.s2, acc0);
            acc0 = dot_acc_sat_4x8packed_ss_int(w16.s3, x16.s3, acc0);
        }
        if (n_base + 1 < N) {
            uint4 w16 = W[(n_base + 1) * K_div16 + k];
            acc1 = dot_acc_sat_4x8packed_ss_int(w16.s0, x16.s0, acc1);
            acc1 = dot_acc_sat_4x8packed_ss_int(w16.s1, x16.s1, acc1);
            acc1 = dot_acc_sat_4x8packed_ss_int(w16.s2, x16.s2, acc1);
            acc1 = dot_acc_sat_4x8packed_ss_int(w16.s3, x16.s3, acc1);
        }
        if (n_base + 2 < N) {
            uint4 w16 = W[(n_base + 2) * K_div16 + k];
            acc2 = dot_acc_sat_4x8packed_ss_int(w16.s0, x16.s0, acc2);
            acc2 = dot_acc_sat_4x8packed_ss_int(w16.s1, x16.s1, acc2);
            acc2 = dot_acc_sat_4x8packed_ss_int(w16.s2, x16.s2, acc2);
            acc2 = dot_acc_sat_4x8packed_ss_int(w16.s3, x16.s3, acc2);
        }
        if (n_base + 3 < N) {
            uint4 w16 = W[(n_base + 3) * K_div16 + k];
            acc3 = dot_acc_sat_4x8packed_ss_int(w16.s0, x16.s0, acc3);
            acc3 = dot_acc_sat_4x8packed_ss_int(w16.s1, x16.s1, acc3);
            acc3 = dot_acc_sat_4x8packed_ss_int(w16.s2, x16.s2, acc3);
            acc3 = dot_acc_sat_4x8packed_ss_int(w16.s3, x16.s3, acc3);
        }
    }
    if (n_base + 0 < N) y[n_base + 0] = (float)acc0 * w_scale[n_base + 0] * x_scale;
    if (n_base + 1 < N) y[n_base + 1] = (float)acc1 * w_scale[n_base + 1] * x_scale;
    if (n_base + 2 < N) y[n_base + 2] = (float)acc2 * w_scale[n_base + 2] * x_scale;
    if (n_base + 3 < N) y[n_base + 3] = (float)acc3 * w_scale[n_base + 3] * x_scale;
}
"""


def bench_kernel(prog, kernel_name, args, gws, lws=None, iters=30, warmup=5):
    knl = getattr(prog, kernel_name)
    for _ in range(warmup):
        knl(queue, gws, lws, *args)
    queue.finish()
    t0 = time.perf_counter()
    for _ in range(iters):
        knl(queue, gws, lws, *args)
    queue.finish()
    return (time.perf_counter() - t0) * 1000.0 / iters


def test_shape(N, K, name, mnn_ref_ms=None):
    print(f"\n=== {name}: 1 x {K} → {N} (MNN ref: {mnn_ref_ms} ms) ===")
    np.random.seed(0)
    W_int8 = np.random.randint(-127, 127, (N, K), dtype=np.int8)
    x_int8 = np.random.randint(-127, 127, K, dtype=np.int8)
    w_scale = np.full(N, 0.01, dtype=np.float32)
    x_scale = np.float32(0.01)

    # Pack as uint4 (16 int8 per uint4)
    W_packed = W_int8.reshape(-1).view(np.uint32).reshape(N, K // 16, 4)
    W_packed_flat = W_packed.reshape(-1)  # uint32 elements
    x_packed = x_int8.view(np.uint32)

    W_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=W_packed_flat)
    x_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x_packed)
    s_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w_scale)
    y_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=N * 4)

    # Tuned single-row
    prog1 = cl.Program(ctx, KERNEL_IDP_TUNED).build()
    args1 = [W_buf, x_buf, s_buf, x_scale, y_buf, np.int32(N), np.int32(K // 16)]
    for lws_size in [16, 32, 64, 128, 256]:
        try:
            ms = bench_kernel(prog1, 'gemv_idp_tuned', args1, (N,), (lws_size,))
            print(f"  IDP tuned  lws={lws_size:4d}: {ms:7.3f} ms")
        except Exception as e:
            pass

    # Multi-row (4 rows per WI)
    prog2 = cl.Program(ctx, KERNEL_IDP_MULTI).build()
    args2 = [W_buf, x_buf, s_buf, x_scale, y_buf, np.int32(N), np.int32(K // 16)]
    N4 = (N + 3) // 4
    for lws_size in [16, 32, 64, 128]:
        try:
            ms = bench_kernel(prog2, 'gemv_idp_multi', args2, (N4,), (lws_size,))
            print(f"  IDP multi4 lws={lws_size:4d}: {ms:7.3f} ms")
        except Exception as e:
            pass


# Llama shapes with MNN reference (from h1_baseline measured at decode)
test_shape(3072, 2048, "QKV proj", mnn_ref_ms="0.82 (avg per call from instrumented)")
test_shape(8192, 2048, "FFN gate/up", mnn_ref_ms="1.15")
test_shape(2048, 8192, "FFN down", mnn_ref_ms="1.15")
test_shape(128256, 2048, "LM head", mnn_ref_ms="13.47")
