"""Test Mali-G610 IDP (Integer Dot Product) int8 GEMM kernel.

Compare three approaches for the same int8 GEMM (1×K @ K×N for decode GEMV):
1. Naive: dequant int8→fp16 first, then fp16 matmul (current MNN approach)
2. IDP (Khronos): use dot_4x8packed_ss_int from cl_khr_integer_dot_product
3. IDP (ARM): use arm_dot from cl_arm_integer_dot_product_int8

Measure throughput and verify numerical equivalence.

Llama-3.2-1B shapes for decode (M=1):
- QKV proj: 1 × 2048 → 3072
- FFN gate: 1 × 2048 → 8192
- FFN down: 1 × 8192 → 2048
- LM head: 1 × 2048 → 128256
"""
import os, sys, time
import numpy as np
import pyopencl as cl
import pyopencl.array as cla

# Force ARM platform
os.environ['PYOPENCL_CTX'] = '0'

# Choose platform/device
plats = cl.get_platforms()
mali_dev = None
for p in plats:
    for d in p.get_devices():
        if 'Mali' in d.name:
            mali_dev = d
            break
    if mali_dev: break

if mali_dev is None:
    print("No Mali device found")
    sys.exit(1)

ctx = cl.Context([mali_dev])
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

print(f"Using device: {mali_dev.name}")
print(f"OpenCL version: {mali_dev.version}")

# === Kernel sources ===

# Approach 1: dequant int8→fp16 then fp matmul
KERNEL_DEQUANT_FP16 = """
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void gemv_dequant_fp16(
    __global const char* W,      // int8 [N, K]
    __global const half* x,      // fp16 [K]
    __global const float* w_scale, // [N]  per-row scale
    __global float* y,           // fp32 [N]
    int N, int K)
{
    int n = get_global_id(0);
    if (n >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        char w_int = W[n * K + k];
        // dequant on the fly
        float w_f = (float)w_int * w_scale[n];
        acc += w_f * (float)x[k];
    }
    y[n] = acc;
}
"""

# Approach 2: IDP using Khronos cl_khr_integer_dot_product
# Function: int dot_4x8packed_ss_int(uint, uint) → int
KERNEL_IDP_KHR = """
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

__kernel void gemv_idp_khr(
    __global const uint* W_packed,  // int8 [N, K/4] packed as 4×int8 per uint
    __global const uint* x_packed,  // int8 [K/4] packed
    __global const float* w_scale,
    float x_scale,
    __global float* y,
    int N, int K_div4)
{
    int n = get_global_id(0);
    if (n >= N) return;
    int acc = 0;
    for (int k = 0; k < K_div4; k++) {
        uint w4 = W_packed[n * K_div4 + k];
        uint x4 = x_packed[k];
        // 4-way int8 dot product accumulate
        acc = dot_acc_sat_4x8packed_ss_int(w4, x4, acc);
    }
    y[n] = (float)acc * w_scale[n] * x_scale;
}
"""

# Approach 3: ARM-specific arm_dot
KERNEL_IDP_ARM = """
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_int8 : enable
#pragma OPENCL EXTENSION cl_arm_integer_dot_product_accumulate_int8 : enable

__kernel void gemv_idp_arm(
    __global const char4* W,    // int8 [N, K/4] as char4
    __global const char4* x,    // int8 [K/4]
    __global const float* w_scale,
    float x_scale,
    __global float* y,
    int N, int K_div4)
{
    int n = get_global_id(0);
    if (n >= N) return;
    int acc = 0;
    for (int k = 0; k < K_div4; k++) {
        char4 w4 = W[n * K_div4 + k];
        char4 x4 = x[k];
        // arm_dot_acc(char4, char4, int) → int, ARM extension
        acc = arm_dot_acc(w4, x4, acc);
    }
    y[n] = (float)acc * w_scale[n] * x_scale;
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


def test_shape(N, K, name):
    print(f"\n=== {name}: 1 x {K} @ {K} x {N} ===")
    np.random.seed(0)
    W_int8 = np.random.randint(-127, 127, (N, K), dtype=np.int8)
    x_int8 = np.random.randint(-127, 127, K, dtype=np.int8)
    x_fp16 = x_int8.astype(np.float16) * 0.01  # for fp16 path
    w_scale = np.full(N, 0.01, dtype=np.float32)
    x_scale = np.float32(0.01)

    # Pack int8 → uint32 (4 ints per uint)
    # Each row of W has K bytes, view as K/4 uint32
    W_packed_u32 = W_int8.reshape(-1).view(np.uint32).reshape(N, K // 4)
    x_packed_u32 = x_int8.view(np.uint32)

    # === Approach 1: dequant fp16 ===
    prog1 = cl.Program(ctx, KERNEL_DEQUANT_FP16).build()
    W_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=W_int8.flatten())
    x_buf_fp16 = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x_fp16)
    s_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w_scale)
    y_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=N * 4)
    args1 = [W_buf, x_buf_fp16, s_buf, y_buf, np.int32(N), np.int32(K)]
    try:
        ms1 = bench_kernel(prog1, 'gemv_dequant_fp16', args1, (N,))
        print(f"  Approach 1 (dequant fp16):    {ms1:.3f} ms")
    except Exception as e:
        print(f"  Approach 1 FAIL: {e}")
        ms1 = None

    # === Approach 2: IDP Khronos ===
    try:
        prog2 = cl.Program(ctx, KERNEL_IDP_KHR).build()
        Wp_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=W_packed_u32)
        xp_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x_packed_u32)
        args2 = [Wp_buf, xp_buf, s_buf, x_scale, y_buf, np.int32(N), np.int32(K // 4)]
        ms2 = bench_kernel(prog2, 'gemv_idp_khr', args2, (N,))
        speedup2 = ms1 / ms2 if ms1 else 0
        print(f"  Approach 2 (IDP Khronos):     {ms2:.3f} ms  ({speedup2:.2f}x vs dequant)")
    except Exception as e:
        print(f"  Approach 2 FAIL: {e}")
        ms2 = None

    # === Approach 3: IDP ARM ===
    try:
        prog3 = cl.Program(ctx, KERNEL_IDP_ARM).build()
        W_char4 = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=W_int8.flatten())
        x_char4 = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x_int8)
        args3 = [W_char4, x_char4, s_buf, x_scale, y_buf, np.int32(N), np.int32(K // 4)]
        ms3 = bench_kernel(prog3, 'gemv_idp_arm', args3, (N,))
        speedup3 = ms1 / ms3 if ms1 else 0
        print(f"  Approach 3 (IDP ARM):         {ms3:.3f} ms  ({speedup3:.2f}x vs dequant)")
    except Exception as e:
        print(f"  Approach 3 FAIL: {e}")


# Llama-3.2-1B shapes
test_shape(3072, 2048, "QKV proj")
test_shape(8192, 2048, "FFN gate/up")
test_shape(2048, 8192, "FFN down")
test_shape(128256, 2048, "LM head")
