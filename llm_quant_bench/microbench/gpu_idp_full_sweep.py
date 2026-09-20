"""GPU IDP standalone benchmark across full ctx range for v10 paper.

Predicts MNN-equivalent prefill_tok_s and decode_tok_s by measuring
the dominant matmul (FFN gate as proxy) at appropriate batch sizes.

Output format matches comprehensive_baselines.csv for easy merging.
"""
import os, sys, time, csv
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
queue = cl.CommandQueue(ctx)

KERNEL = """
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

__kernel void gemm_idp(
    __global const uint4* W, __global const uint4* X,
    __global const float* w_scale, float x_scale,
    __global float* Y,
    int M, int N, int K_div16)
{
    int n = get_global_id(0); int m = get_global_id(1);
    if (n >= N || m >= M) return;
    int acc = 0;
    for (int k = 0; k < K_div16; k++) {
        uint4 w16 = W[n * K_div16 + k]; uint4 x16 = X[m * K_div16 + k];
        acc = dot_acc_sat_4x8packed_ss_int(w16.s0, x16.s0, acc);
        acc = dot_acc_sat_4x8packed_ss_int(w16.s1, x16.s1, acc);
        acc = dot_acc_sat_4x8packed_ss_int(w16.s2, x16.s2, acc);
        acc = dot_acc_sat_4x8packed_ss_int(w16.s3, x16.s3, acc);
    }
    Y[m * N + n] = (float)acc * w_scale[n] * x_scale;
}
"""
prog = cl.Program(ctx, KERNEL).build()
gemm_kern = cl.Kernel(prog, 'gemm_idp')


def bench_gemm(M, K, N, iters=10, warmup=3):
    np.random.seed(0)
    W = np.random.randint(-127, 127, (N, K), dtype=np.int8).reshape(-1).view(np.uint32)
    X = np.random.randint(-127, 127, (M, K), dtype=np.int8).reshape(-1).view(np.uint32)
    w_scale = np.full(N, 0.01, dtype=np.float32)

    Wb = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=W)
    Xb = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=X)
    sb = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w_scale)
    yb = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=M * N * 4)

    # Round up dims to multiples
    lws_n, lws_m = 4, 16
    gws_n = ((N + lws_n - 1) // lws_n) * lws_n
    gws_m = ((M + lws_m - 1) // lws_m) * lws_m
    args = [Wb, Xb, sb, np.float32(0.01), yb, np.int32(M), np.int32(N), np.int32(K // 16)]
    kern = gemm_kern
    for _ in range(warmup):
        kern(queue, (gws_n, gws_m), (lws_n, lws_m), *args)
    queue.finish()
    t0 = time.perf_counter()
    for _ in range(iters):
        kern(queue, (gws_n, gws_m), (lws_n, lws_m), *args)
    queue.finish()
    return (time.perf_counter() - t0) * 1000.0 / iters


def estimate_full_layer_time(M, K, N_total, n_layers, ms_per_matmul, n_matmul_per_layer=5):
    """Estimate per-token time including all layers."""
    return ms_per_matmul * n_matmul_per_layer * n_layers / M


# Llama-3.2-1B dimensions
HIDDEN = 2048
INTER = 8192
N_LAYERS = 16
LM_HEAD = 128256

# For each ctx, simulate prefill (large M) and decode (M=1)
OUT = '/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/paper/v10/v10_data/gpu_idp_baselines.csv'
print(f"GPU IDP standalone benchmark, output: {OUT}")
print(f"{'ctx':>6} {'prefill_pred':>14} {'decode_pred':>13} {'matmul_pf_ms':>14} {'matmul_dc_ms':>14}")

with open(OUT, 'w') as f:
    w = csv.writer(f)
    w.writerow(['config_id', 'ctx', 'prefill_tok_s', 'decode_tok_s', 'note'])
    for c_ctx in [64, 256, 512, 1024, 2048, 3500]:
        # Prefill: M=ctx, dominant op = FFN gate (2048→8192)
        # Decode: M=1, all 5 matmuls per layer
        # Realistic: each layer does 5 matmuls of varying shape
        # Approximation: use FFN gate (2048→8192) shape × M and × 1

        # Prefill time: per-layer = (QKV + O + gate + up + down) ~~ 5 × FFN_gate
        ms_pf_per_matmul = bench_gemm(c_ctx, HIDDEN, INTER, iters=5)  # FFN gate for whole prompt batch
        # Approximation: full layer ≈ 4 × FFN_gate-equivalent matmul time + 1 LM head time
        # More accurate: sum of QKV (3072), O (2048), gate (8192), up (8192), down (8192)
        # In ratio: 3072 + 2048 + 8192 + 8192 + 8192 = 29696, ÷ 8192 = 3.625 × FFN_gate-equivalent
        ms_layer_total_pf = ms_pf_per_matmul * 3.625
        # 16 layers + LM head
        ms_lm_head_pf = bench_gemm(c_ctx, HIDDEN, LM_HEAD, iters=2) if c_ctx <= 256 else bench_gemm(min(c_ctx, 256), HIDDEN, LM_HEAD, iters=2) * c_ctx / 256
        # Total prefill time ≈ 16 × ms_layer + LM head + attn
        ms_prefill_total = ms_layer_total_pf * N_LAYERS + ms_lm_head_pf
        prefill_tok_s = c_ctx * 1000.0 / ms_prefill_total

        # Decode (M=1): 5 matmuls per layer + LM head
        ms_dc_per_matmul = bench_gemm(1, HIDDEN, INTER, iters=10)
        ms_dc_per_layer = ms_dc_per_matmul * 3.625
        ms_dc_lm_head = bench_gemm(1, HIDDEN, LM_HEAD, iters=10)
        ms_decode_per_token = ms_dc_per_layer * N_LAYERS + ms_dc_lm_head
        decode_tok_s = 1000.0 / ms_decode_per_token

        print(f"{c_ctx:>6} {prefill_tok_s:>14.2f} {decode_tok_s:>13.2f} {ms_pf_per_matmul:>14.3f} {ms_dc_per_matmul:>14.3f}")
        w.writerow(['gpu_idp_proj', c_ctx, f'{prefill_tok_s:.2f}', f'{decode_tok_s:.2f}', 'projected_from_microbench'])

print(f"\nSaved: {OUT}")
print("\nNote: gpu_idp_proj is projected from standalone IDP matmul microbench.")
print("To get actual MNN integration measurement, IDP would need to be added")
print("to ConvBufLowMemoryExecution.cpp. Current numbers represent achievable upper bound.")
