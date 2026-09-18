"""IDP kernel accuracy verification: compare against fp32 reference computation.

Tests:
1. Tiny matrix (4x4, 16x16) — exact comparison
2. Realistic shapes (Llama dims) — statistical comparison
3. End-to-end: feed actual int8 weights and check output
"""
import os, sys
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

# Same IDP kernel as production
KERNEL_IDP = """
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable

__kernel void gemv_idp(
    __global const uint4* W,
    __global const uint4* x,
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
        acc = dot_acc_sat_4x8packed_ss_int(w16.s0, x16.s0, acc);
        acc = dot_acc_sat_4x8packed_ss_int(w16.s1, x16.s1, acc);
        acc = dot_acc_sat_4x8packed_ss_int(w16.s2, x16.s2, acc);
        acc = dot_acc_sat_4x8packed_ss_int(w16.s3, x16.s3, acc);
    }
    y[n] = (float)acc * w_scale[n] * x_scale;
}
"""

prog = cl.Program(ctx, KERNEL_IDP).build()

def run_idp(W_int8, x_int8, w_scale, x_scale):
    N, K = W_int8.shape
    W_packed = W_int8.reshape(-1).view(np.uint32)
    x_packed = x_int8.view(np.uint32)
    W_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=W_packed)
    x_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x_packed)
    s_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w_scale.astype(np.float32))
    y_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=N * 4)
    prog.gemv_idp(queue, (N,), None, W_buf, x_buf, s_buf, np.float32(x_scale), y_buf, np.int32(N), np.int32(K // 16))
    queue.finish()
    y = np.empty(N, dtype=np.float32)
    cl.enqueue_copy(queue, y, y_buf)
    return y


def reference(W_int8, x_int8, w_scale, x_scale):
    """Reference: compute in fp32, dequant first."""
    W_f = W_int8.astype(np.float32) * w_scale[:, None]
    x_f = x_int8.astype(np.float32) * x_scale
    return W_f @ x_f


def reference_int_then_scale(W_int8, x_int8, w_scale, x_scale):
    """Mathematical equivalent: int matmul then scale (what IDP does)."""
    # int32 accumulate
    acc = np.einsum('nk,k->n', W_int8.astype(np.int32), x_int8.astype(np.int32))
    return acc.astype(np.float32) * w_scale * x_scale


def compare(name, W_int8, x_int8, w_scale, x_scale):
    print(f"\n=== {name}: W shape {W_int8.shape}, K={W_int8.shape[1]} ===")
    y_idp = run_idp(W_int8, x_int8, w_scale, x_scale)
    y_ref = reference(W_int8, x_int8, w_scale, x_scale)
    y_int_ref = reference_int_then_scale(W_int8, x_int8, w_scale, x_scale)

    # IDP should match int_ref EXACTLY (same arithmetic)
    diff_int = np.abs(y_idp - y_int_ref)
    max_int = diff_int.max()
    print(f"  IDP vs int_then_scale ref: max abs diff = {max_int:.2e}")
    if max_int < 1e-3:
        print(f"  ✓ IDP matches int reference (exact)")
    else:
        print(f"  ✗ IDP differs from int reference!")
        idx = np.argmax(diff_int)
        print(f"    worst at idx {idx}: idp={y_idp[idx]:.6f}, ref={y_int_ref[idx]:.6f}")

    # IDP vs fp32 reference (dequant first then matmul)
    diff_fp = np.abs(y_idp - y_ref)
    max_fp = diff_fp.max()
    rel_fp = (diff_fp / (np.abs(y_ref) + 1e-9)).max()
    print(f"  IDP vs fp32 ref: max abs diff = {max_fp:.2e}, max rel = {rel_fp:.2e}")

    # cosine similarity
    cos = (y_idp @ y_ref) / (np.linalg.norm(y_idp) * np.linalg.norm(y_ref) + 1e-9)
    print(f"  cosine sim: {cos:.6f}")


# Test 1: Tiny exact 16-byte aligned
np.random.seed(0)
W = np.random.randint(-127, 127, (4, 16), dtype=np.int8)
x = np.random.randint(-127, 127, 16, dtype=np.int8)
ws = np.full(4, 0.01)
compare("Tiny 4x16", W, x, ws, 0.01)

# Test 2: Llama QKV proj shape
W = np.random.randint(-127, 127, (3072, 2048), dtype=np.int8)
x = np.random.randint(-127, 127, 2048, dtype=np.int8)
ws = np.full(3072, 0.01)
compare("Llama QKV (3072x2048)", W, x, ws, 0.01)

# Test 3: realistic LM head shape (large N)
W = np.random.randint(-127, 127, (128256, 2048), dtype=np.int8)
x = np.random.randint(-127, 127, 2048, dtype=np.int8)
ws = np.full(128256, 0.01)
compare("Llama LM head (128256x2048)", W, x, ws, 0.01)

# Test 4: realistic LLM weight distribution (more like real Llama weights)
np.random.seed(42)
# Real Llama W8 weights have approximately Gaussian distribution
W_f = np.random.randn(3072, 2048) * 0.05  # typical scale
x_f = np.random.randn(2048) * 0.5
# Quantize symmetrically
w_max = np.abs(W_f).max(axis=1, keepdims=True)
w_scale = w_max.flatten() / 127
W_q = np.clip(W_f / (w_scale[:, None] + 1e-9), -127, 127).astype(np.int8)
x_max = np.abs(x_f).max()
x_scale = x_max / 127
x_q = np.clip(x_f / x_scale, -127, 127).astype(np.int8)

# Reference: compute in full fp32
y_full_fp = (W_f @ x_f)

# IDP-style int8 quantized result
y_idp = run_idp(W_q, x_q, w_scale, x_scale)

# Compare quantized result to fp32 ground truth
diff = np.abs(y_idp - y_full_fp)
rel = (diff / (np.abs(y_full_fp) + 1e-9))
cos = (y_idp @ y_full_fp) / (np.linalg.norm(y_idp) * np.linalg.norm(y_full_fp) + 1e-9)
print(f"\n=== Realistic Llama-like weights (Gaussian init): ===")
print(f"  full-fp32 reference vs IDP int8: max abs = {diff.max():.4f}, mean abs = {diff.mean():.4f}")
print(f"  max rel error = {rel.max():.4f}, mean rel = {rel.mean():.4f}")
print(f"  cosine similarity = {cos:.6f}")
print(f"  (typical W8A8 quantization: cos > 0.999 expected)")
