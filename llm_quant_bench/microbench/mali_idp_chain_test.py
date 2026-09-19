"""End-to-end-ish accuracy test: simulate a multi-layer chain of int8 matmul
to verify error doesn't blow up across layers. Mimics LLM forward pass.
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

KERNEL = """
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
__kernel void gemv_idp(
    __global const uint4* W, __global const uint4* x,
    __global const float* w_scale, float x_scale,
    __global float* y, int N, int K_div16)
{
    int n = get_global_id(0);
    if (n >= N) return;
    int acc = 0;
    for (int k = 0; k < K_div16; k++) {
        uint4 w16 = W[n * K_div16 + k]; uint4 x16 = x[k];
        acc = dot_acc_sat_4x8packed_ss_int(w16.s0, x16.s0, acc);
        acc = dot_acc_sat_4x8packed_ss_int(w16.s1, x16.s1, acc);
        acc = dot_acc_sat_4x8packed_ss_int(w16.s2, x16.s2, acc);
        acc = dot_acc_sat_4x8packed_ss_int(w16.s3, x16.s3, acc);
    }
    y[n] = (float)acc * w_scale[n] * x_scale;
}
"""
prog = cl.Program(ctx, KERNEL).build()


def matmul_idp(W_int8, x_int8, w_scale, x_scale):
    N, K = W_int8.shape
    W_packed = W_int8.reshape(-1).view(np.uint32)
    x_packed = x_int8.view(np.uint32)
    Wb = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=W_packed)
    xb = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x_packed)
    sb = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=w_scale.astype(np.float32))
    yb = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=N * 4)
    prog.gemv_idp(queue, (N,), None, Wb, xb, sb, np.float32(x_scale), yb, np.int32(N), np.int32(K // 16))
    queue.finish()
    y = np.empty(N, dtype=np.float32)
    cl.enqueue_copy(queue, y, yb)
    return y


def quantize_w(W_f, sym=True):
    """Symmetric per-row int8 quantization."""
    w_max = np.abs(W_f).max(axis=1, keepdims=True) + 1e-9
    scale = w_max / 127
    W_q = np.clip(np.round(W_f / scale), -127, 127).astype(np.int8)
    return W_q, scale.flatten()


def quantize_x(x_f):
    x_max = np.abs(x_f).max() + 1e-9
    scale = x_max / 127
    x_q = np.clip(np.round(x_f / scale), -127, 127).astype(np.int8)
    return x_q, float(scale)


def silu(x):
    return x / (1 + np.exp(-x))


# Simulate a 16-layer Llama-style forward pass
np.random.seed(42)
HIDDEN = 2048
INTER = 8192
N_LAYERS = 16

# Generate random "weights"
weights_fp = []
for layer in range(N_LAYERS):
    layer_w = {
        # simplified: only Q + FFN gate + FFN down
        'q': np.random.randn(HIDDEN, HIDDEN).astype(np.float32) * 0.05,
        'gate': np.random.randn(INTER, HIDDEN).astype(np.float32) * 0.05,
        'down': np.random.randn(HIDDEN, INTER).astype(np.float32) * 0.05,
    }
    weights_fp.append(layer_w)

# Initial activation (fake "embedding")
x_fp = np.random.randn(HIDDEN).astype(np.float32) * 0.5
x_initial = x_fp.copy()

# === Path 1: full fp32 reference ===
def forward_fp32(x):
    for layer_w in weights_fp:
        # Q proj
        q = layer_w['q'] @ x
        # Skip attention, use q as residual
        h = x + 0.1 * q  # mimic residual
        # FFN
        gate = silu(layer_w['gate'] @ h)
        ffn_out = layer_w['down'] @ gate
        x = h + 0.1 * ffn_out  # mimic FFN residual
    return x

y_fp32 = forward_fp32(x_initial.copy())

# === Path 2: int8 quantized using IDP ===
def forward_int8_idp(x):
    for layer_w in weights_fp:
        # Q proj
        Wq_q, sq_w = quantize_w(layer_w['q'])
        x_q, sq_x = quantize_x(x)
        q = matmul_idp(Wq_q, x_q, sq_w, sq_x)
        h = x + 0.1 * q
        # FFN gate
        Wg_q, sg_w = quantize_w(layer_w['gate'])
        h_q, sh_x = quantize_x(h)
        gate_pre = matmul_idp(Wg_q, h_q, sg_w, sh_x)
        gate = silu(gate_pre)
        # FFN down
        Wd_q, sd_w = quantize_w(layer_w['down'])
        gate_q, sgate_x = quantize_x(gate)
        ffn_out = matmul_idp(Wd_q, gate_q, sd_w, sgate_x)
        x = h + 0.1 * ffn_out
    return x

y_int8 = forward_int8_idp(x_initial.copy())

# Compare
diff = np.abs(y_fp32 - y_int8)
rel = diff / (np.abs(y_fp32) + 1e-9)
cos = (y_fp32 @ y_int8) / (np.linalg.norm(y_fp32) * np.linalg.norm(y_int8) + 1e-9)

print(f"\n=== 16-layer chain test ({HIDDEN} hidden, {INTER} intermediate) ===")
print(f"  Output norm: fp32={np.linalg.norm(y_fp32):.4f}, idp={np.linalg.norm(y_int8):.4f}")
print(f"  Element-wise: max abs diff={diff.max():.4f}, mean abs={diff.mean():.4f}")
print(f"  Element-wise rel error: max={rel.max():.4f}, mean={rel.mean():.4f}")
print(f"  Cosine similarity: {cos:.6f}")
print(f"  → typical W8A8 16-layer chain expectation: cos > 0.99")

if cos > 0.99:
    print(f"\n  ✓ IDP int8 forward chain preserves output direction")
else:
    print(f"\n  ✗ IDP forward chain DEGRADED (cos {cos:.4f} below 0.99)")

# Compare to a "perfect int8 quant" reference (int_accumulate_then_scale on CPU)
def forward_int8_ref_cpu(x):
    """Same int8 quantization but compute on CPU as reference."""
    for layer_w in weights_fp:
        Wq, sq = quantize_w(layer_w['q'])
        xq, sx = quantize_x(x)
        q = (Wq.astype(np.int32) @ xq.astype(np.int32)).astype(np.float32) * sq * sx
        h = x + 0.1 * q
        Wg, sg = quantize_w(layer_w['gate'])
        hq, shx = quantize_x(h)
        gate_pre = (Wg.astype(np.int32) @ hq.astype(np.int32)).astype(np.float32) * sg * shx
        gate = silu(gate_pre)
        Wd, sd = quantize_w(layer_w['down'])
        gq, sgx = quantize_x(gate)
        ffn_out = (Wd.astype(np.int32) @ gq.astype(np.int32)).astype(np.float32) * sd * sgx
        x = h + 0.1 * ffn_out
    return x

y_cpu = forward_int8_ref_cpu(x_initial.copy())
diff_cpu = np.abs(y_int8 - y_cpu)
print(f"\n=== IDP GPU vs CPU reference (same int8 quantization) ===")
print(f"  max abs diff: {diff_cpu.max():.4e}")
print(f"  mean abs diff: {diff_cpu.mean():.4e}")
if diff_cpu.max() < 1e-2:
    print(f"  ✓ GPU IDP matches CPU int8 reference (numerical equivalence)")
else:
    print(f"  ✗ GPU IDP diverges from CPU int8 reference!")
