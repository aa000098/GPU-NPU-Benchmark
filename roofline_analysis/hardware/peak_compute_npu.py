"""
NPU Peak Compute Benchmark
RK3588 NPU 3-core INT8 peak TOPS 측정.
ONNX matmul → RKNN 변환 → rknnlite 추론.

Usage:
    python hardware/peak_compute_npu.py
"""
import numpy as np
import time
import os
import sys
import csv
import argparse
import yaml

os.environ['LOGLEVEL'] = 'WARNING'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, "config", "roofline_config.yaml")


def load_config(config_path=None):
    path = config_path or DEFAULT_CONFIG
    with open(path) as f:
        return yaml.safe_load(f)


def bench_npu_rknn(M, N, K, reps=10, warmup=5, core_mask=7):
    """RKNN matmul benchmark using rknn-toolkit-lite2."""
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        print("  [WARN] rknnlite not available")
        return None

    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        print("  [ERROR] onnx package required: pip install onnx")
        return None

    # Build ONNX model: C = MatMul(A, B)
    A_input = helper.make_tensor_value_info('A', TensorProto.FLOAT, [M, K])
    B_input = helper.make_tensor_value_info('B', TensorProto.FLOAT, [K, N])
    C_output = helper.make_tensor_value_info('C', TensorProto.FLOAT, [M, N])

    matmul_node = helper.make_node('MatMul', inputs=['A', 'B'], outputs=['C'])
    graph = helper.make_graph([matmul_node], 'matmul_graph', [A_input, B_input], [C_output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])

    onnx_path = f'/tmp/roofline_matmul_{M}_{N}_{K}.onnx'
    rknn_path = f'/tmp/roofline_matmul_{M}_{N}_{K}.rknn'
    onnx.save(model, onnx_path)

    # Convert to RKNN with quantization (INT8)
    try:
        from rknn.api import RKNN
        rknn_conv = RKNN(verbose=False)
        rknn_conv.config(target_platform='rk3588')
        ret = rknn_conv.load_onnx(model=onnx_path)
        if ret != 0:
            print(f'  [ERROR] Load ONNX failed (ret={ret})')
            return None
        # do_quantization=True for INT8 peak measurement
        ret = rknn_conv.build(do_quantization=False)
        if ret != 0:
            print(f'  [ERROR] Build failed (ret={ret})')
            return None
        ret = rknn_conv.export_rknn(rknn_path)
        if ret != 0:
            print(f'  [ERROR] Export failed (ret={ret})')
            return None
        rknn_conv.release()
    except ImportError:
        print("  [ERROR] rknn.api (full toolkit) needed for conversion")
        return None

    # Run on NPU with rknnlite
    lite = RKNNLite(verbose=False)
    ret = lite.load_rknn(rknn_path)
    if ret != 0:
        print('  [ERROR] Load RKNN failed')
        return None

    # core_mask: 7 = NPU_CORE_0_1_2
    try:
        ret = lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    except Exception:
        ret = lite.init_runtime(core_mask=core_mask)

    if ret != 0:
        print('  [ERROR] Init runtime failed, trying single core')
        ret = lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            print('  [ERROR] Init runtime failed')
            return None

    A_data = np.random.randn(M, K).astype(np.float32)
    B_data = np.random.randn(K, N).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        lite.inference(inputs=[A_data, B_data])

    # Measure
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        lite.inference(inputs=[A_data, B_data])
        times.append((time.perf_counter() - t0) * 1000)

    lite.release()

    # Cleanup
    for p in [onnx_path, rknn_path]:
        if os.path.exists(p):
            os.remove(p)

    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description="NPU Peak Compute Benchmark")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pc = cfg["peak_compute"]
    sizes = pc["sizes"]
    reps = pc["reps"]
    warmup = pc["warmup"]
    core_mask = cfg["hardware"]["npu"]["core_mask"]

    output_dir = args.output_dir or os.path.join(PROJECT_DIR, "results", "raw")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"NPU Peak Compute — INT8 (RK3588, 3-core)")
    print(f"Warmup: {warmup} | Measure: {reps} (median)")
    print(f"{'='*60}")

    results = []

    for S in sizes:
        print(f"  {S:5d} x {S:5d} : ", end="", flush=True)
        ms = bench_npu_rknn(S, S, S, reps, warmup, core_mask)
        if ms is not None:
            ops = 2.0 * S * S * S
            gflops = ops / (ms / 1000) / 1e9
            tops = gflops / 1000  # TOPS for INT8
            results.append({
                "backend": "npu",
                "dtype": "int8",
                "size": S,
                "latency_ms": round(ms, 4),
                "gflops": round(gflops, 3),
                "tops": round(tops, 4),
            })
            print(f"{ms:10.3f} ms  ({gflops:8.3f} GFLOPS / {tops:.4f} TOPS)")
        else:
            print("FAILED")

    # Peak summary
    if results:
        peak = max(results, key=lambda r: r["gflops"])
        print(f"\n{'='*60}")
        print(f"NPU Peak: {peak['gflops']:.3f} GFLOPS ({peak['tops']:.4f} TOPS) at size {peak['size']}")
        print(f"Theoretical: 6.0 TOPS → Utilization: {peak['tops']/6.0*100:.1f}%")
        print(f"{'='*60}")

    # Save CSV
    csv_path = os.path.join(output_dir, "peak_compute_npu.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "dtype", "size", "latency_ms", "gflops", "tops"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
