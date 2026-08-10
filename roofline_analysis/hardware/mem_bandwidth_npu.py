"""
NPU Memory Bandwidth Estimation
대형 identity ONNX 모델로 RK3588 NPU DMA 대역폭 추정.

Usage:
    python hardware/mem_bandwidth_npu.py
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


def bench_npu_identity(size_mb, reps=20, warmup=5, core_mask=7):
    """Estimate NPU memory bandwidth via identity (passthrough) model."""
    try:
        from rknnlite.api import RKNNLite
        from rknn.api import RKNN
        import onnx
        from onnx import helper, TensorProto
    except ImportError as e:
        print(f"  [ERROR] Missing dependency: {e}")
        return None

    # Create identity ONNX model: output = input (just copy)
    n_elements = int(size_mb * 1024 * 1024 / 4)  # float32 = 4 bytes
    shape = [1, n_elements]

    inp = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
    out = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)
    node = helper.make_node('Identity', inputs=['input'], outputs=['output'])
    graph = helper.make_graph([node], 'identity_graph', [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])

    onnx_path = f'/tmp/roofline_identity_{size_mb}mb.onnx'
    rknn_path = f'/tmp/roofline_identity_{size_mb}mb.rknn'
    onnx.save(model, onnx_path)

    # Convert
    rknn_conv = RKNN(verbose=False)
    rknn_conv.config(target_platform='rk3588')
    if rknn_conv.load_onnx(model=onnx_path) != 0:
        return None
    if rknn_conv.build(do_quantization=False) != 0:
        return None
    if rknn_conv.export_rknn(rknn_path) != 0:
        return None
    rknn_conv.release()

    # Run
    lite = RKNNLite(verbose=False)
    if lite.load_rknn(rknn_path) != 0:
        return None

    try:
        ret = lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    except Exception:
        ret = lite.init_runtime(core_mask=core_mask)
    if ret != 0:
        ret = lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            return None

    data = np.random.randn(*shape).astype(np.float32)

    for _ in range(warmup):
        lite.inference(inputs=[data])

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        lite.inference(inputs=[data])
        times.append((time.perf_counter() - t0) * 1000)

    lite.release()
    for p in [onnx_path, rknn_path]:
        if os.path.exists(p):
            os.remove(p)

    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description="NPU Memory Bandwidth Estimation")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    mb = cfg["mem_bandwidth"]
    sizes_mb = mb.get("sizes_mb_npu", mb["sizes_mb"])
    reps = mb["reps"]
    warmup = mb["warmup"]
    core_mask = cfg["hardware"]["npu"]["core_mask"]

    output_dir = args.output_dir or os.path.join(PROJECT_DIR, "results", "raw")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"NPU Memory Bandwidth Estimation — RK3588 (Identity Model)")
    print(f"Warmup: {warmup} | Measure: {reps} (median)")
    print(f"{'='*70}")

    results = []

    for size_mb in sizes_mb:
        print(f"  {size_mb:4d} MB : ", end="", flush=True)
        ms = bench_npu_identity(size_mb, reps, warmup, core_mask)
        if ms is not None:
            n_bytes = size_mb * 1024 * 1024
            bw = 2 * n_bytes / (ms / 1000) / 1e9  # read + write
            results.append({
                "backend": "npu",
                "operation": "identity",
                "size_mb": size_mb,
                "best_time_ms": round(ms, 4),
                "bandwidth_gb_s": round(bw, 3),
            })
            print(f"{ms:10.3f} ms  ({bw:8.3f} GB/s)")
        else:
            print("FAILED")

    if results:
        peak = max(results, key=lambda r: r["bandwidth_gb_s"])
        print(f"\n{'='*70}")
        print(f"NPU Peak Bandwidth: {peak['bandwidth_gb_s']:.3f} GB/s (at {peak['size_mb']} MB)")
        print(f"{'='*70}")

    csv_path = os.path.join(output_dir, "mem_bw_npu.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "operation", "size_mb",
                                                "best_time_ms", "bandwidth_gb_s"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    main()
