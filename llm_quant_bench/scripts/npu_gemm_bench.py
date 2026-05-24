"""
CPU vs GPU vs NPU GEMM 벤치마크
- CPU: numpy (OpenBLAS + NEON, 4 threads)
- GPU: ACL cl_sgemm 결과 사용 (이미 측정됨)
- NPU: rknn-toolkit-lite2로 matmul 모델 실행

Usage: python3 scripts/npu_gemm_bench.py
"""
import numpy as np
import time
import os
import json

os.environ['OPENBLAS_NUM_THREADS'] = '4'

def bench_cpu(M, N, K, reps=10, warmup=10):
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    for _ in range(warmup):
        np.matmul(A, B)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        np.matmul(A, B)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def bench_npu_rknn(M, N, K, reps=10, warmup=5):
    """RKNN matmul benchmark using rknn-toolkit-lite2."""
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        print("  [WARN] rknnlite not available, trying rknn")
        try:
            from rknn.api import RKNN as RKNNLite
        except ImportError:
            print("  [ERROR] Neither rknnlite nor rknn available")
            return None

    # Create simple ONNX matmul model
    try:
        import onnx
        from onnx import helper, TensorProto
        HAS_ONNX = True
    except ImportError:
        HAS_ONNX = False

    if not HAS_ONNX:
        print("  [ERROR] onnx package required: pip install onnx")
        return None

    # Build ONNX model: C = MatMul(A, B)
    A_input = helper.make_tensor_value_info('A', TensorProto.FLOAT, [M, K])
    B_input = helper.make_tensor_value_info('B', TensorProto.FLOAT, [K, N])
    C_output = helper.make_tensor_value_info('C', TensorProto.FLOAT, [M, N])

    matmul_node = helper.make_node('MatMul', inputs=['A', 'B'], outputs=['C'])

    graph = helper.make_graph([matmul_node], 'matmul_graph', [A_input, B_input], [C_output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])

    onnx_path = '/tmp/matmul_%d_%d_%d.onnx' % (M, N, K)
    rknn_path = '/tmp/matmul_%d_%d_%d.rknn' % (M, N, K)
    onnx.save(model, onnx_path)

    # Convert to RKNN
    rknn = RKNNLite()

    # For rknnlite, we need a pre-converted model
    # Use rknn-toolkit2 to convert
    try:
        from rknn.api import RKNN
        rknn_conv = RKNN()
        rknn_conv.config(target_platform='rk3588')
        ret = rknn_conv.load_onnx(model=onnx_path)
        if ret != 0:
            print('  [ERROR] Load ONNX failed')
            return None
        ret = rknn_conv.build(do_quantization=False)
        if ret != 0:
            print('  [ERROR] Build failed')
            return None
        ret = rknn_conv.export_rknn(rknn_path)
        if ret != 0:
            print('  [ERROR] Export failed')
            return None
        rknn_conv.release()
    except ImportError:
        print("  [ERROR] rknn.api (full toolkit) needed for conversion")
        return None

    # Run on NPU with rknnlite
    from rknnlite.api import RKNNLite as RKNNLiteAPI
    lite = RKNNLiteAPI()
    ret = lite.load_rknn(rknn_path)
    if ret != 0:
        print('  [ERROR] Load RKNN failed')
        return None

    # core_mask=7 → use all 3 NPU cores
    ret = lite.init_runtime(core_mask=RKNNLiteAPI.NPU_CORE_0_1_2)
    if ret != 0:
        print('  [ERROR] Init runtime failed, trying single core')
        ret = lite.init_runtime(core_mask=RKNNLiteAPI.NPU_CORE_0)
        if ret != 0:
            print('  [ERROR] Init runtime failed')
            return None

    A_data = np.random.randn(M, K).astype(np.float32)
    B_data = np.random.randn(K, N).astype(np.float32)

    # warmup
    for _ in range(warmup):
        lite.inference(inputs=[A_data, B_data])

    # measure
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        lite.inference(inputs=[A_data, B_data])
        times.append((time.perf_counter() - t0) * 1000)

    lite.release()

    # cleanup
    os.remove(onnx_path)
    os.remove(rknn_path)

    return float(np.median(times))


def main():
    sizes = [32, 64, 128, 256, 512, 1024]
    reps = 10

    print("=" * 75)
    print("CPU(OpenBLAS 4T) vs NPU(RK3588 3-core) GEMM — FP32, M=N=K")
    print("Warmup: 10 | Measure: 10 (median)")
    print("=" * 75)

    results = {}

    print("\n[1/2] CPU (OpenBLAS + NEON, 4 threads)...")
    for S in sizes:
        ms = bench_cpu(S, S, S, reps)
        ops = 2.0 * S * S * S
        gf = ops / (ms / 1000) / 1e9
        results.setdefault(S, {})['cpu_ms'] = ms
        results[S]['cpu_gflops'] = gf
        print("  %4d x %4d : %8.3f ms  (%6.2f GFLOPS)" % (S, S, ms, gf))

    print("\n[2/2] NPU (RKNN, 3 cores)...")
    for S in sizes:
        print("  %4d x %4d : " % (S, S), end="", flush=True)
        ms = bench_npu_rknn(S, S, S, reps)
        if ms is not None:
            ops = 2.0 * S * S * S
            gf = ops / (ms / 1000) / 1e9
            results.setdefault(S, {})['npu_ms'] = ms
            results[S]['npu_gflops'] = gf
            print("%8.3f ms  (%6.2f GFLOPS)" % (ms, gf))
        else:
            print("FAILED")

    # Comparison
    print("\n" + "=" * 75)
    print("%-6s | %12s %10s | %12s %10s | %s" % (
        "Size", "CPU(ms)", "GFLOPS", "NPU(ms)", "GFLOPS", "Winner"))
    print("-" * 75)

    for S in sizes:
        r = results.get(S, {})
        cpu_ms = r.get('cpu_ms')
        npu_ms = r.get('npu_ms')
        if cpu_ms and npu_ms:
            if npu_ms < cpu_ms:
                winner = "<< NPU (%.1fx)" % (cpu_ms / npu_ms)
            else:
                winner = "CPU >> (%.1fx)" % (npu_ms / cpu_ms)
            print("%-6d | %8.3f ms %8.2f  | %8.3f ms %8.2f  | %s" % (
                S, cpu_ms, r.get('cpu_gflops', 0), npu_ms, r.get('npu_gflops', 0), winner))

    print("=" * 75)

    # Save
    output_dir = './results/raw/gpu_analysis'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'npu_crossover.json'), 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print("\nSaved to %s/npu_crossover.json" % output_dir)


if __name__ == "__main__":
    main()
