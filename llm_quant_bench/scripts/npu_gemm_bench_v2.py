"""
CPU vs NPU GEMM 벤치마크 v2
- CPU: numpy (OpenBLAS + NEON, 4 threads)
- NPU: ONNX → RKNN 변환을 rknnlite로 직접 수행 (rknn.api 없이)

rknn.api가 torch 에러 때문에 안 되므로,
rknn-toolkit2의 rknn_matmul_api를 사용하거나,
미리 만든 RKNN 모델로 벤치마크.

Usage: python3 scripts/npu_gemm_bench_v2.py
"""
import numpy as np
import time
import os
import json
import ctypes

os.environ['OPENBLAS_NUM_THREADS'] = '4'

# ── RKNN API via ctypes ──────────────────────────────────────────────
RKNN_LIB_PATH = "/home/hyunho.son/install_files/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"

def bench_cpu(M, N, K, reps=10, warmup=10):
    """CPU GEMM: OpenBLAS + NEON + 4 threads."""
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


def bench_npu_via_rknnlite(M, N, K, reps=10, warmup=5):
    """NPU matmul using rknn_matmul_api (if available in rknnlite)."""
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        print("  rknnlite not available")
        return None

    # Check if matmul API is available
    rknn = RKNNLite(verbose=False)

    # rknnlite의 matmul API 확인
    if hasattr(rknn, 'init_matmul'):
        # Direct matmul API
        ret = rknn.init_matmul()
        if ret != 0:
            print("  init_matmul failed")
            return None

        A_data = np.random.randn(M, K).astype(np.float32)
        B_data = np.random.randn(K, N).astype(np.float32)

        # warmup
        for _ in range(warmup):
            rknn.matmul(A_data, B_data)

        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            rknn.matmul(A_data, B_data)
            times.append((time.perf_counter() - t0) * 1000)

        rknn.release()
        return float(np.median(times))
    else:
        rknn.release()
        return None


def bench_npu_via_onnx_rknnlite(M, N, K, reps=10, warmup=5):
    """
    NPU benchmark: ONNX 모델을 만들어서 rknnlite로 변환+실행.
    rknnlite.load_onnx → build → inference
    """
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        print("  rknnlite not available")
        return None

    # ONNX 모델 생성
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        print("  onnx not available")
        return None

    A_input = helper.make_tensor_value_info('A', TensorProto.FLOAT, [M, K])
    B_input = helper.make_tensor_value_info('B', TensorProto.FLOAT, [K, N])
    C_output = helper.make_tensor_value_info('C', TensorProto.FLOAT, [M, N])
    matmul_node = helper.make_node('MatMul', inputs=['A', 'B'], outputs=['C'])
    graph = helper.make_graph([matmul_node], 'matmul', [A_input, B_input], [C_output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    onnx_path = '/tmp/matmul_%d.onnx' % M
    onnx.save(model, onnx_path)

    rknn = RKNNLite(verbose=False)

    # rknnlite에서 load_onnx 가능한지 확인
    if hasattr(rknn, 'load_onnx'):
        ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            print("  load_onnx failed")
            rknn.release()
            return None

        # rknnlite에 build 있는지
        if hasattr(rknn, 'build'):
            ret = rknn.build(do_quantization=False)
            if ret != 0:
                print("  build failed")
                rknn.release()
                return None

        ret = rknn.init_runtime(core_mask=7)  # 3 NPU cores
        if ret != 0:
            ret = rknn.init_runtime()
            if ret != 0:
                print("  init_runtime failed")
                rknn.release()
                return None

        A_data = np.random.randn(M, K).astype(np.float32)
        B_data = np.random.randn(K, N).astype(np.float32)

        for _ in range(warmup):
            rknn.inference(inputs=[A_data, B_data])

        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            rknn.inference(inputs=[A_data, B_data])
            times.append((time.perf_counter() - t0) * 1000)

        rknn.release()
        os.remove(onnx_path)
        return float(np.median(times))
    else:
        print("  rknnlite does not support load_onnx (need full rknn-toolkit2)")
        rknn.release()
        return None


def main():
    sizes = [32, 64, 128, 256, 512, 1024]
    reps = 10

    print("=" * 75)
    print("CPU(OpenBLAS 4T) vs NPU(RK3588) GEMM — FP32, M=N=K")
    print("=" * 75)

    results = {}

    # CPU
    print("\n[1/2] CPU (OpenBLAS + NEON, 4 threads)...")
    for S in sizes:
        ms = bench_cpu(S, S, S, reps)
        ops = 2.0 * S * S * S
        gf = ops / (ms / 1000) / 1e9
        results[S] = {'cpu_ms': ms, 'cpu_gflops': gf}
        print("  %4d x %4d : %8.3f ms  (%6.2f GFLOPS)" % (S, S, ms, gf))

    # NPU - try different methods
    print("\n[2/2] NPU (RKNN)...")

    # Method 1: matmul API
    print("  Trying matmul API...", end=" ", flush=True)
    npu_ms = bench_npu_via_rknnlite(sizes[0], sizes[0], sizes[0])
    if npu_ms is not None:
        print("OK! Using matmul API")
        for S in sizes:
            ms = bench_npu_via_rknnlite(S, S, S, reps)
            if ms:
                ops = 2.0 * S * S * S
                gf = ops / (ms / 1000) / 1e9
                results[S]['npu_ms'] = ms
                results[S]['npu_gflops'] = gf
                print("  %4d x %4d : %8.3f ms  (%6.2f GFLOPS)" % (S, S, ms, gf))
    else:
        print("not available")
        # Method 2: load_onnx
        print("  Trying ONNX load...", end=" ", flush=True)
        npu_ms = bench_npu_via_onnx_rknnlite(sizes[0], sizes[0], sizes[0])
        if npu_ms is not None:
            print("OK! Using ONNX load")
            for S in sizes[1:]:
                ms = bench_npu_via_onnx_rknnlite(S, S, S, reps)
                if ms:
                    ops = 2.0 * S * S * S
                    gf = ops / (ms / 1000) / 1e9
                    results[S]['npu_ms'] = ms
                    results[S]['npu_gflops'] = gf
                    print("  %4d x %4d : %8.3f ms  (%6.2f GFLOPS)" % (S, S, ms, gf))
        else:
            print("not available either")
            print("\n  NPU 벤치마크를 위해서는 PC에서 ONNX→RKNN 변환 후 전송 필요")
            print("  또는 rknn_matmul_api C 예제를 빌드하여 실행")

    # Comparison
    print("\n" + "=" * 75)
    has_npu = any('npu_ms' in v for v in results.values())
    if has_npu:
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
                    S, cpu_ms, r.get('cpu_gflops', 0),
                    npu_ms, r.get('npu_gflops', 0), winner))
    else:
        print("NPU 결과 없음 — CPU 결과만 출력")
        for S in sizes:
            r = results.get(S, {})
            print("%-6d | %8.3f ms %8.2f GFLOPS" % (S, r['cpu_ms'], r['cpu_gflops']))

    print("=" * 75)

    # Save
    output_dir = './results/raw/gpu_analysis'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'npu_crossover.json'), 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print("\nSaved to %s/npu_crossover.json" % output_dir)


if __name__ == "__main__":
    main()
