"""
CPU vs GPU 공정 비교 GEMM Benchmark
CPU: numpy (OpenBLAS + NEON SIMD, multi-thread) — MNN과 동일 수준
GPU: OpenCL naive kernel (Mali-G610)

Usage: python3 scripts/crossover_fair.py
"""
import numpy as np
import subprocess
import time
import os
import json

def bench_cpu_numpy(M, N, K, reps=5):
    """CPU GEMM: OpenBLAS + NEON + multi-thread (MNN과 유사 환경)."""
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    # warmup
    for _ in range(3):
        np.matmul(A, B)

    times = []
    for _ in range(reps):
        start = time.perf_counter()
        np.matmul(A, B)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_ms = np.mean(times)
    gflops = (2.0 * M * N * K) / (avg_ms / 1000) / 1e9
    return avg_ms, gflops


def bench_gpu_opencl(sizes, reps=5):
    """GPU GEMM via compiled OpenCL binary."""
    bin_path = './gpu_gemm_bench'
    if not os.path.exists(bin_path):
        print("ERROR: gpu_gemm_bench not found. Compile first:")
        print("  gcc -O2 -o gpu_gemm_bench scripts/gpu_gemm_bench.c -lOpenCL -lm")
        return {}

    result = subprocess.run([bin_path], capture_output=True, text=True, timeout=600)
    gpu_results = {}
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('=') or line.startswith('-') or line.startswith('Size') \
           or line.startswith('CPU') or line.startswith('Open') or line.startswith('arm'):
            continue
        parts = line.split('|')
        if len(parts) >= 4:
            try:
                size = int(parts[0].strip())
                gpu_part = parts[2].strip().split()
                gpu_ms = float(gpu_part[0])
                gpu_gf = float(gpu_part[1])
                gpu_results[size] = {'ms': gpu_ms, 'gflops': gpu_gf}
            except (ValueError, IndexError):
                continue
    return gpu_results


def main():
    os.environ['OPENBLAS_NUM_THREADS'] = '4'  # MNN과 동일하게 4스레드

    sizes = [32, 64, 128, 256, 512, 1024, 2048]
    reps = 5

    print("=" * 75)
    print("CPU vs GPU GEMM — 공정 비교 (FP32, M=N=K)")
    print("CPU: OpenBLAS + NEON + 4 threads  |  GPU: Mali-G610 OpenCL")
    print("=" * 75)

    # CPU benchmark
    print("\n[1/2] CPU (OpenBLAS, 4 threads) benchmarking...")
    cpu_results = {}
    for S in sizes:
        try:
            ms, gflops = bench_cpu_numpy(S, S, S, reps)
            cpu_results[S] = {'ms': ms, 'gflops': gflops}
            print("  %4d x %4d : %8.3f ms  (%6.2f GFLOPS)" % (S, S, ms, gflops))
        except Exception as e:
            print("  %4d: FAILED (%s)" % (S, e))

    # GPU benchmark
    print("\n[2/2] GPU (OpenCL) benchmarking...")
    gpu_results = bench_gpu_opencl(sizes, reps)
    for S in sorted(gpu_results.keys()):
        r = gpu_results[S]
        print("  %4d x %4d : %8.3f ms  (%6.2f GFLOPS)" % (S, S, r['ms'], r['gflops']))

    # Comparison
    print("\n" + "=" * 75)
    print("%-6s | %10s %10s | %10s %10s | %s" % (
        "Size", "CPU(ms)", "GFLOPS", "GPU(ms)", "GFLOPS", "Winner"))
    print("-" * 75)

    crossover = None
    for S in sizes:
        cpu = cpu_results.get(S)
        gpu = gpu_results.get(S)
        if cpu and gpu:
            if gpu['ms'] < cpu['ms']:
                winner = "<< GPU"
                ratio = cpu['ms'] / gpu['ms']
                if crossover is None:
                    crossover = S
            else:
                winner = "CPU >>"
                ratio = gpu['ms'] / cpu['ms']
            print("%-6d | %8.3f %8.2f  | %8.3f %8.2f  | %s (%.1fx)" % (
                S, cpu['ms'], cpu['gflops'], gpu['ms'], gpu['gflops'], winner, ratio))
        elif cpu:
            print("%-6d | %8.3f %8.2f  |      N/A       N/A  |" % (
                S, cpu['ms'], cpu['gflops']))

    print("=" * 75)
    if crossover:
        print("\nGPU 역전 교차점: M=N=K=%d 이상" % crossover)
    else:
        print("\n테스트 범위에서 GPU가 이기는 구간 없음")

    print("\n참고: LLM decode는 M=1 (행렬×벡터)이므로 GPU가 이길 수 없는 영역")
    print("      LLM prefill은 M=토큰수이나, fallback + 메모리 복사 오버헤드로 GPU 불리")

    # Save
    output_dir = './results/raw/gpu_analysis'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'crossover_fair.json'), 'w') as f:
        json.dump({
            'cpu_backend': 'OpenBLAS + NEON, 4 threads',
            'gpu_backend': 'Mali-G610 OpenCL',
            'cpu': {str(k): v for k, v in cpu_results.items()},
            'gpu': {str(k): v for k, v in gpu_results.items()},
            'crossover': crossover
        }, f, indent=2)
    print("\nResults saved to %s/crossover_fair.json" % output_dir)


if __name__ == "__main__":
    main()
