"""
CPU vs GPU 성능 역전 구간 탐색
행렬 곱 (GEMM) 크기를 키워가며 CPU/GPU 소요 시간 비교.
GPU가 이기는 행렬 크기(교차점)를 찾는다.

Usage: python scripts/cpu_vs_gpu_crossover.py
"""
import numpy as np
import time
import subprocess
import os
import json

# MNN Python API 사용 가능한지 확인
try:
    import MNN
    import MNN.numpy as mp
    import MNN.expr as expr
    HAS_MNN = True
except ImportError:
    HAS_MNN = False

def benchmark_numpy_cpu(M, N, K, reps=10):
    """CPU GEMM benchmark using numpy (OpenBLAS/NEON)."""
    A = np.random.randn(M, K).astype(np.float16)
    B = np.random.randn(K, N).astype(np.float16)

    # warmup
    for _ in range(3):
        np.matmul(A, B)

    times = []
    for _ in range(reps):
        start = time.perf_counter()
        np.matmul(A, B)
        end = time.perf_counter()
        times.append(end - start)

    avg_ms = np.mean(times) * 1000
    gflops = (2.0 * M * N * K) / (avg_ms / 1000) / 1e9
    return avg_ms, gflops


def write_opencl_benchmark():
    """Generate and compile OpenCL GEMM benchmark."""
    code = r'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>

const char* kernel_src =
"__kernel void gemm(__global const half* A, __global const half* B, __global half* C,\n"
"                   const int M, const int N, const int K) {\n"
"    int row = get_global_id(0);\n"
"    int col = get_global_id(1);\n"
"    if (row < M && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < K; k++) {\n"
"            sum += vload_half(row * K + k, A) * vload_half(k * N + col, B);\n"
"        }\n"
"        vstore_half(sum, row * N + col, C);\n"
"    }\n"
"}\n";

double benchmark_gpu(int M, int N, int K, int reps) {
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);

    cl_program program = clCreateProgramWithSource(ctx, 1, &kernel_src, NULL, &err);
    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build error: %s\n", log);
        free(log);
        return -1;
    }
    cl_kernel kernel = clCreateKernel(program, "gemm", &err);

    size_t size_A = M * K * sizeof(cl_half);
    size_t size_B = K * N * sizeof(cl_half);
    size_t size_C = M * N * sizeof(cl_half);

    cl_mem bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, size_A, NULL, NULL);
    cl_mem bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, size_B, NULL, NULL);
    cl_mem bufC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, size_C, NULL, NULL);

    // Init with random data
    cl_half* hostA = (cl_half*)malloc(size_A);
    cl_half* hostB = (cl_half*)malloc(size_B);
    for (int i = 0; i < M*K; i++) hostA[i] = 0x3C00; // 1.0 in fp16
    for (int i = 0; i < K*N; i++) hostB[i] = 0x3C00;

    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, size_A, hostA, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, size_B, hostB, 0, NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &M);
    clSetKernelArg(kernel, 4, sizeof(int), &N);
    clSetKernelArg(kernel, 5, sizeof(int), &K);

    size_t global[2] = {M, N};

    // Warmup
    for (int i = 0; i < 3; i++) {
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    }
    clFinish(queue);

    // Measure
    double total_ms = 0;
    for (int i = 0; i < reps; i++) {
        cl_event event;
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, &event);
        clFinish(queue);

        cl_ulong start, end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
        total_ms += (end - start) / 1e6;
        clReleaseEvent(event);
    }

    double avg_ms = total_ms / reps;

    free(hostA);
    free(hostB);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return avg_ms;
}

int main(int argc, char** argv) {
    int sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int reps = 5;

    printf("M=N=K,gpu_ms,gpu_gflops\n");
    for (int i = 0; i < n_sizes; i++) {
        int S = sizes[i];
        double ms = benchmark_gpu(S, S, S, reps);
        if (ms < 0) continue;
        double gflops = (2.0 * S * S * S) / (ms / 1000.0) / 1e9;
        printf("%d,%.3f,%.2f\n", S, ms, gflops);
        fflush(stdout);
    }
    return 0;
}
'''
    src_path = '/tmp/gpu_gemm_bench.c'
    bin_path = '/tmp/gpu_gemm_bench'
    with open(src_path, 'w') as f:
        f.write(code)

    ret = os.system('gcc -O2 -o %s %s -lOpenCL -lm 2>&1' % (bin_path, src_path))
    if ret != 0:
        print("OpenCL compilation failed")
        return None
    return bin_path


def main():
    sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    reps = 5
    results = []

    print("=" * 70)
    print("CPU vs GPU GEMM Benchmark (FP16, M=N=K)")
    print("=" * 70)

    # CPU benchmark
    print("\n--- CPU (numpy FP16) ---")
    cpu_results = {}
    for S in sizes:
        try:
            ms, gflops = benchmark_numpy_cpu(S, S, S, reps)
            cpu_results[S] = {'ms': ms, 'gflops': gflops}
            print("  %4d x %4d x %4d : %8.3f ms  (%6.2f GFLOPS)" % (S, S, S, ms, gflops))
        except Exception as e:
            print("  %4d: FAILED (%s)" % (S, e))

    # GPU benchmark
    print("\n--- GPU (OpenCL FP16) ---")
    gpu_bin = write_opencl_benchmark()
    gpu_results = {}
    if gpu_bin:
        result = subprocess.run([gpu_bin], capture_output=True, text=True, timeout=300)
        for line in result.stdout.strip().split('\n'):
            if line.startswith('M='):
                continue
            parts = line.split(',')
            if len(parts) == 3:
                S = int(parts[0])
                ms = float(parts[1])
                gflops = float(parts[2])
                gpu_results[S] = {'ms': ms, 'gflops': gflops}
                print("  %4d x %4d x %4d : %8.3f ms  (%6.2f GFLOPS)" % (S, S, S, ms, gflops))

    # Comparison
    print("\n" + "=" * 70)
    print("%-6s | %10s | %10s | %10s | %10s | %s" % (
        "Size", "CPU (ms)", "GPU (ms)", "CPU GFLOPS", "GPU GFLOPS", "Winner"))
    print("-" * 70)

    crossover = None
    for S in sizes:
        cpu = cpu_results.get(S)
        gpu = gpu_results.get(S)
        if cpu and gpu:
            winner = "GPU" if gpu['ms'] < cpu['ms'] else "CPU"
            if winner == "GPU" and crossover is None:
                crossover = S
            ratio = cpu['ms'] / gpu['ms'] if gpu['ms'] > 0 else 0
            print("%-6d | %8.3f ms | %8.3f ms | %8.2f   | %8.2f   | %s (%.1fx)" % (
                S, cpu['ms'], gpu['ms'], cpu['gflops'], gpu['gflops'],
                winner, max(ratio, 1/ratio) if ratio > 0 else 0))

    print("=" * 70)
    if crossover:
        print("GPU가 이기는 교차점: M=N=K=%d 이상" % crossover)
    else:
        print("테스트 범위(~4096)에서 GPU가 이기는 구간 없음")

    # Save results
    output_dir = './results/raw/gpu_analysis'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'crossover_results.json'), 'w') as f:
        json.dump({'cpu': {str(k): v for k, v in cpu_results.items()},
                   'gpu': {str(k): v for k, v in gpu_results.items()},
                   'crossover': crossover}, f, indent=2)
    print("Results saved to %s/crossover_results.json" % output_dir)


if __name__ == "__main__":
    main()
