/*
 * CPU vs GPU GEMM Crossover Benchmark
 * FP32 행렬 곱 크기를 키워가며 CPU/GPU 소요 시간 비교
 *
 * Build: gcc -O2 -o gpu_gemm_bench scripts/gpu_gemm_bench.c -lOpenCL -lm
 * Run:   ./gpu_gemm_bench
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ── CPU GEMM (naive, single-thread for fair comparison) ──────────── */
static double bench_cpu(int M, int N, int K, int reps) {
    float *A = malloc(M * K * sizeof(float));
    float *B = malloc(K * N * sizeof(float));
    float *C = malloc(M * N * sizeof(float));
    for (int i = 0; i < M*K; i++) A[i] = 0.5f;
    for (int i = 0; i < K*N; i++) B[i] = 0.5f;

    /* warmup */
    for (int r = 0; r < 2; r++) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++) sum += A[i*K+k] * B[k*N+j];
                C[i*N+j] = sum;
            }
    }

    double t0 = get_time_ms();
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++) sum += A[i*K+k] * B[k*N+j];
                C[i*N+j] = sum;
            }
    }
    double elapsed = (get_time_ms() - t0) / reps;

    free(A); free(B); free(C);
    return elapsed;
}

/* ── GPU GEMM (OpenCL) ────────────────────────────────────────────── */
static const char* kernel_src =
"__kernel void gemm(__global const float* A, __global const float* B,\n"
"                   __global float* C, const int M, const int N, const int K) {\n"
"    int row = get_global_id(0);\n"
"    int col = get_global_id(1);\n"
"    if (row < M && col < N) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < K; k++)\n"
"            sum += A[row * K + k] * B[k * N + col];\n"
"        C[row * N + col] = sum;\n"
"    }\n"
"}\n";

static cl_context ctx = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;

static int init_opencl() {
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) return -1;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) return -1;

    ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);

    cl_program program = clCreateProgramWithSource(ctx, 1, &kernel_src, NULL, &err);
    err = clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build error: %s\n", log);
        free(log);
        return -1;
    }
    kernel = clCreateKernel(program, "gemm", &err);
    clReleaseProgram(program);
    return 0;
}

static double bench_gpu(int M, int N, int K, int reps) {
    cl_int err;
    size_t sA = M * K * sizeof(float);
    size_t sB = K * N * sizeof(float);
    size_t sC = M * N * sizeof(float);

    cl_mem bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sA, NULL, NULL);
    cl_mem bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sB, NULL, NULL);
    cl_mem bufC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sC, NULL, NULL);

    float *hostA = malloc(sA), *hostB = malloc(sB);
    for (int i = 0; i < M*K; i++) hostA[i] = 0.5f;
    for (int i = 0; i < K*N; i++) hostB[i] = 0.5f;

    clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, sA, hostA, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, sB, hostB, 0, NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &M);
    clSetKernelArg(kernel, 4, sizeof(int), &N);
    clSetKernelArg(kernel, 5, sizeof(int), &K);

    size_t global[2] = {M, N};

    /* warmup */
    for (int i = 0; i < 3; i++) {
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    }
    clFinish(queue);

    /* measure */
    double total_ms = 0;
    for (int i = 0; i < reps; i++) {
        cl_event event;
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, &event);
        clFinish(queue);

        cl_ulong t_start, t_end;
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, NULL);
        total_ms += (t_end - t_start) / 1e6;
        clReleaseEvent(event);
    }

    free(hostA); free(hostB);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);

    return total_ms / reps;
}

int main() {
    int sizes[] = {32, 64, 128, 256, 512, 1024, 2048};
    int n = sizeof(sizes) / sizeof(sizes[0]);
    int reps = 5;

    printf("======================================================================\n");
    printf("CPU vs GPU GEMM Crossover Benchmark (FP32, M=N=K)\n");
    printf("CPU: naive single-thread  |  GPU: Mali-G610 OpenCL\n");
    printf("======================================================================\n\n");

    if (init_opencl() != 0) {
        fprintf(stderr, "OpenCL init failed\n");
        return 1;
    }
    printf("OpenCL initialized.\n\n");

    printf("%-6s | %10s %10s | %10s %10s | %s\n",
           "Size", "CPU(ms)", "GFLOPS", "GPU(ms)", "GFLOPS", "Winner");
    printf("----------------------------------------------------------------------\n");

    for (int i = 0; i < n; i++) {
        int S = sizes[i];
        double ops = 2.0 * S * S * S;

        double cpu_ms = bench_cpu(S, S, S, reps);
        double cpu_gf = ops / (cpu_ms / 1000.0) / 1e9;

        double gpu_ms = bench_gpu(S, S, S, reps);
        double gpu_gf = ops / (gpu_ms / 1000.0) / 1e9;

        const char *winner = (gpu_ms < cpu_ms) ? "<< GPU" : "CPU >>";
        double ratio = (gpu_ms < cpu_ms) ? cpu_ms / gpu_ms : gpu_ms / cpu_ms;

        printf("%-6d | %8.3f %8.2f  | %8.3f %8.2f  | %s (%.1fx)\n",
               S, cpu_ms, cpu_gf, gpu_ms, gpu_gf, winner, ratio);
        fflush(stdout);
    }

    printf("======================================================================\n");
    return 0;
}
