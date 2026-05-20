/*
 * CPU(OpenBLAS) vs GPU(OpenCL) GEMM 공정 비교
 * 둘 다 같은 프로그램에서, 같은 행렬 크기로, 동일 횟수 측정
 *
 * Build: gcc -O2 -o crossover_final scripts/crossover_final.c -lOpenCL -lopenblas -lm
 * Run:   OPENBLAS_NUM_THREADS=4 ./crossover_final
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cblas.h>
#include <CL/cl.h>

#define WARMUP 10
#define REPS   10

static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

static double median(double *arr, int n) {
    qsort(arr, n, sizeof(double), cmp_double);
    return (n % 2) ? arr[n/2] : (arr[n/2-1] + arr[n/2]) / 2.0;
}

/* ── CPU GEMM (OpenBLAS, NEON, multi-thread) ──────────────────────── */
static double bench_cpu(int S) {
    float *A = malloc(S * S * sizeof(float));
    float *B = malloc(S * S * sizeof(float));
    float *C = malloc(S * S * sizeof(float));
    for (int i = 0; i < S*S; i++) { A[i] = 0.5f; B[i] = 0.5f; C[i] = 0.0f; }

    /* warmup */
    for (int i = 0; i < WARMUP; i++)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    S, S, S, 1.0f, A, S, B, S, 0.0f, C, S);

    /* measure */
    double times[REPS];
    for (int i = 0; i < REPS; i++) {
        double t0 = get_time_ms();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    S, S, S, 1.0f, A, S, B, S, 0.0f, C, S);
        times[i] = get_time_ms() - t0;
    }

    free(A); free(B); free(C);
    return median(times, REPS);
}

/* ── GPU GEMM (OpenCL) ────────────────────────────────────────────── */
static const char* kernel_src =
"__kernel void gemm(__global const float* A, __global const float* B,\n"
"                   __global float* C, const int S) {\n"
"    int row = get_global_id(0);\n"
"    int col = get_global_id(1);\n"
"    if (row < S && col < S) {\n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < S; k++)\n"
"            sum += A[row * S + k] * B[k * S + col];\n"
"        C[row * S + col] = sum;\n"
"    }\n"
"}\n";

static cl_context ctx;
static cl_command_queue queue;
static cl_kernel kernel;

static int init_opencl() {
    cl_platform_id plat; cl_device_id dev; cl_int err;
    clGetPlatformIDs(1, &plat, NULL);
    err = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err != CL_SUCCESS) return -1;
    ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    queue = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernel_src, NULL, &err);
    err = clBuildProgram(prog, 1, &dev, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t sz; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
        char *log = malloc(sz); clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sz, log, NULL);
        fprintf(stderr, "Build error: %s\n", log); free(log); return -1;
    }
    kernel = clCreateKernel(prog, "gemm", &err);
    clReleaseProgram(prog);
    return 0;
}

static double bench_gpu(int S) {
    size_t sz = S * S * sizeof(float);
    cl_mem bA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sz, NULL, NULL);
    cl_mem bB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sz, NULL, NULL);
    cl_mem bC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sz, NULL, NULL);

    float *h = malloc(sz);
    for (int i = 0; i < S*S; i++) h[i] = 0.5f;
    clEnqueueWriteBuffer(queue, bA, CL_TRUE, 0, sz, h, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bB, CL_TRUE, 0, sz, h, 0, NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bC);
    clSetKernelArg(kernel, 3, sizeof(int), &S);
    size_t global[2] = {S, S};

    /* warmup */
    for (int i = 0; i < WARMUP; i++)
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
    clFinish(queue);

    /* measure */
    double times[REPS];
    for (int i = 0; i < REPS; i++) {
        cl_event ev;
        clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, &ev);
        clFinish(queue);
        cl_ulong t0, t1;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL);
        times[i] = (t1 - t0) / 1e6;
        clReleaseEvent(ev);
    }

    free(h);
    clReleaseMemObject(bA); clReleaseMemObject(bB); clReleaseMemObject(bC);
    return median(times, REPS);
}

int main() {
    int sizes[] = {32, 64, 128, 256, 512, 1024, 2048};
    int n = sizeof(sizes) / sizeof(sizes[0]);

    printf("==========================================================================\n");
    printf("CPU(OpenBLAS 4T NEON) vs GPU(Mali-G610 OpenCL) GEMM — FP32, M=N=K\n");
    printf("Warmup: %d | Measure: %d (median)\n", WARMUP, REPS);
    printf("==========================================================================\n\n");

    if (init_opencl() != 0) { fprintf(stderr, "OpenCL init failed\n"); return 1; }

    printf("%-6s | %10s %10s | %10s %10s | %s\n",
           "Size", "CPU(ms)", "GFLOPS", "GPU(ms)", "GFLOPS", "Winner");
    printf("--------------------------------------------------------------------------\n");

    for (int i = 0; i < n; i++) {
        int S = sizes[i];
        double ops = 2.0 * S * S * S;

        double cpu_ms = bench_cpu(S);
        double cpu_gf = ops / (cpu_ms / 1000.0) / 1e9;

        double gpu_ms = bench_gpu(S);
        double gpu_gf = ops / (gpu_ms / 1000.0) / 1e9;

        const char *winner;
        double ratio;
        if (gpu_ms < cpu_ms) {
            winner = "<< GPU";
            ratio = cpu_ms / gpu_ms;
        } else {
            winner = "CPU >>";
            ratio = gpu_ms / cpu_ms;
        }

        printf("%-6d | %8.3f %8.2f  | %8.3f %8.2f  | %s (%.1fx)\n",
               S, cpu_ms, cpu_gf, gpu_ms, gpu_gf, winner, ratio);
        fflush(stdout);
    }

    printf("==========================================================================\n");
    return 0;
}
