/*
 * GPU(OpenCL) GEMM Only Benchmark
 * Build: gcc -O2 -o gpu_only_bench scripts/gpu_only_bench.c -lOpenCL -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

#define WARMUP 10
#define REPS   10

static int cmp_double(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}
static double median(double *arr, int n) {
    qsort(arr, n, sizeof(double), cmp_double);
    return (n % 2) ? arr[n/2] : (arr[n/2-1] + arr[n/2]) / 2.0;
}

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

int main() {
    int sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    int n = sizeof(sizes) / sizeof(sizes[0]);

    cl_platform_id plat; cl_device_id dev; cl_int err;
    clGetPlatformIDs(1, &plat, NULL);
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &kernel_src, NULL, &err);
    clBuildProgram(prog, 1, &dev, "-cl-fast-relaxed-math", NULL, NULL);
    cl_kernel kernel = clCreateKernel(prog, "gemm", &err);

    printf("GPU Only GEMM (Mali-G610 OpenCL, FP32, warmup=%d, reps=%d, median)\n\n", WARMUP, REPS);
    printf("%-6s | %10s | %10s\n", "Size", "GPU(ms)", "GFLOPS");
    printf("--------------------------------------\n");

    for (int i = 0; i < n; i++) {
        int S = sizes[i];
        size_t sz = (size_t)S * S * sizeof(float);
        double ops = 2.0 * S * S * S;

        cl_mem bA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sz, NULL, NULL);
        cl_mem bB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sz, NULL, NULL);
        cl_mem bC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sz, NULL, NULL);

        float *h = malloc(sz);
        for (int j = 0; j < S*S; j++) h[j] = 0.5f;
        clEnqueueWriteBuffer(queue, bA, CL_TRUE, 0, sz, h, 0, NULL, NULL);
        clEnqueueWriteBuffer(queue, bB, CL_TRUE, 0, sz, h, 0, NULL, NULL);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &bA);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &bB);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &bC);
        clSetKernelArg(kernel, 3, sizeof(int), &S);
        size_t global[2] = {S, S};

        for (int w = 0; w < WARMUP; w++) {
            clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
        }
        clFinish(queue);

        double times[REPS];
        for (int r = 0; r < REPS; r++) {
            cl_event ev;
            clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, &ev);
            clFinish(queue);
            cl_ulong t0, t1;
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
            clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL);
            times[r] = (t1 - t0) / 1e6;
            clReleaseEvent(ev);
        }

        double ms = median(times, REPS);
        double gf = ops / (ms / 1000.0) / 1e9;
        printf("%-6d | %8.3f   | %8.2f\n", S, ms, gf);
        fflush(stdout);

        free(h);
        clReleaseMemObject(bA); clReleaseMemObject(bB); clReleaseMemObject(bC);
    }

    printf("--------------------------------------\n");
    return 0;
}
