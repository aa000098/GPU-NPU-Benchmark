#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *kernel_src =
"__kernel void vec_add(__global const float* a, __global const float* b, __global float* c) {\n"
"    int gid = get_global_id(0);\n"
"    c[gid] = a[gid] + b[gid];\n"
"}\n";

int main(void) {
    cl_int err;

    // 1. 플랫폼 선택 (ARM Platform)
    cl_uint num_platforms = 0;
    cl_platform_id platforms[4];
    err = clGetPlatformIDs(4, platforms, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "clGetPlatformIDs failed: %d\n", err);
        return 1;
    }

    cl_platform_id arm_platform = NULL;
    for (cl_uint i = 0; i < num_platforms; ++i) {
        char name[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, NULL);
        if (strstr(name, "ARM Platform")) {
            arm_platform = platforms[i];
            printf("Using platform: %s\n", name);
            break;
        }
    }
    if (!arm_platform) {
        fprintf(stderr, "ARM Platform not found\n");
        return 1;
    }

    // 2. 디바이스 선택 (GPU 하나)
    cl_device_id dev;
    err = clGetDeviceIDs(arm_platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "clGetDeviceIDs failed: %d\n", err);
        return 1;
    }

    // 3. 컨텍스트 & 큐 생성
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);

    // 4. 데이터 준비
    const int N = 1024;
    size_t bytes = N * sizeof(float);
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(2*i);
    }

    cl_mem d_a = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_a, &err);
    cl_mem d_b = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, h_b, &err);
    cl_mem d_c = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);

    // 5. 프로그램 & 커널 빌드
    const char *srcs[] = { kernel_src };
    size_t lengths[] = { strlen(kernel_src) };
    cl_program prog = clCreateProgramWithSource(ctx, 1, srcs, lengths, &err);

    err = clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // 빌드 로그 출력
        size_t log_size = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char*)malloc(log_size);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Build failed:\n%s\n", log);
        free(log);
        return 1;
    }

    cl_kernel krn = clCreateKernel(prog, "vec_add", &err);

    // 6. 커널 파라미터 설정
    clSetKernelArg(krn, 0, sizeof(cl_mem), &d_a);
    clSetKernelArg(krn, 1, sizeof(cl_mem), &d_b);
    clSetKernelArg(krn, 2, sizeof(cl_mem), &d_c);

    // 7. 커널 실행
    size_t global = N;
    size_t local = 64;  // G610에서 대충 맞는 값, 나중에 튜닝 가능
    err = clEnqueueNDRangeKernel(q, krn, 1, NULL, &global, &local, 0, NULL, NULL);

    clFinish(q);

    // 8. 결과 읽기
    clEnqueueReadBuffer(q, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);

    // 9. 검증
    int ok = 1;
    for (int i = 0; i < 10; ++i) {
        printf("c[%d] = %f (expect %f)\n", i, h_c[i], h_a[i] + h_b[i]);
    }
    for (int i = 0; i < N; ++i) {
        float exp = h_a[i] + h_b[i];
        if (h_c[i] != exp) {
            ok = 0;
            break;
        }
    }
    printf("Result: %s\n", ok ? "OK" : "MISMATCH");

    // 10. 정리
    clReleaseKernel(krn);
    clReleaseProgram(prog);
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    free(h_a); free(h_b); free(h_c);
    return ok ? 0 : 1;
}

