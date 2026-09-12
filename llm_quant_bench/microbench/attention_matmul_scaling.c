/*
 * attention_matmul_scaling.c
 *
 * RKNN NPU matmul 성능 특성 측정
 *
 * 목표: Llama 3.2 1B의 attention decode matmul이 context length에 대해
 * 실제로 O(n) compute scaling을 보이는지 검증.
 *
 * Decode step에서의 attention 행렬 크기:
 *   Q [1, hidden=2048] × K^T [hidden=2048, context_len]
 *   → scores [1, context_len]
 *
 * 이걸 RKNN matmul API (M=1, K=hidden, N=context_len)로 직접 실행하고
 * context_len별 실측 latency를 기록한다.
 *
 * Build (RK3588):
 *   gcc -O2 -o attention_matmul_scaling attention_matmul_scaling.c \
 *       -I/home/hyunho.son/install_files/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include \
 *       -L/usr/lib/aarch64-linux-gnu -lrknnrt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>

#include "rknn_api.h"
#include "rknn_matmul_api.h"

#define N_WARMUP 20
#define N_MEASURE 100

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

/*
 * 단일 matmul 설정 (M, K, N) 벤치마크
 * FP16 matmul with FP32 output
 * 반환: median, min, mean, std (us), achieved GFLOPS (best case from min)
 */
static int benchmark_matmul(int32_t M, int32_t K, int32_t N,
                              double* median_us, double* min_us,
                              double* mean_us, double* std_us, double* gflops) {
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    memset(&info, 0, sizeof(info));
    memset(&io_attr, 0, sizeof(io_attr));

    info.M = M;
    info.K = K;
    info.N = N;
    info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    info.B_layout = 0;       // normal layout
    info.B_quant_type = 0;   // per layer
    info.AC_layout = 0;      // normal layout
    info.AC_quant_type = 0;
    info.iommu_domain_id = 0;

    rknn_matmul_ctx ctx;
    int ret = rknn_matmul_create(&ctx, &info, &io_attr);
    if (ret != 0) {
        fprintf(stderr, "[M=%d K=%d N=%d] rknn_matmul_create failed: %d\n",
                M, K, N, ret);
        return ret;
    }

    // Allocate tensor memory
    rknn_tensor_mem* A_mem = rknn_create_mem(ctx, io_attr.A.size);
    rknn_tensor_mem* B_mem = rknn_create_mem(ctx, io_attr.B.size);
    rknn_tensor_mem* C_mem = rknn_create_mem(ctx, io_attr.C.size);

    if (!A_mem || !B_mem || !C_mem) {
        fprintf(stderr, "[M=%d K=%d N=%d] mem alloc failed\n", M, K, N);
        rknn_matmul_destroy(ctx);
        return -1;
    }

    // Fill with dummy data (non-zero, to avoid sparsity optimizations)
    uint16_t* a_data = (uint16_t*)A_mem->virt_addr;
    uint16_t* b_data = (uint16_t*)B_mem->virt_addr;
    for (size_t i = 0; i < io_attr.A.size / 2; i++) a_data[i] = 0x3800;  // fp16 0.5
    for (size_t i = 0; i < io_attr.B.size / 2; i++) b_data[i] = 0x3800;

    rknn_matmul_set_io_mem(ctx, A_mem, &io_attr.A);
    rknn_matmul_set_io_mem(ctx, B_mem, &io_attr.B);
    rknn_matmul_set_io_mem(ctx, C_mem, &io_attr.C);

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        rknn_matmul_run(ctx);
    }

    // Measure
    double times[N_MEASURE];
    for (int i = 0; i < N_MEASURE; i++) {
        double t0 = now_sec();
        rknn_matmul_run(ctx);
        double t1 = now_sec();
        times[i] = (t1 - t0) * 1e6;  // us
    }

    // Statistics (mean, std)
    double sum = 0, sum2 = 0;
    for (int i = 0; i < N_MEASURE; i++) {
        sum += times[i];
        sum2 += times[i] * times[i];
    }
    double mean = sum / N_MEASURE;
    double var = sum2 / N_MEASURE - mean * mean;
    double std = var > 0 ? sqrt(var) : 0;

    // Median and min (robust to outliers)
    double sorted_times[N_MEASURE];
    memcpy(sorted_times, times, sizeof(double) * N_MEASURE);
    qsort(sorted_times, N_MEASURE, sizeof(double), cmp_double);
    double median = sorted_times[N_MEASURE / 2];
    double min_val = sorted_times[0];

    *median_us = median;
    *min_us = min_val;
    *mean_us = mean;
    *std_us = std;
    *gflops = (2.0 * M * K * N) / (min_val * 1e-6) / 1e9;  // best-case GFLOPS from min

    // Cleanup
    rknn_destroy_mem(ctx, A_mem);
    rknn_destroy_mem(ctx, B_mem);
    rknn_destroy_mem(ctx, C_mem);
    rknn_matmul_destroy(ctx);

    return 0;
}

int main(int argc, char** argv) {
    // Llama 3.2 1B attention decode matmul shapes
    // Decode: Q (1 token) × K_cached^T (context_len tokens) for each head
    // In aggregate (multi-head): Q [1, hidden=2048] × K [context, hidden=2048]
    //
    // For decode attention:
    //   Score matmul: (1, head_dim) × (context_len, head_dim) = (1, context_len)
    //   Output matmul: (1, context_len) × (context_len, head_dim) = (1, head_dim)
    //
    // But since llama.cpp/RKLLM likely fuses across heads:
    //   Combined: (1, hidden) × (context, hidden)^T = (1, context)
    //
    // RKNN API: matmul is C = A × B with A=(M,K), B=(K,N), C=(M,N)
    // To do Q × K^T we can use A=(1, hidden), B=(hidden, context) → C=(1, context)

    // K alignment: INT8 needs 32-byte align, FP16 needs 32-byte align on RK3588.
    // hidden=2048 is fine (32-aligned).
    // N (context) alignment: FP16 needs 16-byte align. So use multiples of 16.

    int32_t HIDDEN = 2048;  // Llama 3.2 1B hidden size
    int32_t context_lens[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    int n_ctx_lens = sizeof(context_lens) / sizeof(context_lens[0]);

    printf("=== NPU Attention Matmul Scaling (RK3588, NPU@1GHz) ===\n");
    printf("Shapes: (M=1, K=%d, N=context) FP16×FP16→FP32\n", HIDDEN);
    printf("Warmup=%d, Measure=%d iters\n", N_WARMUP, N_MEASURE);
    printf("%-8s %-10s %-10s %-10s %-10s %-12s %-14s %-14s\n",
           "Context", "Min(us)", "Median", "Mean", "Std", "GFLOPS(min)", "MACs", "@1TOPS(us)");
    printf("----------------------------------------------------------------------------------------\n");

    for (int i = 0; i < n_ctx_lens; i++) {
        int32_t N = context_lens[i];
        double median_us, min_us, mean_us, std_us, gflops;
        int ret = benchmark_matmul(1, HIDDEN, N, &median_us, &min_us, &mean_us, &std_us, &gflops);
        if (ret != 0) {
            printf("%-8d FAILED (ret=%d)\n", N, ret);
            continue;
        }
        double macs = (double)HIDDEN * N;
        double theoretical_us_1tops = macs * 2 / 1e12 * 1e6;
        printf("%-8d %-10.2f %-10.2f %-10.2f %-10.2f %-12.2f %-14.3e %-14.3f\n",
               N, min_us, median_us, mean_us, std_us, gflops, macs, theoretical_us_1tops);
        fflush(stdout);
    }

    printf("\n=== Done ===\n");
    return 0;
}
