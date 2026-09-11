/*
 * dispatch_overhead.c
 *
 * NPU matmul dispatch overhead isolation.
 *
 * Strategy:
 *   1. Fixed small matmul (K=64, N=1) → compute ≈ 0, remainder = pure dispatch overhead
 *   2. Same matmul repeated 1000× consecutively → amortized vs per-call overhead
 *   3. Various K (16, 32, 64, 128) with N=1 to see how small we can go
 *
 * Goal: Measure per-matmul dispatch cost (ioctl + cache sync + kernel launch).
 * This isolates the "fixed overhead" from the context-proportional compute cost.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>

#include "rknn_api.h"
#include "rknn_matmul_api.h"

#define N_WARMUP 50
#define N_MEASURE 1000

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}

static int benchmark(int32_t M, int32_t K, int32_t N, int n_calls,
                       double* min_us, double* median_us, double* mean_us) {
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    memset(&info, 0, sizeof(info));
    memset(&io_attr, 0, sizeof(io_attr));
    info.M = M; info.K = K; info.N = N;
    info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;

    rknn_matmul_ctx ctx;
    int ret = rknn_matmul_create(&ctx, &info, &io_attr);
    if (ret != 0) return ret;

    rknn_tensor_mem* A_mem = rknn_create_mem(ctx, io_attr.A.size);
    rknn_tensor_mem* B_mem = rknn_create_mem(ctx, io_attr.B.size);
    rknn_tensor_mem* C_mem = rknn_create_mem(ctx, io_attr.C.size);
    if (!A_mem || !B_mem || !C_mem) { rknn_matmul_destroy(ctx); return -1; }

    uint16_t* a = A_mem->virt_addr;
    uint16_t* b = B_mem->virt_addr;
    for (size_t i = 0; i < io_attr.A.size / 2; i++) a[i] = 0x3800;
    for (size_t i = 0; i < io_attr.B.size / 2; i++) b[i] = 0x3800;

    rknn_matmul_set_io_mem(ctx, A_mem, &io_attr.A);
    rknn_matmul_set_io_mem(ctx, B_mem, &io_attr.B);
    rknn_matmul_set_io_mem(ctx, C_mem, &io_attr.C);

    for (int i = 0; i < N_WARMUP; i++) rknn_matmul_run(ctx);

    double* times = malloc(sizeof(double) * n_calls);
    for (int i = 0; i < n_calls; i++) {
        double t0 = now_sec();
        rknn_matmul_run(ctx);
        double t1 = now_sec();
        times[i] = (t1 - t0) * 1e6;
    }

    double sum = 0;
    for (int i = 0; i < n_calls; i++) sum += times[i];
    *mean_us = sum / n_calls;

    qsort(times, n_calls, sizeof(double), cmp_double);
    *min_us = times[0];
    *median_us = times[n_calls / 2];

    free(times);
    rknn_destroy_mem(ctx, A_mem);
    rknn_destroy_mem(ctx, B_mem);
    rknn_destroy_mem(ctx, C_mem);
    rknn_matmul_destroy(ctx);
    return 0;
}

int main(void) {
    /* Minimum possible matmul: M=1, K=32, N=16 (alignment constraints on RK3588)
     * Fp16 needs K 32-aligned, N 16-aligned
     * K × N MACs in this matmul = 32 × 16 = 512
     * At 1 TOPS sustained: 1 µs for compute only
     * Remainder = pure dispatch */
    struct { const char* label; int32_t K; int32_t N; } cases[] = {
        { "Minimum matmul (32×16)",       32,   16   },
        { "Small (64×32)",                  64,   32   },
        { "Medium (128×64)",                128,  64   },
        { "Medium (512×64)",                512,  64   },
        { "Full K (2048×32)",               2048, 32   },
        { "Full K (2048×128)",              2048, 128  },
        { "Full K (2048×512)",              2048, 512  },
        { "Full K, N=4096 (same as ctx)",   2048, 4096 },
    };
    int n_cases = sizeof(cases) / sizeof(cases[0]);

    printf("=== NPU Dispatch Overhead Isolation (RK3588, NPU@1GHz) ===\n");
    printf("Same matmul called %d times; measure per-call latency\n\n", N_MEASURE);
    printf("%-30s %-10s %-10s %-10s %-10s %-12s %-12s\n",
           "Shape", "Min(us)", "Median", "Mean", "MACs", "compute(us)", "dispatch(us)");
    printf("--------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < n_cases; i++) {
        double min_us, median_us, mean_us;
        int ret = benchmark(1, cases[i].K, cases[i].N, N_MEASURE,
                              &min_us, &median_us, &mean_us);
        if (ret != 0) {
            printf("%-30s FAILED ret=%d\n", cases[i].label, ret);
            continue;
        }
        double macs = (double)cases[i].K * cases[i].N;
        // At sustained 11 GFLOPS (from Phase 4): compute_us = 2*macs / 11e9 * 1e6
        double compute_us = 2 * macs / 11e9 * 1e6;
        double dispatch_us = min_us - compute_us;
        if (dispatch_us < 0) dispatch_us = 0;
        printf("%-30s %-10.2f %-10.2f %-10.2f %-10.1e %-12.3f %-12.3f\n",
               cases[i].label, min_us, median_us, mean_us, macs, compute_us, dispatch_us);
        fflush(stdout);
    }

    return 0;
}
