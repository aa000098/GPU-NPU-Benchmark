/*
 * gqa_matmul_shapes.c
 *
 * Llama 3.2 1B의 GQA attention matmul의 여러 가능한 shape를 측정.
 *
 * Llama 3.2 1B 구성:
 *   - hidden = 2048
 *   - num_attention_heads = 32
 *   - num_key_value_heads = 8 (GQA)
 *   - head_dim = 64
 *
 * 가능한 decode attention matmul shape (per layer, per token):
 *   A. (1, 2048) × (2048, context): 단순 확장 (Q와 KV projection을 같은 dim으로)
 *   B. (1, 512) × (512, context): K/V는 n_kv_heads×head_dim=512만 사용 (GQA)
 *   C. (1, 64) × (64, context) × 32 heads: per-head matmul
 *
 * RKLLM이 실제 어떤 shape로 dispatch 하는지 직접 측정할 수 없지만,
 * 이 3가지 shape의 NPU latency를 비교하면 어떤 것이 실측 9.26 µs/tok slope를 더 잘 설명하는지 확인.
 *
 * Build:
 *   gcc -O2 -o gqa_matmul_shapes gqa_matmul_shapes.c -I<include> -lrknnrt -lm
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

static int benchmark_matmul(int32_t M, int32_t K, int32_t N,
                              double* min_us, double* median_us) {
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    memset(&info, 0, sizeof(info));
    memset(&io_attr, 0, sizeof(io_attr));
    info.M = M; info.K = K; info.N = N;
    info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;

    rknn_matmul_ctx ctx;
    int ret = rknn_matmul_create(&ctx, &info, &io_attr);
    if (ret != 0) {
        *min_us = *median_us = -1;
        return ret;
    }

    rknn_tensor_mem* A_mem = rknn_create_mem(ctx, io_attr.A.size);
    rknn_tensor_mem* B_mem = rknn_create_mem(ctx, io_attr.B.size);
    rknn_tensor_mem* C_mem = rknn_create_mem(ctx, io_attr.C.size);
    if (!A_mem || !B_mem || !C_mem) {
        rknn_matmul_destroy(ctx);
        *min_us = *median_us = -1;
        return -1;
    }

    // Fill with fp16 0.5
    uint16_t* a = A_mem->virt_addr;
    uint16_t* b = B_mem->virt_addr;
    for (size_t i = 0; i < io_attr.A.size / 2; i++) a[i] = 0x3800;
    for (size_t i = 0; i < io_attr.B.size / 2; i++) b[i] = 0x3800;

    rknn_matmul_set_io_mem(ctx, A_mem, &io_attr.A);
    rknn_matmul_set_io_mem(ctx, B_mem, &io_attr.B);
    rknn_matmul_set_io_mem(ctx, C_mem, &io_attr.C);

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) rknn_matmul_run(ctx);

    // Measure
    double times[N_MEASURE];
    for (int i = 0; i < N_MEASURE; i++) {
        double t0 = now_sec();
        rknn_matmul_run(ctx);
        double t1 = now_sec();
        times[i] = (t1 - t0) * 1e6;
    }

    qsort(times, N_MEASURE, sizeof(double), cmp_double);
    *min_us = times[0];
    *median_us = times[N_MEASURE / 2];

    rknn_destroy_mem(ctx, A_mem);
    rknn_destroy_mem(ctx, B_mem);
    rknn_destroy_mem(ctx, C_mem);
    rknn_matmul_destroy(ctx);
    return 0;
}

int main(int argc, char** argv) {
    // GQA: 32 attention heads share 8 KV heads, head_dim=64
    // hidden = 2048, kv_total_dim = 512

    int32_t context_lens[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    int n_ctx = sizeof(context_lens) / sizeof(context_lens[0]);

    // Shape A: K=hidden=2048
    // Shape B: K=kv_dim=512
    // Shape C: K=head_dim=64 (per-head)
    struct { const char* name; int32_t K; int heads_factor; } shapes[] = {
        { "A: K=2048 (hidden, full)", 2048, 1 },
        { "B: K=512  (GQA KV dim)",   512,  1 },
        { "C: K=64   (per-head)",      64,  32 },  // 32 heads would call this 32x per layer
    };
    int n_shapes = sizeof(shapes) / sizeof(shapes[0]);

    for (int s = 0; s < n_shapes; s++) {
        printf("\n=== Shape %s ===\n", shapes[s].name);
        printf("%-8s %-12s %-12s %-14s %-14s\n",
               "Context", "Min(us)", "Median(us)", "GFLOPS(min)", "us/tok");
        printf("---------------------------------------------------------------\n");
        double prev_min = 0;
        int prev_ctx = 0;
        for (int i = 0; i < n_ctx; i++) {
            int32_t N = context_lens[i];
            double min_us, median_us;
            int ret = benchmark_matmul(1, shapes[s].K, N, &min_us, &median_us);
            if (ret != 0) {
                printf("%-8d FAILED (ret=%d)\n", N, ret);
                continue;
            }
            double gflops = (2.0 * shapes[s].K * N) / (min_us * 1e-6) / 1e9;
            double slope = prev_ctx ? (min_us - prev_min) / (N - prev_ctx) : 0;
            printf("%-8d %-12.2f %-12.2f %-14.2f %-14.4f\n",
                   N, min_us, median_us, gflops, slope);
            prev_min = min_us;
            prev_ctx = N;
            fflush(stdout);
        }
    }

    // Per-token slope extrapolation
    printf("\n=== Decode-token Slope Prediction (Llama 3.2 1B, 16 layers) ===\n");
    printf("Assumes 2 attention matmuls per layer (QK^T + AV)\n\n");
    printf("  Shape A: 32 matmul/tok × slope_A = ? µs/tok\n");
    printf("  Shape B: 32 matmul/tok × slope_B = ? µs/tok\n");
    printf("  Shape C: 32 × 32 heads × slope_C = ? µs/tok (if per-head)\n\n");
    printf("Our original measured NPU decode slope: 9.26 µs/tok\n");
    printf("Run the Python analyzer for proper linear fit and matching.\n");

    return 0;
}
