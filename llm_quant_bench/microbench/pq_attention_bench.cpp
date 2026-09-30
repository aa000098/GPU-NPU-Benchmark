/* Microbenchmark for PQ-attention on RK3588 Cortex-A76.
 *
 * Compares:
 *   1) attention_fp16_baseline : standard NEON fp16 attention
 *   2) pq_attention_naive       : C scalar reference (correctness check)
 *   3) pq_attention_neon        : NEON-optimized PQ attention
 *
 * Reports: wall time per call, max abs diff vs naive, decode tok/s estimate.
 */
#include "pq_attention.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <cmath>

#define H_HEADS 8
#define HEAD_DIM 64
#define N_SUB 4
#define SUB_DIM 16
#define KB 256

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

template <typename Fn>
double bench(Fn fn, int n_iter = 5) {
    /* warmup */
    fn();
    double t0 = now_ms();
    for (int i = 0; i < n_iter; i++) fn();
    double dt = now_ms() - t0;
    return dt / n_iter;
}

int main(int argc, char *argv[]) {
    int N = (argc > 1) ? std::atoi(argv[1]) : 3500;
    int n_iter = (argc > 2) ? std::atoi(argv[2]) : 5;
    printf("PQ-attention microbench: N=%d ctx, %d iters\n\n", N, n_iter);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist_f(-1.0f, 1.0f);
    std::uniform_int_distribution<int> dist_i(0, KB - 1);

    /* Allocate inputs */
    __fp16 *q       = (__fp16*)aligned_alloc(64, sizeof(__fp16) * H_HEADS * HEAD_DIM);
    __fp16 *cb_K    = (__fp16*)aligned_alloc(64, sizeof(__fp16) * N_SUB * KB * SUB_DIM);
    __fp16 *cb_V    = (__fp16*)aligned_alloc(64, sizeof(__fp16) * N_SUB * KB * SUB_DIM);
    uint8_t  *k_idx   = (uint8_t* )aligned_alloc(64, sizeof(uint8_t)  * N * H_HEADS * N_SUB);
    uint8_t  *v_idx   = (uint8_t* )aligned_alloc(64, sizeof(uint8_t)  * N * H_HEADS * N_SUB);
    __fp16 *k_full  = (__fp16*)aligned_alloc(64, sizeof(__fp16) * N * H_HEADS * HEAD_DIM);
    __fp16 *v_full  = (__fp16*)aligned_alloc(64, sizeof(__fp16) * N * H_HEADS * HEAD_DIM);
    __fp16 *out_pq_naive = (__fp16*)aligned_alloc(64, sizeof(__fp16) * H_HEADS * HEAD_DIM);
    __fp16 *out_pq_neon  = (__fp16*)aligned_alloc(64, sizeof(__fp16) * H_HEADS * HEAD_DIM);
    __fp16 *out_fp16     = (__fp16*)aligned_alloc(64, sizeof(__fp16) * H_HEADS * HEAD_DIM);

    /* Random init */
    for (int i = 0; i < H_HEADS * HEAD_DIM; i++) q[i] = (__fp16)dist_f(gen);
    for (int i = 0; i < N_SUB * KB * SUB_DIM; i++) {
        cb_K[i] = (__fp16)dist_f(gen);
        cb_V[i] = (__fp16)dist_f(gen);
    }
    for (int i = 0; i < N * H_HEADS * N_SUB; i++) {
        k_idx[i] = (uint8_t)dist_i(gen);
        v_idx[i] = (uint8_t)dist_i(gen);
    }
    /* Reconstruct K_full / V_full from indices for fp16 baseline */
    for (int t = 0; t < N; t++) {
        for (int h = 0; h < H_HEADS; h++) {
            for (int s = 0; s < N_SUB; s++) {
                uint8_t kk = k_idx[(t * H_HEADS + h) * N_SUB + s];
                uint8_t vv = v_idx[(t * H_HEADS + h) * N_SUB + s];
                memcpy(k_full + (t * H_HEADS + h) * HEAD_DIM + s * SUB_DIM,
                       cb_K + (s * KB + kk) * SUB_DIM, SUB_DIM * sizeof(__fp16));
                memcpy(v_full + (t * H_HEADS + h) * HEAD_DIM + s * SUB_DIM,
                       cb_V + (s * KB + vv) * SUB_DIM, SUB_DIM * sizeof(__fp16));
            }
        }
    }

    /* Bench 1: PQ naive */
    double t_naive = bench([&]{
        pq_attention_naive(q, k_idx, v_idx, cb_K, cb_V, N, out_pq_naive);
    }, n_iter);

    /* Bench 2: PQ NEON */
    double t_neon = bench([&]{
        pq_attention_neon(q, k_idx, v_idx, cb_K, cb_V, N, out_pq_neon);
    }, n_iter);

    /* Bench 3: fp16 baseline (no compression) */
    double t_fp16 = bench([&]{
        attention_fp16_baseline(q, k_full, v_full, N, out_fp16);
    }, n_iter);

    /* Compare correctness: PQ naive vs PQ neon */
    float max_diff_naive_neon = 0.0f;
    for (int i = 0; i < H_HEADS * HEAD_DIM; i++) {
        float d = fabsf((float)out_pq_naive[i] - (float)out_pq_neon[i]);
        if (d > max_diff_naive_neon) max_diff_naive_neon = d;
    }
    /* PQ naive vs fp16 baseline (should be exact since K/V reconstructed from same codebook) */
    float max_diff_pq_fp16 = 0.0f;
    for (int i = 0; i < H_HEADS * HEAD_DIM; i++) {
        float d = fabsf((float)out_pq_naive[i] - (float)out_fp16[i]);
        if (d > max_diff_pq_fp16) max_diff_pq_fp16 = d;
    }

    /* Memory/BW analysis */
    size_t pq_kv_bytes = (size_t)N * H_HEADS * N_SUB * 2;     /* k_idx + v_idx */
    size_t fp16_kv_bytes = (size_t)N * H_HEADS * HEAD_DIM * 2 * 2;  /* k+v fp16 */

    printf("=== Wall time per attention step (avg over %d iters) ===\n", n_iter);
    printf("  PQ naive  : %8.3f ms\n", t_naive);
    printf("  PQ NEON   : %8.3f ms  (speedup vs naive: %.2fx)\n", t_neon, t_naive/t_neon);
    printf("  fp16 base : %8.3f ms  (PQ NEON vs fp16: %.2fx)\n", t_fp16, t_fp16/t_neon);
    printf("\n=== Correctness ===\n");
    printf("  PQ naive vs PQ NEON max abs diff: %.6f\n", max_diff_naive_neon);
    printf("  PQ naive vs fp16 (same KV)        : %.6f\n", max_diff_pq_fp16);
    printf("\n=== Memory ===\n");
    printf("  PQ KV bytes: %zu (%.2f MB)\n", pq_kv_bytes, pq_kv_bytes/1024.0/1024.0);
    printf("  fp16 KV bytes: %zu (%.2f MB)\n", fp16_kv_bytes, fp16_kv_bytes/1024.0/1024.0);
    printf("  compression: %.1fx\n", (double)fp16_kv_bytes / pq_kv_bytes);

    free(q); free(cb_K); free(cb_V); free(k_idx); free(v_idx);
    free(k_full); free(v_full);
    free(out_pq_naive); free(out_pq_neon); free(out_fp16);
    return 0;
}
