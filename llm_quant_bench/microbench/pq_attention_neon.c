/* PQ-attention: ARM NEON optimized implementation for Cortex-A76.
 *
 * Strategy:
 *   Step 1 (LUT precompute): scalar (small, only H*S*KB = 8192 ops)
 *   Step 2 (per-token score): scalar 4 lookups + 3 adds (memory-bound)
 *   Step 3 (softmax): scalar
 *   Step 4 (V weighted sum): NEON 8-wide fp16 fmla — main optimization target
 *
 * The Step 4 inner loop loads 16 fp16 codebook values via uint8 index
 * lookup, multiplies by scalar weight, and accumulates into per-head
 * output vector. This is memory-bound (16 KB codebook lookup pattern).
 */
#include "pq_attention.h"
#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define H_HEADS 8
#define HEAD_DIM 64
#define N_SUB 4
#define SUB_DIM 16
#define KB 256

/* helper: fp16 horizontal sum over 8 lanes */
static inline float vsum8_f16(float16x8_t v) {
    float32x4_t lo = vcvt_f32_f16(vget_low_f16(v));
    float32x4_t hi = vcvt_f32_f16(vget_high_f16(v));
    float32x4_t s = vaddq_f32(lo, hi);
    return vaddvq_f32(s);
}

void pq_attention_neon(
    const __fp16 *q,             /* [H][D] */
    const uint8_t *k_idx,          /* [N][H][S] */
    const uint8_t *v_idx,          /* [N][H][S] */
    const __fp16 *cb_K,          /* [S][KB][SD] */
    const __fp16 *cb_V,          /* [S][KB][SD] */
    int N,
    __fp16 *out                  /* [H][D] */
) {
    /* Step 1: precompute LUT_K[H][S][KB] — fp32 for accuracy.
     * Parallelize over (h, s) — 32 small units, but distributable across cores. */
    static float lut_k[H_HEADS][N_SUB][KB];
    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < H_HEADS; h++) {
        for (int s = 0; s < N_SUB; s++) {
            const __fp16 *qs = q + h * HEAD_DIM + s * SUB_DIM;
            float16x8_t q_lo = vld1q_f16(qs);
            float16x8_t q_hi = vld1q_f16(qs + 8);
            for (int i = 0; i < KB; i++) {
                const __fp16 *cb = cb_K + (s * KB + i) * SUB_DIM;
                float16x8_t cb_lo = vld1q_f16(cb);
                float16x8_t cb_hi = vld1q_f16(cb + 8);
                /* dot product via fma + horizontal sum */
                float16x8_t prod_lo = vmulq_f16(q_lo, cb_lo);
                float16x8_t prod_hi = vmulq_f16(q_hi, cb_hi);
                float32x4_t a0 = vcvt_f32_f16(vget_low_f16(prod_lo));
                float32x4_t a1 = vcvt_f32_f16(vget_high_f16(prod_lo));
                float32x4_t a2 = vcvt_f32_f16(vget_low_f16(prod_hi));
                float32x4_t a3 = vcvt_f32_f16(vget_high_f16(prod_hi));
                float32x4_t s_acc = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
                lut_k[h][s][i] = vaddvq_f32(s_acc);
            }
        }
    }

    /* Step 2: per-token score — parallelize over tokens (N is large). */
    float *scores = (float*)malloc(sizeof(float) * N * H_HEADS);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < N; t++) {
        for (int h = 0; h < H_HEADS; h++) {
            const uint8_t *idx = k_idx + (t * H_HEADS + h) * N_SUB;
            scores[t * H_HEADS + h] = lut_k[h][0][idx[0]] + lut_k[h][1][idx[1]]
                                    + lut_k[h][2][idx[2]] + lut_k[h][3][idx[3]];
        }
    }

    /* Step 3: per-head softmax — parallelize across heads (8 ind units). */
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    static float weights[H_HEADS * 16384];
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < H_HEADS; h++) {
        float maxv = -INFINITY;
        for (int t = 0; t < N; t++) {
            float s = scores[t * H_HEADS + h] * scale;
            if (s > maxv) maxv = s;
        }
        float sum = 0.0f;
        for (int t = 0; t < N; t++) {
            float e = expf(scores[t * H_HEADS + h] * scale - maxv);
            weights[h * N + t] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        for (int t = 0; t < N; t++) {
            weights[h * N + t] *= inv;
        }
    }

    /* Step 4: V weighted sum via NEON 8-wide fp16 fmla.
     * Parallelize over (h, s) — 32 work units, each iterating over N tokens. */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < H_HEADS; h++) {
        for (int s = 0; s < N_SUB; s++) {
            float16x8_t acc_lo = vdupq_n_f16((__fp16)0.0f);
            float16x8_t acc_hi = vdupq_n_f16((__fp16)0.0f);
            for (int t = 0; t < N; t++) {
                uint8_t idx = v_idx[(t * H_HEADS + h) * N_SUB + s];
                const __fp16 *cb = cb_V + (s * KB + idx) * SUB_DIM;
                float16x8_t v_lo = vld1q_f16(cb);
                float16x8_t v_hi = vld1q_f16(cb + 8);
                __fp16 w = (__fp16)weights[h * N + t];
                acc_lo = vfmaq_n_f16(acc_lo, v_lo, w);
                acc_hi = vfmaq_n_f16(acc_hi, v_hi, w);
            }
            vst1q_f16(out + h * HEAD_DIM + s * SUB_DIM, acc_lo);
            vst1q_f16(out + h * HEAD_DIM + s * SUB_DIM + 8, acc_hi);
        }
    }

    free(scores);
}

/* In-place Walsh-Hadamard transform on 64 fp16 elements (6 stages, normalize 1/8).
 * Stages 1-3 (stride 1, 2, 4): scalar within 8-element blocks.
 * Stages 4-6 (stride 8, 16, 32): NEON 8-wide pairwise add/sub.
 * Final scale by 1/sqrt(64) = 1/8 to keep H orthonormal.
 */
void hadamard64_neon(__fp16 *x) {
    /* stride 1, 2, 4 within 8-element blocks (scalar; cheap, 64 ops total) */
    for (int blk = 0; blk < 64; blk += 8) {
        __fp16 *p = x + blk;
        __fp16 t;
        /* stride 1 */
        for (int i = 0; i < 8; i += 2) { t = p[i] + p[i+1]; p[i+1] = p[i] - p[i+1]; p[i] = t; }
        /* stride 2 */
        for (int i = 0; i < 8; i += 4) {
            __fp16 a0=p[i], a1=p[i+1], b0=p[i+2], b1=p[i+3];
            p[i] = a0+b0; p[i+1] = a1+b1; p[i+2] = a0-b0; p[i+3] = a1-b1;
        }
        /* stride 4 */
        for (int i = 0; i < 4; i++) { t = p[i] + p[i+4]; p[i+4] = p[i] - p[i+4]; p[i] = t; }
    }
    /* stride 8: NEON 8-wide butterfly between adjacent blocks */
    for (int blk = 0; blk < 64; blk += 16) {
        float16x8_t a = vld1q_f16(x + blk);
        float16x8_t b = vld1q_f16(x + blk + 8);
        vst1q_f16(x + blk,     vaddq_f16(a, b));
        vst1q_f16(x + blk + 8, vsubq_f16(a, b));
    }
    /* stride 16 */
    for (int blk = 0; blk < 64; blk += 32) {
        for (int j = 0; j < 16; j += 8) {
            float16x8_t a = vld1q_f16(x + blk + j);
            float16x8_t b = vld1q_f16(x + blk + 16 + j);
            vst1q_f16(x + blk + j,      vaddq_f16(a, b));
            vst1q_f16(x + blk + 16 + j, vsubq_f16(a, b));
        }
    }
    /* stride 32 */
    for (int j = 0; j < 32; j += 8) {
        float16x8_t a = vld1q_f16(x + j);
        float16x8_t b = vld1q_f16(x + 32 + j);
        vst1q_f16(x + j,      vaddq_f16(a, b));
        vst1q_f16(x + 32 + j, vsubq_f16(a, b));
    }
    /* normalize 1/sqrt(64) = 1/8 */
    float16x8_t inv8 = vdupq_n_f16((__fp16)(1.0f / 8.0f));
    for (int i = 0; i < 64; i += 8) {
        float16x8_t v = vld1q_f16(x + i);
        vst1q_f16(x + i, vmulq_f16(v, inv8));
    }
}

/* Batched orthonormal rotation: Y[n] = R @ X[n] for all rows.
 * Y[n][d] = sum_j R[d][j] * X[n][j]. Inner loop NEON 8-wide fp16 fmla.
 */
void rotate_batch_neon(const __fp16 *X, const __fp16 *R, int N, __fp16 *Y) {
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const __fp16 *xn = X + n * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) {
            const __fp16 *rd = R + d * HEAD_DIM;
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int j = 0; j < HEAD_DIM; j += 8) {
                float16x8_t r = vld1q_f16(rd + j);
                float16x8_t xv = vld1q_f16(xn + j);
                float16x8_t prod = vmulq_f16(r, xv);
                float32x4_t lo = vcvt_f32_f16(vget_low_f16(prod));
                float32x4_t hi = vcvt_f32_f16(vget_high_f16(prod));
                acc = vaddq_f32(acc, vaddq_f32(lo, hi));
            }
            Y[n * HEAD_DIM + d] = (__fp16)vaddvq_f32(acc);
        }
    }
}

/* Quantize fp16 vectors to PQ indices using NEON SIMD nearest-codebook search. */
void pq_quantize_neon(
    const __fp16 *x,             /* [N][64] */
    const __fp16 *codebook,      /* [N_SUB=4][KB=256][SUB_DIM=16] */
    int N,
    uint8_t *indices             /* [N][N_SUB=4] */
) {
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const __fp16 *xn = x + n * HEAD_DIM;
        for (int s = 0; s < N_SUB; s++) {
            const __fp16 *xs = xn + s * SUB_DIM;
            float16x8_t x_lo = vld1q_f16(xs);
            float16x8_t x_hi = vld1q_f16(xs + 8);
            float best_dist = 1e30f;
            int best_i = 0;
            for (int i = 0; i < KB; i++) {
                const __fp16 *cb = codebook + (s * KB + i) * SUB_DIM;
                float16x8_t cb_lo = vld1q_f16(cb);
                float16x8_t cb_hi = vld1q_f16(cb + 8);
                float16x8_t d_lo = vsubq_f16(x_lo, cb_lo);
                float16x8_t d_hi = vsubq_f16(x_hi, cb_hi);
                float16x8_t sq_lo = vmulq_f16(d_lo, d_lo);
                float16x8_t sq_hi = vmulq_f16(d_hi, d_hi);
                float32x4_t a = vcvt_f32_f16(vget_low_f16(sq_lo));
                float32x4_t b = vcvt_f32_f16(vget_high_f16(sq_lo));
                float32x4_t c = vcvt_f32_f16(vget_low_f16(sq_hi));
                float32x4_t d = vcvt_f32_f16(vget_high_f16(sq_hi));
                float32x4_t sum = vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d));
                float dist = vaddvq_f32(sum);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_i = i;
                }
            }
            indices[n * N_SUB + s] = (uint8_t)best_i;
        }
    }
}

/* Drop-in: quantize fp16 cache on-the-fly + run PQ attention.
 * Note: this re-quantizes ALL cached K,V every call. For incremental decode
 * use a separate "append" path that only quantizes new tokens.
 */
void pq_attention_from_fp16_cache(
    const __fp16 *q, const __fp16 *K_cache, const __fp16 *V_cache,
    const __fp16 *cb_K, const __fp16 *cb_V, int N, __fp16 *out
) {
    /* Quantize K_cache (N×H×D) → k_idx (N×H×S).
     * Layout: input is [token][head][dim] flat fp16; output is [token][head][sub] uint8.
     * Each (token, head) is a 64-dim vector → 4 indices.
     */
    int total = N * H_HEADS;
    uint8_t *k_idx = (uint8_t*)malloc(sizeof(uint8_t) * total * N_SUB);
    uint8_t *v_idx = (uint8_t*)malloc(sizeof(uint8_t) * total * N_SUB);
    pq_quantize_neon(K_cache, cb_K, total, k_idx);
    pq_quantize_neon(V_cache, cb_V, total, v_idx);
    pq_attention_neon(q, k_idx, v_idx, cb_K, cb_V, N, out);
    free(k_idx); free(v_idx);
}

/* Reference baseline: standard fp16 attention (no PQ). */
void attention_fp16_baseline(
    const __fp16 *q,             /* [H][D] */
    const __fp16 *k,             /* [N][H][D] */
    const __fp16 *v,             /* [N][H][D] */
    int N,
    __fp16 *out
) {
    float *scores = (float*)malloc(sizeof(float) * N * H_HEADS);
    /* Step 1: dot-product Q*K^T, NEON 8-wide fp16 — parallelized over tokens. */
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < N; t++) {
        for (int h = 0; h < H_HEADS; h++) {
            const __fp16 *qh = q + h * HEAD_DIM;
            const __fp16 *kth = k + (t * H_HEADS + h) * HEAD_DIM;
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int d = 0; d < HEAD_DIM; d += 8) {
                float16x8_t qv = vld1q_f16(qh + d);
                float16x8_t kv = vld1q_f16(kth + d);
                float16x8_t prod = vmulq_f16(qv, kv);
                float32x4_t lo = vcvt_f32_f16(vget_low_f16(prod));
                float32x4_t hi = vcvt_f32_f16(vget_high_f16(prod));
                acc = vaddq_f32(acc, vaddq_f32(lo, hi));
            }
            scores[t * H_HEADS + h] = vaddvq_f32(acc);
        }
    }
    /* softmax — parallelize over heads */
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    static float weights[H_HEADS * 16384];
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < H_HEADS; h++) {
        float maxv = -INFINITY;
        for (int t = 0; t < N; t++) {
            float s = scores[t * H_HEADS + h] * scale;
            if (s > maxv) maxv = s;
        }
        float sum = 0.0f;
        for (int t = 0; t < N; t++) {
            float e = expf(scores[t * H_HEADS + h] * scale - maxv);
            weights[h * N + t] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        for (int t = 0; t < N; t++) {
            weights[h * N + t] *= inv;
        }
    }
    /* Step 4: V weighted sum, NEON — parallelize over (h, d) blocks */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < H_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d += 8) {
            float16x8_t acc = vdupq_n_f16((__fp16)0.0f);
            for (int t = 0; t < N; t++) {
                const __fp16 *vth = v + (t * H_HEADS + h) * HEAD_DIM + d;
                float16x8_t vv = vld1q_f16(vth);
                __fp16 w = (__fp16)weights[h * N + t];
                acc = vfmaq_n_f16(acc, vv, w);
            }
            vst1q_f16(out + h * HEAD_DIM + d, acc);
        }
    }
    free(scores);
}

/* ===== TurboQuant Stage 2: rotated scalar attention with 4-bit LUT ===== */

/* Quantize already-rotated fp16 vector to 4-bit indices + norm.
 * Lloyd-Max codebook (16 centroids), per-row normalization.
 * idx_out: 2 dims packed per byte — low nibble = even d, high nibble = odd d.
 */
void rot_quantize_neon(
    const __fp16 *x_rot, int N, int H,
    const __fp16 *codebook,
    uint8_t *idx_out, __fp16 *norm_out
) {
    /* Boundaries (scalar fp32 once) */
    float bnd[15];
    for (int i = 0; i < 15; i++) {
        bnd[i] = 0.5f * ((float)codebook[i] + (float)codebook[i+1]);
    }
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            const __fp16 *xnh = x_rot + (n * H + h) * HEAD_DIM;
            /* Compute norm */
            float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
            for (int d = 0; d < HEAD_DIM; d += 8) {
                float16x8_t v = vld1q_f16(xnh + d);
                float32x4_t lo = vcvt_f32_f16(vget_low_f16(v));
                float32x4_t hi = vcvt_f32_f16(vget_high_f16(v));
                s0 = vmlaq_f32(s0, lo, lo);
                s1 = vmlaq_f32(s1, hi, hi);
            }
            float norm2 = vaddvq_f32(vaddq_f32(s0, s1));
            float norm = sqrtf(norm2);
            float inv = (norm > 1e-9f) ? 1.0f / norm : 0.0f;
            norm_out[n * H + h] = (__fp16)norm;
            uint8_t *idx_row = idx_out + (n * H + h) * (HEAD_DIM / 2);
            /* Per-coord nearest-centroid (scalar binary search; per-row 64 ops) */
            for (int d = 0; d < HEAD_DIM; d += 2) {
                float v0 = (float)xnh[d] * inv;
                float v1 = (float)xnh[d+1] * inv;
                int lo0 = 0, hi0 = 15;
                while (lo0 < hi0) { int m = (lo0+hi0)>>1; if (v0 < bnd[m]) hi0 = m; else lo0 = m+1; }
                int lo1 = 0, hi1 = 15;
                while (lo1 < hi1) { int m = (lo1+hi1)>>1; if (v1 < bnd[m]) hi1 = m; else lo1 = m+1; }
                idx_row[d/2] = (uint8_t)(lo0 | (lo1 << 4));
            }
        }
    }
}

/* TurboQuant Stage 2 attention: Q (rotated) × compressed K,V via LUT.
 *
 * Algorithm:
 *   1. LUT[h][d][i] = q_rot[h][d] * codebook[i]    (per-head precompute, 8*64*16 entries)
 *   2. score[t][h] = k_norm[t][h] * sum_d LUT[h][d][idx[t][h][d]]
 *   3. softmax → weights[t][h]
 *   4. out_rot[h][d] = sum_t weights[t][h] * v_norm[t][h] * codebook[v_idx[t][h][d]]
 *
 * Optimizations:
 *   - LUT precompute: NEON 4-wide fp32 fma over 16 centroids
 *   - Per-token score: scalar 64 LUT lookups (memory-bound)
 *   - V weighted sum: NEON-style, but indexed reads break SIMD locality
 */
void rot_attention_neon(
    const __fp16 *q_rotated,
    const uint8_t *k_idx,
    const __fp16 *k_norm,
    const uint8_t *v_idx,
    const __fp16 *v_norm,
    const __fp16 *codebook,
    int N,
    __fp16 *out_rotated
) {
    /* Step 1: LUT_K[H][D][16] = q_rotated[h][d] * codebook[i] */
    static float lut_k[H_HEADS][HEAD_DIM][16];
    /* Convert codebook to fp32 once */
    float cb_f[16];
    for (int i = 0; i < 16; i++) cb_f[i] = (float)codebook[i];
    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < H_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            float q = (float)q_rotated[h * HEAD_DIM + d];
            for (int i = 0; i < 16; i++) {
                lut_k[h][d][i] = q * cb_f[i];
            }
        }
    }

    /* Step 2: per-token score, parallelized over tokens */
    float *scores = (float*)malloc(sizeof(float) * N * H_HEADS);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < N; t++) {
        for (int h = 0; h < H_HEADS; h++) {
            const uint8_t *idx_row = k_idx + (t * H_HEADS + h) * (HEAD_DIM / 2);
            float acc = 0.0f;
            for (int d = 0; d < HEAD_DIM; d += 2) {
                uint8_t b = idx_row[d / 2];
                int i0 = b & 0xF;
                int i1 = (b >> 4) & 0xF;
                acc += lut_k[h][d][i0] + lut_k[h][d+1][i1];
            }
            scores[t * H_HEADS + h] = (float)k_norm[t * H_HEADS + h] * acc;
        }
    }

    /* Step 3: softmax (per head) */
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    static float weights[H_HEADS * 16384];
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < H_HEADS; h++) {
        float maxv = -INFINITY;
        for (int t = 0; t < N; t++) {
            float s = scores[t * H_HEADS + h] * scale;
            if (s > maxv) maxv = s;
        }
        float sum = 0.0f;
        for (int t = 0; t < N; t++) {
            float e = expf(scores[t * H_HEADS + h] * scale - maxv);
            weights[h * N + t] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        for (int t = 0; t < N; t++) weights[h * N + t] *= inv;
    }

    /* Step 4: V weighted sum.
     * out_rot[h][d] = sum_t weights[t][h] * v_norm[t][h] * codebook[v_idx[t][h][d]]
     * Inner loop over t accumulates per-(h,d). Indexed loads break SIMD across d,
     * so we accumulate per dimension scalar fp32 → store as fp16.
     */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < H_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            float acc = 0.0f;
            for (int t = 0; t < N; t++) {
                const uint8_t *idx_row = v_idx + (t * H_HEADS + h) * (HEAD_DIM / 2);
                uint8_t b = idx_row[d / 2];
                int i = (d & 1) ? ((b >> 4) & 0xF) : (b & 0xF);
                acc += weights[h * N + t] * (float)v_norm[t * H_HEADS + h] * cb_f[i];
            }
            out_rotated[h * HEAD_DIM + d] = (__fp16)acc;
        }
    }
    free(scores);
}

/* GQA variant: handles n_q_heads ≠ n_kv_heads with group_size = n_q_heads/n_kv_heads.
 * For each Q head q_h, the corresponding KV head is q_h / group_size.
 * Llama-3.2-1B: 32 Q × 8 KV → group_size=4 (each KV head shared by 4 Q heads).
 */
void rot_attention_gqa_neon(
    const __fp16 *q_rotated,
    const uint8_t *k_idx,
    const __fp16 *k_norm,
    const uint8_t *v_idx,
    const __fp16 *v_norm,
    const __fp16 *codebook,
    int N, int n_q_heads, int n_kv_heads,
    __fp16 *out_rotated
) {
    int group_size = n_q_heads / n_kv_heads;
    float cb_f[16];
    for (int i = 0; i < 16; i++) cb_f[i] = (float)codebook[i];

    /* Per Q head: compute score, softmax, weighted V sum via codebook lookup */
    #pragma omp parallel for schedule(static)
    for (int qh = 0; qh < n_q_heads; qh++) {
        int kvh = qh / group_size;
        const __fp16 *qh_v = q_rotated + qh * HEAD_DIM;

        /* Q values converted to fp32 for LUT precompute */
        float q_f[HEAD_DIM];
        for (int d = 0; d < HEAD_DIM; d++) q_f[d] = (float)qh_v[d];

        /* LUT[d][i] = q_f[d] * cb_f[i] — 64*16 = 1024 floats per head */
        float lut[HEAD_DIM][16];
        for (int d = 0; d < HEAD_DIM; d++) {
            for (int i = 0; i < 16; i++) lut[d][i] = q_f[d] * cb_f[i];
        }

        /* Step 1: scores per token */
        float *scores = (float*)malloc(sizeof(float) * N);
        for (int t = 0; t < N; t++) {
            const uint8_t *idx_row = k_idx + (t * n_kv_heads + kvh) * (HEAD_DIM / 2);
            float acc = 0.0f;
            for (int d = 0; d < HEAD_DIM; d += 2) {
                uint8_t b = idx_row[d / 2];
                int i0 = b & 0xF;
                int i1 = (b >> 4) & 0xF;
                acc += lut[d][i0] + lut[d+1][i1];
            }
            scores[t] = (float)k_norm[t * n_kv_heads + kvh] * acc;
        }

        /* Step 2: softmax */
        float scale = 1.0f / sqrtf((float)HEAD_DIM);
        float maxv = -INFINITY;
        for (int t = 0; t < N; t++) {
            float s = scores[t] * scale;
            if (s > maxv) maxv = s;
        }
        float sum = 0.0f;
        for (int t = 0; t < N; t++) {
            float e = expf(scores[t] * scale - maxv);
            scores[t] = e;
            sum += e;
        }
        float inv = 1.0f / sum;

        /* Step 3: weighted V sum, output to out_rotated[qh] */
        float out_f[HEAD_DIM] = {0.0f};
        for (int t = 0; t < N; t++) {
            const uint8_t *idx_row = v_idx + (t * n_kv_heads + kvh) * (HEAD_DIM / 2);
            float w = scores[t] * inv * (float)v_norm[t * n_kv_heads + kvh];
            for (int d = 0; d < HEAD_DIM; d += 2) {
                uint8_t b = idx_row[d / 2];
                int i0 = b & 0xF;
                int i1 = (b >> 4) & 0xF;
                out_f[d]   += w * cb_f[i0];
                out_f[d+1] += w * cb_f[i1];
            }
        }
        for (int d = 0; d < HEAD_DIM; d++) {
            out_rotated[qh * HEAD_DIM + d] = (__fp16)out_f[d];
        }
        free(scores);
    }
}

/* Helper: dequantize 32 dims (16 packed bytes) from indices to fp16 using
 * NEON vqtbl2q_u8 against codebook table. Output: 4 float16x8_t (32 fp16).
 * Sequential dim ordering: out0 = dims 0..7, out1 = 8..15, etc.
 */
static inline void dequant_32_dims_neon(
    uint8x16_t bytes,           // 16 packed input bytes (2 nibbles each = 32 dims)
    uint8x16x2_t cb_tbl,        // codebook as 2x16-byte table
    float16x8_t *o0, float16x8_t *o1, float16x8_t *o2, float16x8_t *o3
) {
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    uint8x16_t lo_n = vandq_u8(bytes, mask_lo);
    uint8x16_t hi_n = vshrq_n_u8(bytes, 4);

    // Sequential order: dim 0,2,4,...,30 = lo_n[0..15]; dim 1,3,5,...,31 = hi_n[0..15]
    // For sequential dims 0..15 in 16 bytes: zip lo and hi (low half)
    uint8x16_t seq_n_0_15 = vzip1q_u8(lo_n, hi_n);
    uint8x16_t seq_n_16_31 = vzip2q_u8(lo_n, hi_n);

    // Convert nibble (0..15) to fp16 byte indices: byte0 = 2*nibble, byte1 = 2*nibble+1
    uint8x16_t db0_a = vshlq_n_u8(seq_n_0_15, 1);
    uint8x16_t db1_a = vorrq_u8(db0_a, vdupq_n_u8(1));
    uint8x16_t db0_b = vshlq_n_u8(seq_n_16_31, 1);
    uint8x16_t db1_b = vorrq_u8(db0_b, vdupq_n_u8(1));

    // Interleave (byte0, byte1) for each nibble → 32 byte indices per 16 nibbles
    // We need 2 vqtbl2q_u8 calls per 16 nibbles (each call uses 16 byte indices)
    // Lower 8 dims: byte indices = (2*n0, 2*n0+1, 2*n1, 2*n1+1, ..., 2*n7, 2*n7+1)
    uint8x16_t bi_0_7   = vzip1q_u8(db0_a, db1_a);
    uint8x16_t bi_8_15  = vzip2q_u8(db0_a, db1_a);
    uint8x16_t bi_16_23 = vzip1q_u8(db0_b, db1_b);
    uint8x16_t bi_24_31 = vzip2q_u8(db0_b, db1_b);

    *o0 = vreinterpretq_f16_u8(vqtbl2q_u8(cb_tbl, bi_0_7));
    *o1 = vreinterpretq_f16_u8(vqtbl2q_u8(cb_tbl, bi_8_15));
    *o2 = vreinterpretq_f16_u8(vqtbl2q_u8(cb_tbl, bi_16_23));
    *o3 = vreinterpretq_f16_u8(vqtbl2q_u8(cb_tbl, bi_24_31));
}

/* Stage 2 v2: vqtbl2q_u8-vectorized dequant + fp16 fmla.
 * Same algorithm as v1, but dequant K/V via NEON tbl avoids per-dim LUT precompute
 * and exploits SIMD efficiency.
 */
void rot_attention_neon_v2(
    const __fp16 *q_rotated,
    const uint8_t *k_idx,
    const __fp16 *k_norm,
    const uint8_t *v_idx,
    const __fp16 *v_norm,
    const __fp16 *codebook,
    int N,
    __fp16 *out_rotated
) {
    /* Codebook table for vqtbl2q_u8 (32-byte 2-reg) */
    uint8x16x2_t cb_tbl;
    cb_tbl.val[0] = vld1q_u8((const uint8_t*)codebook);
    cb_tbl.val[1] = vld1q_u8((const uint8_t*)codebook + 16);

    /* Step 2: per-token score via vectorized dequant + fp16 fmla */
    float *scores = (float*)malloc(sizeof(float) * N * H_HEADS);
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < N; t++) {
        for (int h = 0; h < H_HEADS; h++) {
            const uint8_t *idx_row = k_idx + (t * H_HEADS + h) * (HEAD_DIM / 2);
            const __fp16 *qh = q_rotated + h * HEAD_DIM;

            float32x4_t acc_f32 = vdupq_n_f32(0.0f);
            // Two 16-byte chunks → 32 dims each → 64 dims total
            for (int chunk = 0; chunk < 2; chunk++) {
                uint8x16_t bytes = vld1q_u8(idx_row + chunk * 16);
                float16x8_t k0, k1, k2, k3;
                dequant_32_dims_neon(bytes, cb_tbl, &k0, &k1, &k2, &k3);

                float16x8_t q0 = vld1q_f16(qh + chunk * 32 + 0);
                float16x8_t q1 = vld1q_f16(qh + chunk * 32 + 8);
                float16x8_t q2 = vld1q_f16(qh + chunk * 32 + 16);
                float16x8_t q3 = vld1q_f16(qh + chunk * 32 + 24);

                float16x8_t p0 = vmulq_f16(q0, k0);
                float16x8_t p1 = vmulq_f16(q1, k1);
                float16x8_t p2 = vmulq_f16(q2, k2);
                float16x8_t p3 = vmulq_f16(q3, k3);

                acc_f32 = vaddq_f32(acc_f32, vcvt_f32_f16(vget_low_f16(p0)));
                acc_f32 = vaddq_f32(acc_f32, vcvt_f32_f16(vget_high_f16(p0)));
                acc_f32 = vaddq_f32(acc_f32, vcvt_f32_f16(vget_low_f16(p1)));
                acc_f32 = vaddq_f32(acc_f32, vcvt_f32_f16(vget_high_f16(p1)));
                acc_f32 = vaddq_f32(acc_f32, vcvt_f32_f16(vget_low_f16(p2)));
                acc_f32 = vaddq_f32(acc_f32, vcvt_f32_f16(vget_high_f16(p2)));
                acc_f32 = vaddq_f32(acc_f32, vcvt_f32_f16(vget_low_f16(p3)));
                acc_f32 = vaddq_f32(acc_f32, vcvt_f32_f16(vget_high_f16(p3)));
            }
            float dot = vaddvq_f32(acc_f32);
            scores[t * H_HEADS + h] = (float)k_norm[t * H_HEADS + h] * dot;
        }
    }

    /* Step 3: softmax */
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    static float weights[H_HEADS * 16384];
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < H_HEADS; h++) {
        float maxv = -INFINITY;
        for (int t = 0; t < N; t++) {
            float s = scores[t * H_HEADS + h] * scale;
            if (s > maxv) maxv = s;
        }
        float sum = 0.0f;
        for (int t = 0; t < N; t++) {
            float e = expf(scores[t * H_HEADS + h] * scale - maxv);
            weights[h * N + t] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        for (int t = 0; t < N; t++) weights[h * N + t] *= inv;
    }

    /* Step 4: V weighted sum with vectorized dequant.
     * Per head, accumulate fp32 over 64 dims in 16 float32x4_t slots.
     */
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < H_HEADS; h++) {
        float32x4_t acc[16];
        for (int i = 0; i < 16; i++) acc[i] = vdupq_n_f32(0.0f);
        for (int t = 0; t < N; t++) {
            const uint8_t *idx_row = v_idx + (t * H_HEADS + h) * (HEAD_DIM / 2);
            float w_norm = weights[h * N + t] * (float)v_norm[t * H_HEADS + h];
            float32x4_t w_v = vdupq_n_f32(w_norm);
            for (int chunk = 0; chunk < 2; chunk++) {
                uint8x16_t bytes = vld1q_u8(idx_row + chunk * 16);
                float16x8_t k0, k1, k2, k3;
                dequant_32_dims_neon(bytes, cb_tbl, &k0, &k1, &k2, &k3);
                int base = chunk * 8;
                acc[base+0] = vaddq_f32(acc[base+0], vmulq_f32(w_v, vcvt_f32_f16(vget_low_f16(k0))));
                acc[base+1] = vaddq_f32(acc[base+1], vmulq_f32(w_v, vcvt_f32_f16(vget_high_f16(k0))));
                acc[base+2] = vaddq_f32(acc[base+2], vmulq_f32(w_v, vcvt_f32_f16(vget_low_f16(k1))));
                acc[base+3] = vaddq_f32(acc[base+3], vmulq_f32(w_v, vcvt_f32_f16(vget_high_f16(k1))));
                acc[base+4] = vaddq_f32(acc[base+4], vmulq_f32(w_v, vcvt_f32_f16(vget_low_f16(k2))));
                acc[base+5] = vaddq_f32(acc[base+5], vmulq_f32(w_v, vcvt_f32_f16(vget_high_f16(k2))));
                acc[base+6] = vaddq_f32(acc[base+6], vmulq_f32(w_v, vcvt_f32_f16(vget_low_f16(k3))));
                acc[base+7] = vaddq_f32(acc[base+7], vmulq_f32(w_v, vcvt_f32_f16(vget_high_f16(k3))));
            }
        }
        // Store 64 fp32 → fp16 output, in pairs of 4
        for (int i = 0; i < 16; i++) {
            float tmp[4]; vst1q_f32(tmp, acc[i]);
            for (int j = 0; j < 4; j++) {
                int d_idx = i * 4 + j;
                out_rotated[h * HEAD_DIM + d_idx] = (__fp16)tmp[j];
            }
        }
    }
    free(scores);
}
