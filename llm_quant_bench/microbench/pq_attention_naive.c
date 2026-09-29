/* PQ-attention: naive C reference implementation.
 *
 * Computes attention output for a single decode step using PQ-compressed KV cache.
 *
 * Inputs:
 *   q[H][D]        : query tensor, fp16, H=8, D=64
 *   k_idx[N][H][S] : K indices, uint8, S=4 sub-positions per (token, head)
 *   v_idx[N][H][S] : V indices, uint8
 *   cb_K[S][KK][SD] : K codebook, fp16, KK=256 entries, SD=16 sub-dim
 *   cb_V[S][KK][SD] : V codebook, fp16
 *
 * Output:
 *   out[H][D] : attention output, fp16
 */
#include "pq_attention.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define H_HEADS 8
#define HEAD_DIM 64
#define N_SUB 4
#define SUB_DIM 16
#define KB 256

void pq_attention_naive(
    const __fp16 *q,             /* [H][D] */
    const uint8_t *k_idx,          /* [N][H][S] */
    const uint8_t *v_idx,          /* [N][H][S] */
    const __fp16 *cb_K,          /* [S][KB][SD] */
    const __fp16 *cb_V,          /* [S][KB][SD] */
    int N,                         /* ctx length */
    __fp16 *out                  /* [H][D] */
) {
    /* Step 1: precompute LUT_K[H][S][KB] = q[H][s*SD..s*SD+SD] · cb_K[s][i] */
    static float lut_k[H_HEADS][N_SUB][KB];
    for (int h = 0; h < H_HEADS; h++) {
        for (int s = 0; s < N_SUB; s++) {
            const __fp16 *qs = q + h * HEAD_DIM + s * SUB_DIM;
            for (int i = 0; i < KB; i++) {
                const __fp16 *cb = cb_K + (s * KB + i) * SUB_DIM;
                float sum = 0.0f;
                for (int d = 0; d < SUB_DIM; d++) {
                    sum += (float)qs[d] * (float)cb[d];
                }
                lut_k[h][s][i] = sum;
            }
        }
    }

    /* Step 2: per-token score for each head */
    float *scores = (float*)malloc(sizeof(float) * N * H_HEADS);
    for (int t = 0; t < N; t++) {
        for (int h = 0; h < H_HEADS; h++) {
            const uint8_t *idx = k_idx + (t * H_HEADS + h) * N_SUB;
            float s0 = lut_k[h][0][idx[0]];
            float s1 = lut_k[h][1][idx[1]];
            float s2 = lut_k[h][2][idx[2]];
            float s3 = lut_k[h][3][idx[3]];
            scores[t * H_HEADS + h] = s0 + s1 + s2 + s3;
        }
    }

    /* Step 3: per-head softmax with sqrt(D) scaling */
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    static float weights[8 * 16384];   /* up to N=16384 */
    for (int h = 0; h < H_HEADS; h++) {
        /* find max for stable softmax */
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

    /* Step 4: V weighted sum, output[h][d] = sum_t w[t] * cb_V[d/SD][v_idx[t][h][d/SD]][d%SD] */
    for (int h = 0; h < H_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            int s = d / SUB_DIM;
            int dd = d % SUB_DIM;
            float acc = 0.0f;
            for (int t = 0; t < N; t++) {
                const uint8_t vi = v_idx[(t * H_HEADS + h) * N_SUB + s];
                float v = (float)cb_V[(s * KB + vi) * SUB_DIM + dd];
                acc += weights[h * N + t] * v;
            }
            out[h * HEAD_DIM + d] = (__fp16)acc;
        }
    }
    free(scores);
}
