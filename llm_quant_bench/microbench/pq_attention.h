#ifndef PQ_ATTENTION_H
#define PQ_ATTENTION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void pq_attention_naive(
    const __fp16 *q,
    const uint8_t *k_idx,
    const uint8_t *v_idx,
    const __fp16 *cb_K,
    const __fp16 *cb_V,
    int N,
    __fp16 *out
);

void pq_attention_neon(
    const __fp16 *q,
    const uint8_t *k_idx,
    const uint8_t *v_idx,
    const __fp16 *cb_K,
    const __fp16 *cb_V,
    int N,
    __fp16 *out
);

/* Reference baseline: standard fp16 attention (no PQ). */
void attention_fp16_baseline(
    const __fp16 *q,             /* [H][D] */
    const __fp16 *k,             /* [N][H][D] */
    const __fp16 *v,             /* [N][H][D] */
    int N,
    __fp16 *out                  /* [H][D] */
);

/* Quantize fp16 KV to PQ indices.
 * Input:  x[N][D] fp16 vectors (D=64)
 * Output: indices[N][N_SUB] uint8
 * codebook: [N_SUB][KB][SUB_DIM] fp16
 */
void pq_quantize_neon(
    const __fp16 *x,             /* [N][HEAD_DIM=64] */
    const __fp16 *codebook,      /* [N_SUB=4][KB=256][SUB_DIM=16] */
    int N,
    uint8_t *indices             /* [N][N_SUB=4] */
);

/* In-place Walsh-Hadamard rotation on a 64-element fp16 vector.
 * Applies H_64 / sqrt(64). Caller responsible for any sign-flip diagonal.
 * Cost: 6 stages of pairwise add/sub, normalized by 1/8.
 * Used for TurboQuant-style data-oblivious quantization (rotate before quant).
 */
void hadamard64_neon(__fp16 *x);

/* TurboQuant-style rotated scalar attention (Stage 2 LUT kernel).
 *
 * Cache layout (compressed):
 *   k_idx[N][H][32]     packed 4-bit indices (2 dims per byte, 32 bytes per token-head)
 *   k_norm[N][H]        fp16 per-token-head norm
 *   v_idx[N][H][32]     packed 4-bit indices
 *   v_norm[N][H]        fp16 per-token-head norm
 *   codebook[16]        shared fp16 Lloyd-Max centroids (32 bytes total)
 *
 * Compute (per decode step):
 *   Q rotated externally (caller must apply Hadamard×Diagonal first)
 *   LUT[H][D][16] = Rq[h][d] × codebook[i]   precomputed per head
 *   score[t][h] = norm[t][h] × Σ_d LUT[h][d][idx[t][h][d]]
 *   weights = softmax(scores / sqrt(D))
 *   out[h][d] = Σ_t weights[t][h] × norm[t][h] × codebook[v_idx[t][h][d]]
 *   out is in rotated space; caller applies inverse rotation if needed
 *
 * Memory savings: 34 bytes/token-head (32 idx + 2 norm) vs 128 fp16 = 3.76× compressed.
 * Compute: 64 LUT lookups per token (vs 64 fp16 multiplies in fp16 baseline) — same op count
 * but indices small in cache → much less DDR traffic.
 */
void rot_attention_neon(
    const __fp16 *q_rotated,     /* [H][D] — caller-applied Hadamard×Diag */
    const uint8_t *k_idx,        /* [N][H][D/2] packed 4-bit, 2 dims per byte */
    const __fp16 *k_norm,        /* [N][H] */
    const uint8_t *v_idx,        /* [N][H][D/2] */
    const __fp16 *v_norm,        /* [N][H] */
    const __fp16 *codebook,      /* [16] shared centroids */
    int N,
    __fp16 *out_rotated          /* [H][D] — in rotated space */
);

/* Quantize fp16 vector (after rotation) to 4-bit indices + norm.
 * Caller applies rotation first; this function takes already-rotated data.
 */
void rot_quantize_neon(
    const __fp16 *x_rotated,     /* [N][H][D] — already rotated */
    int N, int H,
    const __fp16 *codebook,      /* [16] sorted centroids */
    uint8_t *idx_out,            /* [N][H][D/2] packed 4-bit */
    __fp16 *norm_out             /* [N][H] */
);

/* Vectorized variant: NEON vqtbl2q_u8 batched 16-dim dequant + fp16 fma.
 * Same algorithm as rot_attention_neon but optimized inner loop.
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
);

/* GQA variant: n_q_heads != n_kv_heads, group_size = n_q_heads / n_kv_heads.
 * Each Q head q_h shares KV head q_h / group_size.
 * For Llama-3.2-1B: n_q_heads=32, n_kv_heads=8, group_size=4.
 */
void rot_attention_gqa_neon(
    const __fp16 *q_rotated,    /* [n_q_heads][HEAD_DIM] */
    const uint8_t *k_idx,       /* [N][n_kv_heads][HEAD_DIM/2] */
    const __fp16 *k_norm,       /* [N][n_kv_heads] */
    const uint8_t *v_idx,       /* [N][n_kv_heads][HEAD_DIM/2] */
    const __fp16 *v_norm,       /* [N][n_kv_heads] */
    const __fp16 *codebook,     /* [16] */
    int N,
    int n_q_heads,
    int n_kv_heads,
    __fp16 *out_rotated         /* [n_q_heads][HEAD_DIM] */
);

/* Apply rotation R to N×64 vectors (row-wise), batched.
 * R is fp16 64x64 orthonormal matrix (typically Hadamard-Diagonal).
 * Out = X @ R^T (row form: each row x_i becomes R x_i).
 */
void rotate_batch_neon(
    const __fp16 *X,             /* [N][64] */
    const __fp16 *R,             /* [64][64] orthonormal */
    int N,
    __fp16 *Y                    /* [N][64] */
);

/* Combined: read fp16 cached K,V → quantize to indices → run PQ attention.
 * This is the "drop-in" replacement for fp16 attention when used in MNN.
 * Memory layout unchanged (KV stays fp16); compute path uses PQ.
 */
void pq_attention_from_fp16_cache(
    const __fp16 *q,             /* [H][D] current Q */
    const __fp16 *K_cache,       /* [N][H][D] cached K (fp16) */
    const __fp16 *V_cache,       /* [N][H][D] cached V (fp16) */
    const __fp16 *cb_K,          /* [S][KB][SD] */
    const __fp16 *cb_V,
    int N,
    __fp16 *out                  /* [H][D] */
);

#ifdef __cplusplus
}
#endif

#endif
