#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// C[M, N] = A[M, K] * B[K, N]
__kernel void matmul_fp16(
    int M, int N, int K,
    __global const half* restrict A, int lda,
    __global const half* restrict B, int ldb,
    __global half* restrict C, int ldc)
{
    // 워크그룹(타일) 좌표: M 방향 / N 방향
    const int wg_row = get_group_id(0);
    const int wg_col = get_group_id(1);

    // 워크아이템(타일 내부) 좌표
    const int lid_row = get_local_id(0);
    const int lid_col = get_local_id(1);

    const int global_row = wg_row * TILE_M + lid_row;
    const int global_col = wg_col * TILE_N + lid_col;

    __local half As[TILE_M * TILE_K];
    __local half Bs[TILE_K * TILE_N];

    float acc = 0.0f;
    const int num_tiles = (K + TILE_K - 1) / TILE_K;

    for (int t = 0; t < num_tiles; ++t) {
        const int k_base = t * TILE_K;

        // A 타일 로드 [TILE_M x TILE_K]
        int a_row = global_row;
        int a_col = k_base + lid_col;
        if (a_row < M && a_col < K) {
            As[lid_row * TILE_K + lid_col] = A[a_row * lda + a_col];
        } else {
            As[lid_row * TILE_K + lid_col] = (half)0.0h;
        }

        // B 타일 로드 [TILE_K x TILE_N]
        int b_row = k_base + lid_row;
        int b_col = global_col;
        if (b_row < K && b_col < N) {
            Bs[lid_row * TILE_N + lid_col] = B[b_row * ldb + b_col];
        } else {
            Bs[lid_row * TILE_N + lid_col] = (half)0.0h;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // 타일 내부 곱셈-누적
        for (int kk = 0; kk < TILE_K; ++kk) {
            float a_val = convert_float(As[lid_row * TILE_K + kk]);
            float b_val = convert_float(Bs[kk * TILE_N + lid_col]);
            acc += a_val * b_val;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 결과 저장
    if (global_row < M && global_col < N) {
        C[global_row * ldc + global_col] = convert_half(acc);
    }
}

