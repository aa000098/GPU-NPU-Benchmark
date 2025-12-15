#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void matmul_fp16_naive(
    __global half* A,
    __global half* B,
    __global half* C,
    int M, int N, int K)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < M && col < N) {
        half acc = (half)0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}
