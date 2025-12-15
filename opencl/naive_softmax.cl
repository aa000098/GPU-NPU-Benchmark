#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void softmax_fp16(
    __global half* X,
    __global half* Y,
    int D)
{
    int row = get_global_id(0);

    // row 단위 softmax
    half max_val = (half)-65504.0f;
    for (int i=0; i<D; i++)
        max_val = max(max_val, X[row * D + i]);

    half sum = (half)0.0f;
    for (int i=0; i<D; i++)
        sum += exp(X[row * D + i] - max_val);

    for (int i=0; i<D; i++)
        Y[row * D + i] = exp(X[row * D + i] - max_val) / sum;
}

