#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void softmax_fp16(
    __global const half* restrict X,
    __global half* restrict Y,
    int D)
{
    int row  = get_global_id(0);
    int base = row * D;

    // 1) row별 최대값 (float로 계산)
    float max_val = -INFINITY;
    for (int i = 0; i < D; i++) {
        float v = convert_float(X[base + i]);
        max_val = fmax(max_val, v);
    }

    // 2) exp(x - max)를 한 번만 계산해서 Y에 저장 + sum(float) 누적
    float sum = 0.0f;
    for (int i = 0; i < D; i++) {
        float v = convert_float(X[base + i]);
        float e = native_exp(v - max_val); // 정확도가 중요하면 exp()로 바꿔도 됨
        sum += e;
        Y[base + i] = convert_half(e);     // 임시로 e를 Y에 저장
    }

    // 3) 정규화
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < D; i++) {
        float e = convert_float(Y[base + i]);  // 아까 저장한 exp 값
        Y[base + i] = convert_half(e * inv_sum);
    }
}
