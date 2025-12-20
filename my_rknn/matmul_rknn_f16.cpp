#include <stdint.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <cmath>

#include <stdio.h>

#include "rknn_matmul_api.h"

static inline int align_up(int x, int a) { return ((x + a - 1) / a) * a; }

// A normal [M,K] -> AC_layout=1 packed A : (K/8, M, 8)
static void pack_A_rk3588_f16_ac1(
    const uint16_t* A, int M, int K,
    uint16_t* Ap, int Kp
) {
    const int KB = Kp / 8;
    for (int kb = 0; kb < KB; ++kb) {
        for (int m = 0; m < M; ++m) {
            for (int ki = 0; ki < 8; ++ki) {
                const int k = kb * 8 + ki;
                uint16_t v = 0;
                if (k < K) v = A[m * K + k];
                Ap[((kb * M + m) * 8) + ki] = v;
            }
        }
    }
}


static void pack_B_rk3588_f16_native_v2(
    const uint16_t* B, int K, int N,
    uint16_t* Bp, int Kp, int Np
) {
    memset(Bp, 0, Kp * Np * sizeof(uint16_t));
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            const int b_n = n / 16;
            const int i_n = n % 16;
            const int b_k = k / 32;
            const int i_k = k % 32;
            const int idx = (((b_n * (Kp / 32) + b_k) * 16 + i_n) * 32 + i_k);
            Bp[idx] = B[k * N + n];
        }
    }
}

// B normal [K,N] -> B_layout=1 packed B : (N/16, K/32, 16, 32)
static void pack_B_rk3588_f16_native(
    const uint16_t* B, int K, int N,
    uint16_t* Bp, int Kp, int Np
) {
    const int NB = Np / 16;
    const int KB = Kp / 32;
    for (int nb = 0; nb < NB; ++nb) {
        for (int kb = 0; kb < KB; ++kb) {
            for (int ni = 0; ni < 16; ++ni) {
                for (int ki = 0; ki < 32; ++ki) {
                    const int k = kb * 32 + ki;
                    const int n = nb * 16 + ni;
                    uint16_t v = 0;
                    if (k < K && n < N) v = B[k * N + n];
                    const int idx = (((nb * KB + kb) * 16 + ni) * 32 + ki);
                    Bp[idx] = v;
                }
            }
        }
    }
}

template <typename Ti, typename To>
void norm_layout_to_perf_layout(Ti *src, To *dst, int32_t M, int32_t K, int32_t subK, bool isInt4Type)
{
    int outter_size = (int)std::ceil(K * 1.0f / subK);
    for (int i = 0; i < outter_size; i++)
    {
        for (int m = 0; m < M; m++)
        {
            for (int j = 0; j < subK; j++)
            {
                int ki = i * subK + j;
                if (isInt4Type)
                {
                    int input_index = m * K + ki;
                    int output_index = i * M * subK + m * subK + j;
                    int8_t int4 = src[input_index];
                    if (ki >= K)
                    {
                        int4 = 0;
                    }
                    else
                    {
                        int4 = int4 & 0xf;
                    }
                    if (output_index % 2 == 0)
                    {
                        dst[output_index / 2] = int4;
                    }
                    else
                    {
                        int8_t temp = dst[output_index / 2];
                        int8_t result = temp | (int4 << 4);
                        dst[output_index / 2] = result;
                    }
                }
                else
                {
                    if (ki >= K)
                    {
                        dst[i * M * subK + m * subK + j] = 0;
                    }
                    else
                    {
                        dst[i * M * subK + m * subK + j] = src[m * K + ki];
                    }
                }
            }
        }
    }
}

//template void norm_layout_to_perf_layout<int8_t, int8_t>(int8_t *src, int8_t *dst, int32_t M, int32_t K, int32_t subK, bool isInt4Type);
//template void norm_layout_to_perf_layout<float16, float16>(float16 *src, float16 *dst, int32_t M, int32_t K, int32_t subK, bool isInt4Type);



// C packed (N/4, M, 4) -> normal [M,N]
static void unpack_C_rk3588_fp32_ac1(
    const float* Cp, int M, int N,
    float* C, int Np
) {
    const int NB = Np / 4;
    for (int m = 0; m < M; ++m) {
        for (int nb = 0; nb < NB; ++nb) {
            for (int ni = 0; ni < 4; ++ni) {
                const int n = nb * 4 + ni;
                if (n < N) {
                    C[m * N + n] = Cp[((nb * M + m) * 4) + ni];
                }
            }
        }
    }
}

struct MatmulHandleF16 {
    int M=0, K=0, N=0;
    int Kp=0, Np=0;

    rknn_matmul_ctx ctx = 0;
    rknn_matmul_io_attr io_attr{};
    rknn_tensor_mem* memA = nullptr;
    rknn_tensor_mem* memB = nullptr;
    rknn_tensor_mem* memC = nullptr;

    bool b_ready = false;
};

extern "C" {

// create handle for fixed (M,K,N)
void* matmul_rknn_f16_create(int M, int K, int N, int core_mask=RKNN_NPU_CORE_ALL) {
    if (M <= 0 || K <= 0 || N <= 0) return nullptr;

    auto* h = new MatmulHandleF16();
    h->M = M; h->K = K; h->N = N;
    h->Kp = align_up(K, 32);
    h->Np = align_up(N, 16);

    rknn_matmul_info info;
    memset(&info, 0, sizeof(info));
    info.M = h->M;
    info.K = h->Kp;
    info.N = h->Np;
    info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    info.B_layout = 0;   // packed B
    info.AC_layout = 0;  // packed A/C (matches demo behavior)

    memset(&h->io_attr, 0, sizeof(h->io_attr));

    int ret = rknn_matmul_create(&h->ctx, &info, &h->io_attr);
    if (ret != 0) { delete h; return nullptr; }

    rknn_matmul_set_core_mask(h->ctx, (rknn_core_mask)core_mask);

    h->memA = rknn_create_mem(h->ctx, h->io_attr.A.size);
    h->memB = rknn_create_mem(h->ctx, h->io_attr.B.size);
    h->memC = rknn_create_mem(h->ctx, h->io_attr.C.size);
    if (!h->memA || !h->memB || !h->memC) {
        if (h->memA) rknn_destroy_mem(h->ctx, h->memA);
        if (h->memB) rknn_destroy_mem(h->ctx, h->memB);
        if (h->memC) rknn_destroy_mem(h->ctx, h->memC);
        rknn_matmul_destroy(h->ctx);
        delete h;
        return nullptr;
    }

    ret = rknn_matmul_set_io_mem(h->ctx, h->memA, &h->io_attr.A); if (ret) { /* fallthrough */ }
    ret = rknn_matmul_set_io_mem(h->ctx, h->memB, &h->io_attr.B); if (ret) { /* fallthrough */ }
    ret = rknn_matmul_set_io_mem(h->ctx, h->memC, &h->io_attr.C); if (ret) { /* fallthrough */ }

    return (void*)h;
}

// set W once (recommended: W is constant weight)
int matmul_rknn_f16_set_B(void* handle, const uint16_t* W) {
    if (!handle || !W) return -1;
    auto* h = (MatmulHandleF16*)handle;

    uint16_t* Bdst = (uint16_t*)h->memB->virt_addr;
    pack_B_rk3588_f16_native_v2(W, h->K, h->N, Bdst, h->Kp, h->Np);
    //memcpy(h->memB->virt_addr, W, h->K * h->N * sizeof(uint16_t));

    h->b_ready = true;
    return 0;
}

// run once: given X, produce C_out (FP32 [M,N])
int matmul_rknn_f16_run(void* handle, const uint16_t* X, float* C_out) {
    if (!handle || !X || !C_out) return -1;
    auto* h = (MatmulHandleF16*)handle;
    //if (!h->b_ready) return -2; // call set_B first

    // pack A into memA
    //uint16_t* Adst = (uint16_t*)h->memA->virt_addr;
    //pack_A_rk3588_f16_ac1(X, h->M, h->K, Adst, h->Kp);
    memcpy(h->memA->virt_addr, X, h->M * h->K * sizeof(uint16_t));

    int ret = rknn_matmul_run(h->ctx);
    if (ret != 0) return ret;

    // unpack C from memC
    //const float* Csrc = (const float*)h->memC->virt_addr;
    //unpack_C_rk3588_fp32_ac1(Csrc, h->M, h->N, C_out, h->Np);
    memcpy(C_out, h->memC->virt_addr, h->M * h->N * sizeof(float));
    return 0;
}

void matmul_rknn_f16_destroy(void* handle) {
    if (!handle) return;
    auto* h = (MatmulHandleF16*)handle;

    if (h->memA) rknn_destroy_mem(h->ctx, h->memA);
    if (h->memB) rknn_destroy_mem(h->ctx, h->memB);
    if (h->memC) rknn_destroy_mem(h->ctx, h->memC);
    rknn_matmul_destroy(h->ctx);

    delete h;
}

int matmul_rknn_f16(int M, int N, int K, 
                const uint16_t* X, const uint16_t* W,
                float* C_out, int core_mask = RKNN_NPU_CORE_ALL) {
    //printf("matmul_rknn_f16 M=%d N=%d K=%d core_mask=0x%X\n", M, N, K, core_mask);
    void* handle = matmul_rknn_f16_create(M, K, N, core_mask);
    if (!handle) return -1;
    int ret = matmul_rknn_f16_set_B(handle, W);
    if (ret != 0) {
        matmul_rknn_f16_destroy(handle);
        return ret;
    }
    ret = matmul_rknn_f16_run(handle, X, C_out);
    matmul_rknn_f16_destroy(handle);
    
    return 0; 
}

} // extern "C"

