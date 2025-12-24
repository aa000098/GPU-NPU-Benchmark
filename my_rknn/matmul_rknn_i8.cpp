#include <stdint.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdio.h>

#include "rknn_matmul_api.h"

static inline int align_up(int x, int a) { return ((x + a - 1) / a) * a; }

/**
 * RK3588 INT8 MatMul 전용 B Matrix 패킹 함수
 * B layout: [K, N] -> [N/16, K/32, 16, 32]
 */
static void pack_B_rk3588_i8(
    const int8_t* B, int K, int N,
    int8_t* Bp, int Kp, int Np
) {
    memset(Bp, 0, Kp * Np * sizeof(int8_t));
    const int NB = Np / 16;
    const int KB = Kp / 32;
    for (int nb = 0; nb < NB; ++nb) {
        for (int kb = 0; kb < KB; ++kb) {
            for (int ni = 0; ni < 16; ++ni) {
                for (int ki = 0; ki < 32; ++ki) {
                    const int k = kb * 32 + ki;
                    const int n = nb * 16 + ni;
                    int8_t v = 0;
                    if (k < K && n < N) v = B[k * N + n];
                    const int idx = (((nb * KB + kb) * 16 + ni) * 32 + ki);
                    Bp[idx] = v;
                }
            }
        }
    }
}

/**
 * RK3588 INT8 MatMul 전용 A Matrix 패킹 함수 (AC_layout=0 기준)
 * A layout: [M, K] -> [K/32, M, 32]
 */
static void pack_A_rk3588_i8(
    const int8_t* A, int M, int K,
    int8_t* Ap, int Kp
) {
    memset(Ap, 0, M * Kp * sizeof(int8_t));
    const int KB = Kp / 32;
    for (int kb = 0; kb < KB; ++kb) {
        for (int m = 0; m < M; ++m) {
            for (int ki = 0; ki < 32; ++ki) {
                const int k = kb * 32 + ki;
                int8_t v = 0;
                if (k < K) v = A[m * K + k];
                Ap[((kb * M + m) * 32) + ki] = v;
            }
        }
    }
}

/**
 * INT32로 출력된 C Matrix를 일반 레이아웃으로 변환
 * C layout: [N/16, M, 16] -> [M, N] (INT32)
 */
static void unpack_C_rk3588_i32(
    const int32_t* Cp, int M, int N,
    int32_t* C, int Np
) {
    const int NB = Np / 16;
    for (int m = 0; m < M; ++m) {
        for (int nb = 0; nb < NB; ++nb) {
            for (int ni = 0; ni < 16; ++ni) {
                const int n = nb * 16 + ni;
                if (n < N) {
                    C[m * N + n] = Cp[((nb * M + m) * 16) + ni];
                }
            }
        }
    }
}

struct MatmulHandleInt8 {
    int M=0, K=0, N=0;
    int Kp=0, Np=0;

    rknn_matmul_ctx ctx = 0;
    rknn_matmul_io_attr io_attr{};
    rknn_tensor_mem* memA = nullptr;
    rknn_tensor_mem* memB = nullptr;
    rknn_tensor_mem* memC = nullptr;
};

extern "C" {

void* matmul_rknn_i8_create(int M, int K, int N, int core_mask=RKNN_NPU_CORE_ALL) {
    if (M <= 0 || K <= 0 || N <= 0) return nullptr;

    auto* h = new MatmulHandleInt8();
    h->M = M; h->K = K; h->N = N;
    // INT8 MatMul의 경우 K는 32배수, N은 16배수 정렬 권장
    h->Kp = align_up(K, 32);
    h->Np = align_up(N, 32);

    rknn_matmul_info info;
    memset(&info, 0, sizeof(info));
    info.M = h->M;
    info.K = h->Kp;
    info.N = h->Np;
    info.type = RKNN_INT8_MM_INT8_TO_INT32; // INT8 연산 설정
    info.B_layout = 0;  // 0: 패킹된 데이터 입력 시
    info.AC_layout = 0; // 0: 패킹된 데이터 입력 시

    int ret = rknn_matmul_create(&h->ctx, &info, &h->io_attr);
    if (ret != 0) { delete h; return nullptr; }

    rknn_matmul_set_core_mask(h->ctx, (rknn_core_mask)core_mask);

    h->memA = rknn_create_mem(h->ctx, h->io_attr.A.size);
    h->memB = rknn_create_mem(h->ctx, h->io_attr.B.size);
    h->memC = rknn_create_mem(h->ctx, h->io_attr.C.size);

    rknn_matmul_set_io_mem(h->ctx, h->memA, &h->io_attr.A);
    rknn_matmul_set_io_mem(h->ctx, h->memB, &h->io_attr.B);
    rknn_matmul_set_io_mem(h->ctx, h->memC, &h->io_attr.C);

    return (void*)h;
}

int matmul_rknn_i8_set_B(void* handle, const int8_t* W) {
    if (!handle || !W) return -1;
    auto* h = (MatmulHandleInt8*)handle;
    int8_t* Bdst = (int8_t*)h->memB->virt_addr;
    pack_B_rk3588_i8(W, h->K, h->N, Bdst, h->Kp, h->Np);
    return 0;
}

int matmul_rknn_i8_run(void* handle, const int8_t* X, int32_t* C_out) {
    if (!handle || !X || !C_out) return -1;
    auto* h = (MatmulHandleInt8*)handle;

    int8_t* Adst = (int8_t*)h->memA->virt_addr;
    pack_A_rk3588_i8(X, h->M, h->K, Adst, h->Kp);
    //memcpy(h->memA->virt_addr, X, h->M * h->K * sizeof(int8_t));

    int ret = rknn_matmul_run(h->ctx);
    if (ret != 0) return ret;

    const int32_t* Csrc = (const int32_t*)h->memC->virt_addr;
    unpack_C_rk3588_i32(Csrc, h->M, h->N, C_out, h->Np);
    //memcpy(C_out, h->memC->virt_addr, h->M * h->N * sizeof(int32_t));
    return 0;
}

void matmul_rknn_i8_destroy(void* handle) {
    if (!handle) return;
    auto* h = (MatmulHandleInt8*)handle;
    if (h->memA) rknn_destroy_mem(h->ctx, h->memA);
    if (h->memB) rknn_destroy_mem(h->ctx, h->memB);
    if (h->memC) rknn_destroy_mem(h->ctx, h->memC);
    rknn_matmul_destroy(h->ctx);
    delete h;
}

} // extern "C"
