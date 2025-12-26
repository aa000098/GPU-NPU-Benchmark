// matmul_acl_int8.cpp (개략적인 틀)
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/core/Types.h"

#include <cstdlib>
#include <cstdint>
#include <cstring>

using namespace arm_compute;

struct ACLMatmulHandleInt8 {
    CLTensor A, B, C;
    CLGEMMLowpMatrixMultiplyCore gemm_lowp; // INT8용 GEMM 코어
    int M, N, K;
    bool b_ready = false;
};

extern "C" {

void* matmul_acl_i8_create(int M, int N, int K) {
    static bool env_initialized = false;
    if (!env_initialized) {
        CLScheduler::get().default_init();
        env_initialized = true;
    }

    auto* h = new ACLMatmulHandleInt8();
    h->M = M; h->N = N; h->K = K;

    // INT8을 위한 양자화 정보 (Scale: 1.0, Offset: 0)
    // 실제 모델에서는 각 텐서마다 고유의 scale/offset이 필요하지만,
    // 단순 연산 성능 측정을 위해 0으로 설정합니다.
    const QuantizationInfo q_info(1.0f, 0);
    const DataType dt_in = DataType::QASYMM8; // 0~255 (또는 S8 사용 가능)
    const DataType dt_out = DataType::S32;    // INT8 x INT8 결과는 보통 S32로 나옴

    // ACL TensorShape: (Width, Height) -> (K, M) 순서
    h->A.allocator()->init(TensorInfo(TensorShape(K, M), 1, dt_in, q_info));
    h->B.allocator()->init(TensorInfo(TensorShape(N, K), 1, dt_in, q_info));
    h->C.allocator()->init(TensorInfo(TensorShape(N, M), 1, dt_out, q_info));

    // GEMM Lowp 설정
    h->gemm_lowp.configure(&h->A, &h->B, nullptr, &h->C);

    // 실제 GPU 메모리 할당
    h->A.allocator()->allocate();
    h->B.allocator()->allocate();
    h->C.allocator()->allocate();

    return (void*)h;
}

int matmul_acl_i8_set_B(void* handle, const uint8_t* hB) {
    if (!handle || !hB) return -1;
    auto* h = (ACLMatmulHandleInt8*)handle;

    h->B.map(true);
    std::memcpy(h->B.buffer(), hB, sizeof(uint8_t) * h->K * h->N);
    h->B.unmap();

    h->b_ready = true;
    return 0;
}

int matmul_acl_i8_run(void* handle, const uint8_t* hA, int32_t* hC) {
    if (!handle || !hA || !hC) return -1;
    auto* h = (ACLMatmulHandleInt8*)handle;

    if (!h->b_ready) return -2;

    // A 행렬 복사
    h->A.map(true);
    std::memcpy(h->A.buffer(), hA, sizeof(uint8_t) * h->M * h->K);
    h->A.unmap();

    // GEMM 실행
    h->gemm_lowp.run();
    CLScheduler::get().sync();

    // C 행렬 복사 (S32 타입으로 복사)
    h->C.map(true);
    std::memcpy(hC, h->C.buffer(), sizeof(int32_t) * h->M * h->N);
    h->C.unmap();

    return 0;
}

void matmul_acl_i8_destroy(void* handle) {
    if (!handle) return;
    auto* h = (ACLMatmulHandleInt8*)handle;
    delete h;
}

// 원샷 호출 버전
int matmul_acl_i8(int M, int N, int K,
                 const uint8_t* hA,
                 const uint8_t* hB,
                 int32_t* hC)
{
    void* handle = matmul_acl_i8_create(M, N, K);
    if (!handle) return -1;

    int ret = matmul_acl_i8_set_B(handle, hB);
    if (ret == 0) {
        ret = matmul_acl_i8_run(handle, hA, hC);
    }

    matmul_acl_i8_destroy(handle);
    return ret;
}

} // extern "C"
