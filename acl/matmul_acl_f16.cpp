#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/core/Types.h"

#include <cstdlib>
#include <cstdint>

using namespace arm_compute;

struct ACLMatmulHandle {
    CLTensor A, B, C;
    CLGEMM gemm;
    int M, N, K;
    bool b_ready = false;
};


extern "C" {

void* matmul_acl_f16_create(int M, int N, int K) {
    static bool env_initialized = false;
    if (!env_initialized) {
        CLScheduler::get().default_init();
        env_initialized = true;
    }

    auto* h = new ACLMatmulHandle();
    h->M = M; h->N = N; h->K = K;

    // ACL TensorShape: (Width, Height) -> (K, M) 순서
    const DataType dt = DataType::F16;
    h->A.allocator()->init(TensorInfo(TensorShape(K, M), 1, dt));
    h->B.allocator()->init(TensorInfo(TensorShape(N, K), 1, dt));
    h->C.allocator()->init(TensorInfo(TensorShape(N, M), 1, DataType::F32));

    // GEMM 설정 (커널 컴파일 및 최적화 경로 탐색)
    h->gemm.configure(&h->A, &h->B, nullptr, &h->C, 1.0f, 0.0f, GEMMInfo());

    // 실제 GPU 메모리 할당
    h->A.allocator()->allocate();
    h->B.allocator()->allocate();
    h->C.allocator()->allocate();

    return (void*)h;
}

int matmul_acl_f16_set_B(void* handle, const uint16_t* hB) {
    if (!handle || !hB) return -1;
    auto* h = (ACLMatmulHandle*)handle;

    h->B.map(true);
    std::memcpy(h->B.buffer(), hB, sizeof(uint16_t) * h->K * h->N);
    h->B.unmap();
    
    h->b_ready = true;
    return 0;
}

int matmul_acl_f16_run(void* handle, const uint16_t* hA, float* hC) {
    if (!handle || !hA || !hC) return -1;
    auto* h = (ACLMatmulHandle*)handle;

    if (!h->b_ready) {
        // B 행렬이 설정되지 않음
        return -2;
    }

    // A 행렬 복사
    h->A.map(true);
    std::memcpy(h->A.buffer(), hA, sizeof(uint16_t) * h->M * h->K);
    h->A.unmap();

    // GEMM 실행
    h->gemm.run();
    CLScheduler::get().sync();

    // C 행렬 복사
    h->C.map(true);
    std::memcpy(hC, h->C.buffer(), sizeof(float) * h->M * h->N);
    h->C.unmap();

    return 0;
}

void matmul_acl_f16_destroy(void* handle) {
    if (!handle) return;
    auto* h = (ACLMatmulHandle*)handle;
    delete h;
}

int matmul_acl_f16(int M, int N, int K,
               const uint16_t* hA,  // host A (FP16 비트패턴)
               const uint16_t* hB,  // host B
               float* hC)        // host C (out)
{
    void* handle = matmul_acl_f16_create(M, N, K);
    if (!handle) return -1;

    int ret = matmul_acl_f16_set_B(handle, hB);
    if (ret != 0) {
        matmul_acl_f16_destroy(handle);
        return ret;
    }

    ret = matmul_acl_f16_run(handle, hA, hC);
    matmul_acl_f16_destroy(handle);
    return ret;
}

int matmul_acl_f16_old(int M, int N, int K,
               const uint16_t* hA,  // host A (FP16 비트패턴)
               const uint16_t* hB,  // host B
               float* hC)        // host C (out)
{
    static bool initialized = false;
    if (!initialized) {
        CLScheduler::get().default_init();  // Mali OpenCL 컨텍스트 1회 초기화
        initialized = true;
    }

    CLTensor A, B, C;

    // ACL에서 TensorShape는 (width, height) = (X, Y) 순서라
    // 너가 기존에 쓰던 대로 [K, M], [N, K], [N, M] 유지
    TensorShape shapeA(K, M); // [K, M]
    TensorShape shapeB(N, K); // [N, K]
    TensorShape shapeC(N, M); // [N, M]

    const DataType dt = DataType::F16;

    A.allocator()->init(TensorInfo(shapeA, 1, dt));
    B.allocator()->init(TensorInfo(shapeB, 1, dt));
    C.allocator()->init(TensorInfo(shapeC, 1, dt));

    CLGEMM gemm;
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    GEMMInfo gemm_info;

    gemm.configure(&A, &B, nullptr, &C, alpha, beta, gemm_info);

    A.allocator()->allocate();
    B.allocator()->allocate();
    C.allocator()->allocate();

    // -------- Host → Device copy (메모리 이동 포함) --------
    A.map(true); // blocking map
    std::memcpy(A.buffer(), hA, sizeof(uint16_t) * M * K);
    A.unmap();

    B.map(true);
    std::memcpy(B.buffer(), hB, sizeof(uint16_t) * K * N);
    B.unmap();

    // -------- GEMM 실행 --------
    gemm.run();
    CLScheduler::get().sync();

    // -------- Device → Host copy --------
    C.map(true);
    std::memcpy(hC, C.buffer(), sizeof(float) * M * N);
    C.unmap();

    return 0;
}

} // extern "C"
