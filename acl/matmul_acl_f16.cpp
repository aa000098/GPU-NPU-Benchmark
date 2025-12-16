#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/core/Types.h"

#include <cstdlib>
#include <cstdint>

using namespace arm_compute;

extern "C"
int matmul_acl_f16(int M, int N, int K,
               const uint16_t* hA,  // host A (FP16 비트패턴)
               const uint16_t* hB,  // host B
               uint16_t* hC)        // host C (out)
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
    std::memcpy(hC, C.buffer(), sizeof(uint16_t) * M * N);
    C.unmap();

    return 0;
}

