// matmul_acl_int8.cpp (개략적인 틀)
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpMatrixMultiplyCore.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/core/Types.h"

#include <cstdint>
#include <cstring>

using namespace arm_compute;

extern "C"
int matmul_acl_int8(int M, int N, int K,
                  const int8_t* hA,
                  const int8_t* hB,
                  int8_t* hC)
{
    static bool initialized = false;
    if (!initialized)
    {
        CLScheduler::get().default_init();
        initialized = true;
    }

    CLTensor A, B, C;
    TensorShape shapeA(K, M);
    TensorShape shapeB(N, K);
    TensorShape shapeC(N, M);

    const DataType dt_in = DataType::QASYMM8_SIGNED;
    const DataType dt_out = DataType::S32;  

    TensorInfo a_info(shapeA, 1, dt_in);
    TensorInfo b_info(shapeB, 1, dt_in);
    TensorInfo c_info(shapeC, 1, dt_out);

    // 아주 단순한 quantization (scale=1, offset=0) – 실제론 튜닝 필요
    a_info.set_quantization_info(QuantizationInfo(1.0f, 0));
    b_info.set_quantization_info(QuantizationInfo(1.0f, 0));
    c_info.set_quantization_info(QuantizationInfo(1.0f, 0));

    A.allocator()->init(a_info);
    B.allocator()->init(b_info);
    C.allocator()->init(c_info);

    CLGEMMLowpMatrixMultiplyCore gemm_lowp;
    GEMMInfo gemm_info;
    gemm_lowp.configure(&A, &B, nullptr, &C, gemm_info);

    A.allocator()->allocate();
    B.allocator()->allocate();
    C.allocator()->allocate();

    A.map(true);
    std::memcpy(A.buffer(), hA, sizeof(int8_t) * M * K);
    A.unmap();

    B.map(true);
    std::memcpy(B.buffer(), hB, sizeof(int8_t) * K * N);
    B.unmap();

    gemm_lowp.run();
    CLScheduler::get().sync();

    C.map(true);
    std::memcpy(hC, C.buffer(), sizeof(int8_t) * M * N);
    C.unmap();

    return 0;
}

