#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLGEMM.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/core/Types.h"

#include <cstdlib>

using namespace arm_compute;

extern "C" int matmul_acl(int M, int N, int K)
{
    // CLScheduler 초기화는 한 번만 하도록 static 처리 (원하면 제거해도 됨)
    static bool initialized = false;
    if (!initialized)
    {
        CLScheduler::get().default_init();
        initialized = true;
    }

    CLTensor A, B, C;

    // ACL TensorShape는 [width, height] 순서 (x, y)
    TensorShape shapeA((unsigned int)K, (unsigned int)M); // [K, M]
    TensorShape shapeB((unsigned int)N, (unsigned int)K); // [N, K]
    TensorShape shapeC((unsigned int)N, (unsigned int)M); // [N, M]

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

    // 실제에서는 A/B에 값 채우고 싶으면 map/unmap 해서 써도 됨
    gemm.run();
    CLScheduler::get().sync();

    return 0; // 에러 코드 없으니 그냥 0 리턴
}
