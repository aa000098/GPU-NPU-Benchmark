#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/functions/CLSoftmaxLayer.h"  // ✅ softmax
#include "arm_compute/core/Types.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>      // ✅ memcpy
#include <stdexcept>

using namespace arm_compute;

static inline void copy_host_to_tensor_u16_2d(CLTensor &t, const uint16_t *src, int M, int N)
{
    // TensorShape(N, M)에서 strides[1] == 한 row(=Y 한 칸) 이동 바이트
    const size_t row_stride = t.info()->strides_in_bytes()[1];
    const size_t row_bytes  = (size_t)N * sizeof(uint16_t);
    uint8_t *dst = t.buffer();

    // row_stride가 row_bytes보다 크면 padding이 있다는 뜻
    for(int r = 0; r < M; ++r)
        std::memcpy(dst + (size_t)r * row_stride, src + (size_t)r * N, row_bytes);
}

static inline void copy_tensor_to_host_u16_2d(const CLTensor &t, uint16_t *dst, int M, int N)
{
    const size_t row_stride = t.info()->strides_in_bytes()[1];
    const size_t row_bytes  = (size_t)N * sizeof(uint16_t);
    const uint8_t *src = t.buffer();

    for(int r = 0; r < M; ++r)
        std::memcpy(dst + (size_t)r * N, src + (size_t)r * row_stride, row_bytes);
}

extern "C"
int softmax_acl_f16(int M, int N,
                    const uint16_t* hX,   // host input (FP16 bitpattern), size M*N
                    uint16_t* hY,         // host output (FP16 bitpattern), size M*N
                    float beta,
                    int axis)
{
    static bool initialized = false;
    if (!initialized) {
        CLScheduler::get().default_init(); // Mali OpenCL 컨텍스트 1회 초기화
        initialized = true;
    }

    CLTensor X, Y;

    // X/Y shape: (width=N, height=M) == [N, M]
    TensorShape shape(N, M);
    const DataType dt = DataType::F16;

    X.allocator()->init(TensorInfo(shape, 1, dt));
    Y.allocator()->init(TensorInfo(shape, 1, dt));

    CLSoftmaxLayer softmax;

    // row별 softmax: axis=0 (X dimension=width=N 방향)
    softmax.configure(&X, &Y, beta, axis);

    X.allocator()->allocate();
    Y.allocator()->allocate();

    // Host -> Device
    X.map(true);
//    std::memcpy(X.buffer(), hX, sizeof(uint16_t) * (size_t)M * (size_t)N);
    copy_host_to_tensor_u16_2d(X, hX, M, N);
    X.unmap();

    // Run
    softmax.run();
    CLScheduler::get().sync();

    // Device -> Host
    Y.map(true);
    // std::memcpy(hY, Y.buffer(), sizeof(uint16_t) * (size_t)M * (size_t)N);
    copy_tensor_to_host_u16_2d(Y, hY, M, N);
    Y.unmap();

    return 0;
}
