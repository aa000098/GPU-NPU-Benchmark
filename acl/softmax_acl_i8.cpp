#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/functions/CLSoftmaxLayer.h"
#include "arm_compute/core/Types.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

using namespace arm_compute;

// Padding을 고려한 Host <-> Tensor 복사 함수 (uint8_t 버전)
static inline void copy_host_to_tensor_u8_2d(CLTensor &t, const uint8_t *src, int M, int N)
{
    const size_t row_stride = t.info()->strides_in_bytes()[1];
    const size_t row_bytes   = (size_t)N * sizeof(uint8_t);
    uint8_t *dst = t.buffer();

    for(int r = 0; r < M; ++r)
        std::memcpy(dst + (size_t)r * row_stride, src + (size_t)r * N, row_bytes);
}

static inline void copy_tensor_to_host_u8_2d(const CLTensor &t, uint8_t *dst, int M, int N)
{
    const size_t row_stride = t.info()->strides_in_bytes()[1];
    const size_t row_bytes   = (size_t)N * sizeof(uint8_t);
    const uint8_t *src = t.buffer();

    for(int r = 0; r < M; ++r)
        std::memcpy(dst + (size_t)r * N, src + (size_t)r * row_stride, row_bytes);
}

extern "C"
int softmax_acl_i8(int M, int N,
                   const uint8_t* hX,   // host input (QASYMM8)
                   uint8_t* hY,         // host output (QASYMM8)
                   float beta,
                   int axis)
{
    static bool initialized = false;
    if (!initialized) {
        CLScheduler::get().default_init();
        initialized = true;
    }

    CLTensor X, Y;

    // 양자화 정보 설정 (Softmax 입력은 보통 Scale과 Offset이 필요함)
    // 여기서는 성능 측정을 위해 Scale 1.0, Offset 0인 기본값 사용
    const QuantizationInfo qinfo(1.0f / 256.0f, 0); // 예시: 0~1 범위를 표현하는 quantization
    const DataType dt = DataType::QASYMM8;

    TensorShape shape(N, M);
    X.allocator()->init(TensorInfo(shape, 1, dt, qinfo));
    Y.allocator()->init(TensorInfo(shape, 1, dt, qinfo));

    CLSoftmaxLayer softmax;

    // ACL Softmax 설정
    softmax.configure(&X, &Y, beta, axis);

    X.allocator()->allocate();
    Y.allocator()->allocate();

    // Host -> Device (Input)
    X.map(true);
    copy_host_to_tensor_u8_2d(X, hX, M, N);
    X.unmap();

    // GPU 실행
    softmax.run();
    CLScheduler::get().sync();

    // Device -> Host (Output)
    Y.map(true);
    copy_tensor_to_host_u8_2d(Y, hY, M, N);
    Y.unmap();

    return 0;
}
