#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/functions/CLGEMMLowpOutputStage.h"
#include "arm_compute/core/Types.h"
#include "utils/Utils.h"
#include <vector>
#include <cmath>

using namespace arm_compute;
using namespace utils;

/**
 * INT8 Attention Pipeline
 * Data Type: QASYMM8_SIGNED (int8)
 * Flow: GEMM -> Softmax -> GEMM
 * Note: Scaling (1/sqrt(d)) is assumed to be fused into quantization parameters (zero cost)
 */
class AttentionPipelineI8 {
public:
    // Tensors
    CLTensor q, k, v, out;
    CLTensor qk, probs;

    CLTensor qk_s32, out_s32;

    // Operators
    CLGEMMLowpMatrixMultiplyCore gemm1; // Q @ K^T
    CLSoftmaxLayer softmax;
    CLGEMMLowpMatrixMultiplyCore gemm2; // Probs @ V

    // Quant
    CLGEMMLowpOutputStage quant1; // S32 -> U8
    CLGEMMLowpOutputStage quant2; // S32 -> U8
    
    int M, D;

    // [수정 1] 패딩을 고려하여 데이터를 넣는 헬퍼 함수
    void fill_tensor(CLTensor& tensor, void* data_ptr) {
        tensor.map(true);
        Window window;
        window.use_tensor_dimensions(tensor.info()->tensor_shape());
        
        // Iterator를 사용해 패딩을 건너뛰며 복사
        Iterator it(&tensor, window);
        uint8_t* src = (uint8_t*)data_ptr;
        size_t row_size = tensor.info()->dimension(0) * tensor.info()->element_size(); // 한 행의 실제 데이터 크기 (byte)

        execute_window_loop(window, [&](const Coordinates& id) {
            // id[1]은 현재 행(row) 인덱스입니다.
            // 소스 포인터 위치: 시작점 + (현재 행 번호 * 한 행의 크기)
            std::memcpy(it.ptr(), src + (id[1] * row_size), row_size);
        }, it);
        
        tensor.unmap();
    }

    // [수정 2] 출력 데이터를 가져오는 헬퍼 함수
    void read_tensor(CLTensor& tensor, void* out_ptr) {
        tensor.map(true);
        Window window;
        window.use_tensor_dimensions(tensor.info()->tensor_shape());
        
        Iterator it(&tensor, window);
        uint8_t* dst = (uint8_t*)out_ptr;
        size_t row_size = tensor.info()->dimension(0) * tensor.info()->element_size();

        execute_window_loop(window, [&](const Coordinates& id) {
            std::memcpy(dst + (id[1] * row_size), it.ptr(), row_size);
        }, it);
        
        tensor.unmap();
    }

    void configure(int M, int D) {
        this->M = M;
        this->D = D;

        // 1. Shapes
        TensorShape shape_q(D, M);
        TensorShape shape_k(M, D);
        TensorShape shape_v(D, M);
        TensorShape shape_out(D, M);
        TensorShape shape_inter(M, M); // [M, M]
        TensorShape shape_out_s32(D, M);

        // 2. Quantization Info (Arbitrary values for benchmark)
        // Scale=0.01, Offset=0 (Symmetric)
        QuantizationInfo q_info(0.01f, 0);

        // 3. Init Tensors (QASYMM8_SIGNED = int8)
        DataType dt = DataType::QASYMM8;
        q.allocator()->init(TensorInfo(shape_q, 1, dt, q_info));
        k.allocator()->init(TensorInfo(shape_k, 1, dt, q_info));
        v.allocator()->init(TensorInfo(shape_v, 1, dt, q_info));
        out.allocator()->init(TensorInfo(shape_out, 1, dt, q_info));

        qk.allocator()->init(TensorInfo(shape_inter, 1, dt, q_info));
        probs.allocator()->init(TensorInfo(shape_inter, 1, dt, QuantizationInfo(1.0f/256.0f, 0))); // Softmax out

        // 중간 S32 (GEMM 결과 저장용)
        qk_s32.allocator()->init(TensorInfo(shape_inter, 1, DataType::S32));
        out_s32.allocator()->init(TensorInfo(shape_out_s32, 1, DataType::S32));
        
        // 4. Configure Operators

        // Step A: GEMM1 (Q @ K^T) -> QK
        //GEMMInfo info1;
        //info1.set_pretranspose_B(true); // Transpose K
        //gemm1.configure(&q, &k, nullptr, &qk, 1.0f, 0.0f, info1);

        GEMMLowpOutputStageInfo info_quant;
        info_quant.type = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
        info_quant.gemmlowp_multiplier = 123456789; // 임의 값
        info_quant.gemmlowp_shift = 1;
        info_quant.gemmlowp_offset = 0;
        info_quant.gemmlowp_min_bound = 0;
        info_quant.gemmlowp_max_bound = 255;
        info_quant.output_data_type = dt;

        GEMMInfo gemm_info;

        //gemm1.configure(&q, &k, nullptr, &qk, gemm_info);
        gemm1.configure(&q, &k, nullptr, &qk_s32, gemm_info);
        quant1.configure(&qk_s32, nullptr, &qk, info_quant);

        // Step B: Softmax (QK -> Probs)
        // Note: Scaling (1/sqrt(d)) is mathematically fused into qk's scale in a real scenario.
        softmax.configure(&qk, &probs);

        // Step C: GEMM2 (Probs @ V) -> Out
        //gemm2.configure(&probs, &v, nullptr, &out, gemm_info);
        gemm2.configure(&probs, &v, nullptr, &out_s32, gemm_info);
        quant2.configure(&out_s32, nullptr, &out, info_quant);

        // 5. Allocate Intermediate
        qk.allocator()->allocate();
        probs.allocator()->allocate();

        q.allocator()->allocate();
        k.allocator()->allocate();
        v.allocator()->allocate();
        out.allocator()->allocate();

        qk_s32.allocator()->allocate();
        out_s32.allocator()->allocate();
    }

    void run(void* q_ptr, void* k_ptr, void* v_ptr, void* out_ptr) {
        //q.allocator()->import_memory(q_ptr);
        //k.allocator()->import_memory(k_ptr);
        //v.allocator()->import_memory(v_ptr);
        //out.allocator()->import_memory(out_ptr);

        size_t input_size = M * D * sizeof(int8_t);
        q.map(true);
        std::memcpy(q.buffer(), q_ptr, input_size);
        q.unmap();

        k.map(true);
        std::memcpy(k.buffer(), k_ptr, input_size);
        k.unmap();

        v.map(true);
        std::memcpy(v.buffer(), v_ptr, input_size);
        v.unmap();

        gemm1.run();
        quant1.run();
        softmax.run();
        gemm2.run();
        quant2.run();

        CLScheduler::get().sync();

        //q.allocator()->free();
        //k.allocator()->free();
        //v.allocator()->free();
        //out.allocator()->free();
        out.map(true);
        std::memcpy(out_ptr, out.buffer(), input_size);
        out.unmap();
    }
};

// Global Cache for INT8
static std::map<std::vector<int>, AttentionPipelineI8*> pipelines_i8;

extern "C" {
    // ... (기존 FP16 함수들) ...

    // [INT8 Create]
    void* attention_acl_i8_create(int M, int D) {
        CLScheduler::get().default_init();
        AttentionPipelineI8* pipe = new AttentionPipelineI8();
        pipe->configure(M, D);
        return (void*)pipe;
    }

    // [INT8 Run]
    int attention_acl_i8_run(void* handle, void* q_ptr, void* k_ptr, void* v_ptr, void* out_ptr) {
        AttentionPipelineI8* pipe = (AttentionPipelineI8*)handle;
        pipe->run(q_ptr, k_ptr, v_ptr, out_ptr);
        return 0;
    }
}
