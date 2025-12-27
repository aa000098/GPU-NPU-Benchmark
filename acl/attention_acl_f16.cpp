#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/core/Types.h"
#include "utils/Utils.h"
#include <vector>
#include <cmath>

using namespace arm_compute;
using namespace utils;

/**
 * Attention Pipeline Class
 * Logic: Softmax((Q @ K^T) * scale) @ V
 */
class AttentionPipeline {
public:
    // Tensors
    CLTensor q, k, v, out;           // Inputs/Output
    CLTensor score, scaled_score, probs; // Intermediates

    // Operators
    CLGEMM gemm1;                // Q @ K^T
    CLActivationLayer scaler;    // * scale
    CLSoftmaxLayer softmax;      // Softmax
    CLGEMM gemm2;                // Probs @ V
    
    int M, D; // Dimensions

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

        // 1. Tensor Shape Definition
        TensorShape shape_q(D, M); // ACL uses [x, y] -> [width, height] -> [dim, seq]
        TensorShape shape_k(M, D);
        TensorShape shape_v(D, M);
        TensorShape shape_out(D, M);
        TensorShape shape_score(M, M); // [M, M]
        
        // 2. Init Tensors Info (FP16)
        q.allocator()->init(TensorInfo(shape_q, 1, DataType::F16));
        k.allocator()->init(TensorInfo(shape_k, 1, DataType::F16));
        v.allocator()->init(TensorInfo(shape_v, 1, DataType::F16));
        out.allocator()->init(TensorInfo(shape_out, 1, DataType::F16));

        score.allocator()->init(TensorInfo(shape_score, 1, DataType::F16));
        scaled_score.allocator()->init(TensorInfo(shape_score, 1, DataType::F16));
        probs.allocator()->init(TensorInfo(shape_score, 1, DataType::F16));

        // 3. Configure Operators
        
        // Step A: GEMM1 (Q @ K^T)
        // Config GEMM1: Q(D, M) x K(D, M)^T -> Score(M, M)
        // We set reshaped information in GEMMInfo if available, but let's stick to default CLGEMM for now.
        // To perform Q * K^T, we define K shape carefully or use GEMMInfo.
        
        // 위에서 K Shape을 뒤집었으므로, 이제 일반 GEMM을 수행하면 됩니다.
        // pretranspose_B 옵션 제거 (안정성 확보)
        //GEMMInfo info1;
        //info1.set_pretranspose_B(true); // Tell GEMM to transpose B internally
        //gemm1.configure(&q, &k, nullptr, &score, 1.0f, 0.0f, info1);
        gemm1.configure(&q, &k, nullptr, &score, 1.0f, 0.0f);

        // Step B: Scale (using Activation Layer LINEAR: y = ax + b)
        float scale_val = 1.0f / sqrt((float)D);
        scaler.configure(&score, &scaled_score, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::LINEAR, scale_val, 0.0f));

        // Step C: Softmax
        softmax.configure(&scaled_score, &probs);

        // Step D: GEMM2 (Probs @ V)
        // Probs(M, M) x V(D, M) -> Out(D, M)
        // No transpose needed here.
        gemm2.configure(&probs, &v, nullptr, &out, 1.0f, 0.0f);

        // 4. Allocate Intermediate Memory
        score.allocator()->allocate();
        scaled_score.allocator()->allocate();
        probs.allocator()->allocate();

        q.allocator()->allocate();
        k.allocator()->allocate();
        v.allocator()->allocate();
        out.allocator()->allocate();
    }

    void run(void* q_ptr, void* k_ptr, void* v_ptr, void* out_ptr) {
        // Zero-copy import (assuming pointers are page-aligned and good for OpenCL)
        // If import fails, we might need map/copy/unmap, but for performance we try import.
        
        //q.allocator()->import_memory(q_ptr);
        //k.allocator()->import_memory(k_ptr);
        //v.allocator()->import_memory(v_ptr);
        //out.allocator()->import_memory(out_ptr);
        
        // If import_memory is not feasible, we can map/copy/unmap instead:
        size_t input_size = M * D * sizeof(uint16_t); // FP16 sizeof
                                                      
        q.map(true);
        std::memcpy(q.buffer(), q_ptr, input_size);
        q.unmap();

        k.map(true);
        std::memcpy(k.buffer(), k_ptr, input_size);
        k.unmap();

        v.map(true);
        std::memcpy(v.buffer(), v_ptr, input_size);
        v.unmap();

        // Execute Pipeline
        gemm1.run();
        scaler.run();
        softmax.run();
        gemm2.run();

        // Sync (ensure GPU is done before Python reads)
        CLScheduler::get().sync();
        
        // Release memory hold to avoid double-free issues if Python GC acts up, 
        // but import_memory doesn't take ownership so it's fine.
        //q.allocator()->free();
        //k.allocator()->free();
        //v.allocator()->free();
        //out.allocator()->free();
       
        // Copy output back if not using import_memory
        out.map(true);
        std::memcpy(out_ptr, out.buffer(), input_size);
        out.unmap();
    }
};

// Global handles map to keep objects alive across Python calls
static std::map<std::vector<int>, AttentionPipeline*> pipelines;

extern "C" {
    // Create & Configure Pipeline
    void* attention_acl_f16_create(int M, int D) {
        CLScheduler::get().default_init(); // Init OpenCL if not yet

        AttentionPipeline* pipe = new AttentionPipeline();
        pipe->configure(M, D);
        return (void*)pipe;
    }

    // Run Pipeline
    int attention_acl_f16_run(void* handle, void* q_ptr, void* k_ptr, void* v_ptr, void* out_ptr) {
        AttentionPipeline* pipe = (AttentionPipeline*)handle;
        pipe->run(q_ptr, k_ptr, v_ptr, out_ptr);
        return 0;
    }
}
