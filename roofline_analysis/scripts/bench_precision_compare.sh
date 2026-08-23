#!/bin/bash
# Precision 비교: Low(INT8 attn) vs Normal(FP16) vs High(FP32) on CPU and GPU
# GPU가 FP16 네이티브에서 제 성능을 발휘하는지 확인
#
# Usage:
#   bash scripts/bench_precision_compare.sh

set -e

LLM_BENCH="/home/hyunho.son/install_files/MNN/build/llm_bench"
M9_CONFIG="/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/mnn_models/M9_w8a8_ch_direct/config.json"
RESULTS_DIR="/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/roofline_analysis/results/raw"

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Precision Comparison: Low vs Normal vs High on CPU and GPU"
echo "Model: M9 W8A8"
echo "============================================================"

# CPU - Low precision (INT8 attention)
echo ""
echo "--- CPU, precision=Low (INT8 attention) ---"
$LLM_BENCH -m "$M9_CONFIG" -a cpu -t 4 -c 2 -qatten 1 -p 64,256 -n 128 -rep 10 2>&1 | tee "${RESULTS_DIR}/precision_cpu_low.log"

# CPU - Normal precision (FP16)
echo ""
echo "--- CPU, precision=Normal (FP16) ---"
$LLM_BENCH -m "$M9_CONFIG" -a cpu -t 4 -c 0 -qatten 0 -p 64,256 -n 128 -rep 10 2>&1 | tee "${RESULTS_DIR}/precision_cpu_normal.log"

# CPU - High precision (FP32)
echo ""
echo "--- CPU, precision=High (FP32) ---"
$LLM_BENCH -m "$M9_CONFIG" -a cpu -t 4 -c 1 -qatten 0 -p 64,256 -n 128 -rep 10 2>&1 | tee "${RESULTS_DIR}/precision_cpu_high.log"

# GPU - Low precision (INT8 attention)
echo ""
echo "--- GPU, precision=Low (INT8 attention) ---"
$LLM_BENCH -m "$M9_CONFIG" -a opencl -t 4 -c 2 -qatten 1 -p 64,256 -n 128 -rep 10 2>&1 | tee "${RESULTS_DIR}/precision_gpu_low.log"

# GPU - Normal precision (FP16)
echo ""
echo "--- GPU, precision=Normal (FP16) ---"
$LLM_BENCH -m "$M9_CONFIG" -a opencl -t 4 -c 0 -qatten 0 -p 64,256 -n 128 -rep 10 2>&1 | tee "${RESULTS_DIR}/precision_gpu_normal.log"

# GPU - High precision (FP32)
echo ""
echo "--- GPU, precision=High (FP32) ---"
$LLM_BENCH -m "$M9_CONFIG" -a opencl -t 4 -c 1 -qatten 0 -p 64,256 -n 128 -rep 10 2>&1 | tee "${RESULTS_DIR}/precision_gpu_high.log"

echo ""
echo "============================================================"
echo "DONE — Results in: ${RESULTS_DIR}/precision_*.log"
echo "============================================================"
echo ""
echo "NPU R1 (W8A8) reference:"
echo "  Prefill p64:  1296 tok/s"
echo "  Prefill p256: 4971 tok/s"
echo "  Decode:       20.0 tok/s"
