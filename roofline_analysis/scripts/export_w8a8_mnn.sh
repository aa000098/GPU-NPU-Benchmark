#!/bin/bash
# W8A8 MNN 모델 Export + CPU/GPU 벤치마크
# MNN의 act_bit=8로 W8A8 모델 생성 후, CPU와 GPU(OpenCL)에서 벤치마크
#
# Usage:
#   bash scripts/export_w8a8_mnn.sh

set -e

LLMEXPORT_DIR="/home/hyunho.son/install_files/MNN/transformers/llm/export"
LLMEXPORT_SCRIPT="${LLMEXPORT_DIR}/llmexport.py"
MNNCONVERT="/home/hyunho.son/install_files/MNN/build/MNNConvert"
LLM_BENCH="/home/hyunho.son/install_files/MNN/build/llm_bench"

MODEL_PATH="/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/models/Llama-3.2-1B-Instruct"
OUTPUT_BASE="/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/mnn_models"
RESULTS_DIR="/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/roofline_analysis/results/raw"

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Step 1: Export MNN W8A8 (quant_bit=8, act_bit=8)"
echo "============================================================"

DST_PATH="${OUTPUT_BASE}/M9_w8a8_ch_direct"

if [ -f "${DST_PATH}/llm.mnn.weight" ]; then
    echo "[SKIP] M9_w8a8_ch_direct already exists"
else
    cd "$LLMEXPORT_DIR"
    python llmexport.py \
        --path "$MODEL_PATH" \
        --export mnn \
        --quant_bit 8 \
        --quant_block 0 \
        --act_bit 8 \
        --dst_path "$DST_PATH" \
        --mnnconvert "$MNNCONVERT"
    echo "[OK] Exported M9_w8a8_ch_direct"
fi

echo ""
echo "============================================================"
echo "Step 2: Benchmark M9 on CPU"
echo "============================================================"

CONFIG_PATH="${DST_PATH}/llm_config.json"
if [ ! -f "$CONFIG_PATH" ]; then
    CONFIG_PATH="${DST_PATH}/config.json"
fi

echo "--- CPU Backend ---"
$LLM_BENCH -m "$CONFIG_PATH" -a cpu -t 4 -p 64 -n 256 -rep 10 2>&1 | tee "${RESULTS_DIR}/w8a8_mnn_cpu_p64.log"
$LLM_BENCH -m "$CONFIG_PATH" -a cpu -t 4 -p 256 -n 256 -rep 10 2>&1 | tee "${RESULTS_DIR}/w8a8_mnn_cpu_p256.log"

echo ""
echo "============================================================"
echo "Step 3: Benchmark M9 on GPU (OpenCL)"
echo "============================================================"

echo "--- GPU Backend ---"
$LLM_BENCH -m "$CONFIG_PATH" -a opencl -t 4 -p 64 -n 256 -rep 10 2>&1 | tee "${RESULTS_DIR}/w8a8_mnn_gpu_p64.log"
$LLM_BENCH -m "$CONFIG_PATH" -a opencl -t 4 -p 256 -n 256 -rep 10 2>&1 | tee "${RESULTS_DIR}/w8a8_mnn_gpu_p256.log"

echo ""
echo "============================================================"
echo "DONE — Results in: ${RESULTS_DIR}/w8a8_mnn_*.log"
echo "============================================================"
echo ""
echo "Compare with NPU R1 (W8A8):"
echo "  Decode: 19.97 tok/s (p64), 19.07 tok/s (p256)"
echo "  TTFT:   49.4 ms (p64), 51.5 ms (p256)"
echo "  PPL:    16.93"
