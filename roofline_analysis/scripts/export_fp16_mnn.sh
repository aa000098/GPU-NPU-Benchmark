#!/bin/bash
# FP16 네이티브 MNN 모델 Export + CPU/GPU 벤치마크
# 양자화 없는 FP16 모델로 CPU vs GPU 공정 비교
#
# Usage:
#   bash scripts/export_fp16_mnn.sh

set -e

LLMEXPORT_DIR="/home/hyunho.son/install_files/MNN/transformers/llm/export"
MNNCONVERT="/home/hyunho.son/install_files/MNN/build/MNNConvert"
LLM_BENCH="/home/hyunho.son/install_files/MNN/build/llm_bench"

MODEL_PATH="/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/models/Llama-3.2-1B-Instruct"
DST_PATH="/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/mnn_models/M10_fp16_direct"

echo "============================================================"
echo "Step 1: Export FP16 MNN Model (quant_bit=16)"
echo "============================================================"

if [ -f "${DST_PATH}/llm.mnn.weight" ]; then
    echo "[SKIP] M10_fp16_direct already exists"
else
    cd "$LLMEXPORT_DIR"
    python llmexport.py \
        --path "$MODEL_PATH" \
        --export mnn \
        --quant_bit 16 \
        --quant_block 0 \
        --dst_path "$DST_PATH" \
        --mnnconvert "$MNNCONVERT"
    echo "[OK] Exported M10_fp16_direct"
fi

# tmp 디렉토리 생성 (OpenCL cache)
mkdir -p "${DST_PATH}/tmp"

# config.json에 tmp_path 추가
python3 -c "
import json
config_path = '${DST_PATH}/config.json'
with open(config_path) as f:
    config = json.load(f)
config['tmp_path'] = '${DST_PATH}/tmp'
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)
print('Updated config.json with tmp_path')
"

CONFIG_PATH="${DST_PATH}/config.json"
if [ ! -f "$CONFIG_PATH" ]; then
    CONFIG_PATH="${DST_PATH}/llm_config.json"
fi

echo ""
echo "============================================================"
echo "Step 2: CPU Benchmark (FP16 model)"
echo "============================================================"
echo "--- CPU Normal ---"
$LLM_BENCH -m "$CONFIG_PATH" -a cpu -t 4 -c 0 -qatten 0 -p 64,256 -n 128 -rep 10

echo ""
echo "============================================================"
echo "Step 3: GPU Benchmark (FP16 model)"
echo "============================================================"
echo "--- GPU Normal ---"
$LLM_BENCH -m "$CONFIG_PATH" -a opencl -t 4 -c 0 -qatten 0 -p 64,256 -n 128 -rep 10

echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"
echo ""
echo "Reference (W8A8 INT8 model):"
echo "  CPU Low:  pp64=154, pp256=146, decode=15.9"
echo "  GPU Low:  pp64=149, pp256=175, decode=13.7"
echo "  NPU:      pp64=1296, pp256=4971, decode=20.0"
