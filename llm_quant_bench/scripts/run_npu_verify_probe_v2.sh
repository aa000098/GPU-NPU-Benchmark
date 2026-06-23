#!/bin/bash
# Probe v2: context-aware RKLLM verify latency probe.
# Measures three modes per (ctx, k): A (fresh), B (combined), C (staged keep_history).
#
# Usage:
#   bash llm_quant_bench/scripts/run_npu_verify_probe_v2.sh
#   bash llm_quant_bench/scripts/run_npu_verify_probe_v2.sh 512 2048 4096   # contexts override
set -euo pipefail

ROOT="/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark"
cd "$ROOT"

source ~/local/venv/bin/activate
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl-codex
export LD_LIBRARY_PATH=/home/hyunho.son/install_files/rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64:${LD_LIBRARY_PATH:-}

MODEL_PATH="llm_quant_bench/rkllm_models/Llama-3.2-1B-Instruct_W8A8_RK3588.rkllm"
TOKENIZER_PATH="llm_quant_bench/models/Llama-3.2-1B-Instruct"
VERIFY_LENGTHS=(1 2 4 8 16)
WARMUP=2
MEASURE=5

if [ "$#" -eq 0 ]; then
    CONTEXTS=(32 256 1024 4096)
else
    CONTEXTS=("$@")
fi

OUT_JSON="llm_quant_bench/results/raw/npu_verify_v2_ctx${CONTEXTS[-1]}.json"

echo "============================================================"
echo "NPU verify probe v2 ($(date))"
echo "contexts=${CONTEXTS[*]}  k=${VERIFY_LENGTHS[*]}"
echo "============================================================"

python llm_quant_bench/benchmark/npu_verify_probe_v2.py \
    --model_path "$MODEL_PATH" \
    --tokenizer_path "$TOKENIZER_PATH" \
    --contexts "${CONTEXTS[@]}" \
    --verify_lengths "${VERIFY_LENGTHS[@]}" \
    --warmup "$WARMUP" \
    --measure "$MEASURE" \
    --output_json "$OUT_JSON"
