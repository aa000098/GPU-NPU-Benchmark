#!/bin/bash
# Run RKLLM NPU verify probe with the environment known to work on this machine.
#
# Example:
#   bash llm_quant_bench/scripts/run_npu_verify_probe.sh 4096
#   bash llm_quant_bench/scripts/run_npu_verify_probe.sh 512 1024 2048 4096
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
    CONTEXTS=(4096)
else
    CONTEXTS=("$@")
fi

for CTX in "${CONTEXTS[@]}"; do
    echo "============================================================"
    echo "NPU verify probe: context_len=$CTX ($(date))"
    echo "============================================================"

    python llm_quant_bench/benchmark/npu_verify_probe.py \
        --model_path "$MODEL_PATH" \
        --tokenizer_path "$TOKENIZER_PATH" \
        --context_len "$CTX" \
        --verify_lengths "${VERIFY_LENGTHS[@]}" \
        --warmup "$WARMUP" \
        --measure "$MEASURE" \
        --output_json "llm_quant_bench/results/raw/npu_verify_ctx${CTX}.json" \
        --oracle_json "llm_quant_bench/results/raw/npu_verify_ctx${CTX}_oracle.json"

    echo ""
done
