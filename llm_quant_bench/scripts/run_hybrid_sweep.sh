#!/bin/bash
# E2E hybrid measurement sweep: (k) × modes for a fixed prompt.
# Produces actual measured tok/s for sequential and async hybrid
# against NPU-only baseline. All measurements are real wall-clock.
#
# Usage:
#   tmux new -s hybrid_e2e 'bash llm_quant_bench/scripts/run_hybrid_sweep.sh'
set -euo pipefail

ROOT="/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark"
cd "$ROOT"

source ~/local/venv/bin/activate
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl-codex
export LD_LIBRARY_PATH=/home/hyunho.son/install_files/rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64:${LD_LIBRARY_PATH:-}

TARGET="llm_quant_bench/rkllm_models/Llama-3.2-1B-Instruct_W8A8_RK3588.rkllm"
DRAFT="llm_quant_bench/draft_models/Llama-3.2-1B-Instruct-Q4_0.gguf"
TOK="llm_quant_bench/models/Llama-3.2-1B-Instruct"

PROMPT="The key finding of our work is that edge NPU inference on"
N_GEN=32
OUT_DIR="llm_quant_bench/results/raw/hybrid_e2e"
mkdir -p "$OUT_DIR"

# Baseline: NPU-only (k and mode don't matter for this)
echo "=============================="
echo "Baseline: NPU-only"
echo "=============================="
python llm_quant_bench/benchmark/hybrid_runtime_v2.py \
    --target_model_path "$TARGET" \
    --draft_model_path "$DRAFT" \
    --tokenizer_path "$TOK" \
    --prompt "$PROMPT" \
    --n_gen "$N_GEN" \
    --k 1 \
    --mode npu_only \
    --output_json "$OUT_DIR/npu_only.json"

# Sweep k for each hybrid mode
for K in 1 2 4 8; do
    for MODE in sequential async; do
        echo ""
        echo "=============================="
        echo "mode=$MODE, k=$K"
        echo "=============================="
        python llm_quant_bench/benchmark/hybrid_runtime_v2.py \
            --target_model_path "$TARGET" \
            --draft_model_path "$DRAFT" \
            --tokenizer_path "$TOK" \
            --prompt "$PROMPT" \
            --n_gen "$N_GEN" \
            --k "$K" \
            --mode "$MODE" \
            --output_json "$OUT_DIR/${MODE}_k${K}.json"
    done
done

echo ""
echo "=== ALL DONE ($(date)) ==="
echo "Results in: $OUT_DIR"
