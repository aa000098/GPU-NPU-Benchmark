#!/bin/bash
# PLD sweep: multiple prompt types × k × compared to NPU-only baseline.
# Goal: show PLD speedup varies by prompt repetitiveness but is always >= 1× on the NPU.
#
# Usage: tmux new -s pld 'bash llm_quant_bench/scripts/run_pld_sweep.sh'
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

OUT_DIR="llm_quant_bench/results/raw/pld_sweep"
mkdir -p "$OUT_DIR"

N_GEN=48

# Prompts: 5 categories × 1 prompt each
declare -A PROMPTS=(
    ["copy"]="Copy the following text exactly: 'The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.' The exact copy is:"
    ["code"]="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial("
    ["list"]="List 10 programming languages: 1. Python 2. JavaScript 3. Java 4. C++ 5. Ruby 6. Go 7."
    ["qa"]="Q: What is the capital of France? A: Paris. Q: What is the capital of Germany? A: Berlin. Q: What is the capital of Japan? A:"
    ["text"]="In this short paper, we present a novel approach to edge LLM inference. Our method achieves significant speedup by"
)

# Run npu_only baseline for each prompt (PLD comparison)
for ptype in copy code list qa text; do
    prompt="${PROMPTS[$ptype]}"
    echo ""
    echo "============================================================"
    echo "[$ptype] NPU-only baseline"
    echo "============================================================"
    python llm_quant_bench/benchmark/hybrid_runtime_v2.py \
        --target_model_path "$TARGET" \
        --draft_model_path "$DRAFT" \
        --tokenizer_path "$TOK" \
        --prompt "$prompt" \
        --n_gen "$N_GEN" \
        --k 1 \
        --mode npu_only \
        --output_json "$OUT_DIR/${ptype}_npu_only.json"

    # Sweep PLD k
    for K in 2 4 8; do
        echo ""
        echo "============================================================"
        echo "[$ptype] PLD k=$K"
        echo "============================================================"
        python llm_quant_bench/benchmark/hybrid_runtime_v2.py \
            --target_model_path "$TARGET" \
            --draft_model_path "$DRAFT" \
            --tokenizer_path "$TOK" \
            --prompt "$prompt" \
            --n_gen "$N_GEN" \
            --k "$K" \
            --mode pld \
            --output_json "$OUT_DIR/${ptype}_pld_k${K}.json"
    done
done

echo ""
echo "=== ALL DONE ($(date)) ==="
