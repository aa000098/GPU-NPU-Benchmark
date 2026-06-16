#!/bin/bash
# Bench CPU draft model candidates with llama.cpp.
# Measures prefill pp{32,256,1024,4096} and decode tg16 for each GGUF.
#
# Usage: tmux new -s draft 'bash llm_quant_bench/scripts/run_draft_bench.sh'
set -euo pipefail

ROOT="/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark"
cd "$ROOT"

source ~/local/venv/bin/activate
export PYTHONPATH=.

DRAFTS=(
    "llm_quant_bench/draft_models/Llama-3.2-1B-Instruct-Q4_0.gguf"
    "llm_quant_bench/draft_models/Llama-3.2-1B-Instruct-Q4_0_4_8.gguf"
    "llm_quant_bench/draft_models/Llama-3.2-1B-Instruct-IQ3_M.gguf"
    "llm_quant_bench/draft_models/qwen2.5-0.5b-instruct-q4_0.gguf"
)

echo "============================================================"
echo "Draft CPU bench ($(date))"
echo "============================================================"

python llm_quant_bench/benchmark/draft_bench.py \
    --models "${DRAFTS[@]}" \
    --threads 4 \
    --contexts 32 256 1024 4096 \
    --gen_lens 16 \
    --repeats 3 \
    --T_verify_k8_ms 418.0 \
    --output_json llm_quant_bench/results/raw/draft_bench.json

echo ""
echo "=== DONE ($(date)) ==="
