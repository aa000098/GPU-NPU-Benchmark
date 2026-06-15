#!/bin/bash
# Collect acceptance traces for speculative decoding analysis.
# Measures α_i = Σ min(p, q) at each generated position,
# where p = target (RKLLM) distribution, q = draft (llama.cpp) distribution.
#
# Usage:
#   tmux new -s accept 'bash llm_quant_bench/scripts/run_acceptance_trace.sh'
set -euo pipefail

ROOT="/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark"
cd "$ROOT"

source ~/local/venv/bin/activate
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl-codex
export LD_LIBRARY_PATH=/home/hyunho.son/install_files/rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64:${LD_LIBRARY_PATH:-}

TARGET="llm_quant_bench/rkllm_models/Llama-3.2-1B-Instruct_W8A8_RK3588.rkllm"
TOKENIZER="llm_quant_bench/models/Llama-3.2-1B-Instruct"
DRAFT_LLAMA="llm_quant_bench/draft_models/Llama-3.2-1B-Instruct-Q4_0.gguf"
DRAFT_QWEN="llm_quant_bench/draft_models/qwen2.5-0.5b-instruct-q4_0.gguf"

echo "============================================================"
echo "Acceptance trace: Llama Q4_0 draft → Llama target ($(date))"
echo "============================================================"
python llm_quant_bench/benchmark/collect_acceptance_traces.py \
    --target_model_path "$TARGET" \
    --draft_model_path "$DRAFT_LLAMA" \
    --tokenizer_path "$TOKENIZER" \
    --n_prompts 20 \
    --gen_tokens 64 \
    --draft_threads 4 \
    --output_json "llm_quant_bench/results/raw/acceptance_llama_draft.json"

echo ""
echo "============================================================"
echo "Acceptance trace: Qwen 0.5B draft → Llama target ($(date))"
echo "  (different tokenizer → underestimate; for ablation only)"
echo "============================================================"
python llm_quant_bench/benchmark/collect_acceptance_traces.py \
    --target_model_path "$TARGET" \
    --draft_model_path "$DRAFT_QWEN" \
    --tokenizer_path "$TOKENIZER" \
    --n_prompts 20 \
    --gen_tokens 64 \
    --draft_threads 4 \
    --output_json "llm_quant_bench/results/raw/acceptance_qwen_draft.json" || true

echo ""
echo "=== ALL DONE ($(date)) ==="
