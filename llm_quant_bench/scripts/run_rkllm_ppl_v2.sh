#!/bin/bash
# RKNN-LLM PPL 평가 v2 (버그 수정 후, max_tokens 제한)
# Usage: tmux new -s rkppl 'bash scripts/run_rkllm_ppl_v2.sh'
set -e

cd /home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench
MODELS_DIR="./rkllm_models"
RESULTS="./results/raw"
TOKENIZER="./models/Llama-3.2-1B-Instruct"
MAX_TOKENS=30000

export LD_LIBRARY_PATH=/home/hyunho.son/install_files/rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64:$LD_LIBRARY_PATH

echo "=== RKNN-LLM PPL v2 Start: $(date) ==="
echo "Max tokens: $MAX_TOKENS"

for model in "$MODELS_DIR"/*.rkllm; do
    name=$(basename "$model" .rkllm)
    size=$(du -m "$model" | cut -f1)

    if [ "$size" -lt 100 ]; then
        echo "[SKIP] $name - too small (${size}MB)"
        continue
    fi

    echo ""
    echo "============================================================"
    echo "PPL: $name ($(date))"
    echo "============================================================"
    python eval/ppl_rkllm.py \
        --model_path "$model" \
        --tokenizer_path "$TOKENIZER" \
        --output_dir "$RESULTS" \
        --stride 512 \
        --context_length 768 \
        --max_tokens $MAX_TOKENS 2>&1 | tee "$RESULTS/ppl_rkllm_${name}.log"
    echo "=== $name DONE: $(date) ==="
done

echo ""
echo "=== ALL PPL COMPLETE: $(date) ==="
