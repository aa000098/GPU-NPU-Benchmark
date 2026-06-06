#!/bin/bash
# RKNN-LLM PPL만 재실행 (벤치마크는 이미 완료)
# Usage: tmux new -s rkppl 'bash scripts/run_rkllm_ppl_only.sh'
set -e

cd /home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench
MODELS_DIR="./rkllm_models"
RESULTS="./results/raw"
TOKENIZER="./models/Llama-3.2-1B-Instruct"

export LD_LIBRARY_PATH=/home/hyunho.son/install_files/rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64:$LD_LIBRARY_PATH

echo "=== RKNN-LLM PPL Re-run Start: $(date) ==="

for model in "$MODELS_DIR"/*.rkllm; do
    name=$(basename "$model" .rkllm)
    size=$(du -m "$model" | cut -f1)

    if [ "$size" -lt 100 ]; then
        echo "[SKIP] $name - too small (${size}MB)"
        continue
    fi

    echo ""
    echo "=== PPL $name === $(date)"
    python eval/ppl_rkllm.py \
        --model_path "$model" \
        --tokenizer_path "$TOKENIZER" \
        --output_dir "$RESULTS" \
        --stride 512 \
        --context_length 768 2>&1 | tee "$RESULTS/ppl_rkllm_${name}.log"
    echo "=== PPL $name DONE === $(date)"
done

echo ""
echo "=== ALL PPL COMPLETE: $(date) ==="
