#!/bin/bash
# RKNN-LLM 벤치마크 + PPL 평가 스크립트 (tmux에서 실행)
# Usage: tmux new -s rkllm 'bash scripts/run_rkllm_bench_ppl.sh'
set -e

cd /home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench
MODELS_DIR="./rkllm_models"
RESULTS_DIR="./results/raw"
TOKENIZER="./models/Llama-3.2-1B-Instruct"

export LD_LIBRARY_PATH=/home/hyunho.son/install_files/rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64:$LD_LIBRARY_PATH

mkdir -p "$RESULTS_DIR"

echo "=== RKNN-LLM Benchmark Start: $(date) ==="
echo ""

# Step 1: Benchmark all .rkllm models
for model in "$MODELS_DIR"/*.rkllm; do
    name=$(basename "$model" .rkllm)
    size=$(du -m "$model" | cut -f1)

    # Skip obviously incomplete files (< 100MB)
    if [ "$size" -lt 100 ]; then
        echo "[SKIP] $name - only ${size}MB, likely incomplete"
        continue
    fi

    echo "============================================================"
    echo "Benchmarking: $name (${size}MB)"
    echo "Start: $(date)"
    echo "============================================================"

    python benchmark/bench_rkllm.py \
        --model_path "$model" \
        --output_dir "$RESULTS_DIR" \
        --prompt_lengths 64 256 \
        --gen_length 256 \
        --warmup 3 \
        --measure 5 2>&1 | tee "$RESULTS_DIR/bench_rkllm_${name}.log"

    echo "=== $name Benchmark DONE: $(date) ==="
    echo ""
done

# Step 2: PPL evaluation for all .rkllm models
echo ""
echo "=== RKNN-LLM PPL Evaluation Start: $(date) ==="

for model in "$MODELS_DIR"/*.rkllm; do
    name=$(basename "$model" .rkllm)
    size=$(du -m "$model" | cut -f1)

    if [ "$size" -lt 100 ]; then
        echo "[SKIP PPL] $name - too small"
        continue
    fi

    echo "============================================================"
    echo "PPL: $name"
    echo "Start: $(date)"
    echo "============================================================"

    python eval/ppl_rkllm.py \
        --model_path "$model" \
        --tokenizer_path "$TOKENIZER" \
        --output_dir "$RESULTS_DIR" \
        --stride 512 \
        --context_length 768 2>&1 | tee "$RESULTS_DIR/ppl_rkllm_${name}.log"

    echo "=== $name PPL DONE: $(date) ==="
    echo ""
done

echo ""
echo "=== ALL RKNN-LLM COMPLETE: $(date) ==="
