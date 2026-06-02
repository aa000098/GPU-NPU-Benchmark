#!/bin/bash
# M1-M5 PPL 평가 스크립트 (tmux에서 실행)
# Usage: tmux new -s ppl 'bash scripts/run_ppl_all.sh'
set -e

cd /home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench
MODEL_BASE="./mnn_models"
OUTPUT="./results/raw"
PPL_SCRIPT="/home/hyunho.son/install_files/MNN/transformers/llm/eval/evaluate_perplexity.py"

mkdir -p "$OUTPUT"

echo "=== PPL Evaluation Start: $(date) ==="

for dir in M1_q4_ch_direct M2_q4_b64_direct M3_q4_b128_direct M4_q8_ch_direct M5_q8_b64_direct; do
    config="$MODEL_BASE/$dir/config.json"
    if [ ! -f "$config" ]; then
        echo "[SKIP] $dir - config not found"
        continue
    fi

    echo ""
    echo "=== $dir === $(date)"
    python "$PPL_SCRIPT" -m "$config" -d wikitext/wikitext-2-raw-v1 2>&1 | tee "$OUTPUT/ppl_${dir}.log"
    echo "=== $dir DONE === $(date)"
done

echo ""
echo "=== All PPL Evaluation Done: $(date) ==="
