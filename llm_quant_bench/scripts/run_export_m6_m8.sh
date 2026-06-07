#!/bin/bash
# M6-M8 (AWQ/Smooth/HQQ) 변환 스크립트 (tmux에서 실행)
# Usage: tmux new -s export 'bash scripts/run_export_m6_m8.sh'
set -e

cd /home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench
MODEL="/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/models/Llama-3.2-1B-Instruct"
OUTPUT="/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/mnn_models"

echo "=== M6-M8 Export Start: $(date) ==="

# M6 재시도 (AWQ) - 기존 실패 디렉토리 삭제
rm -rf "$OUTPUT/M6_q4_b64_awq"
echo ""
echo "=== M6 (AWQ) === $(date)"
python export/export_mnn.py --model_path "$MODEL" --config M6 --output_dir "$OUTPUT" 2>&1
echo "=== M6 DONE === $(date)"

echo ""
echo "=== M7 (Smooth Quant) === $(date)"
python export/export_mnn.py --model_path "$MODEL" --config M7 --output_dir "$OUTPUT" 2>&1
echo "=== M7 DONE === $(date)"

echo ""
echo "=== M8 (HQQ) === $(date)"
python export/export_mnn.py --model_path "$MODEL" --config M8 --output_dir "$OUTPUT" 2>&1
echo "=== M8 DONE === $(date)"

echo ""
echo "=== M6-M8 Export All Done: $(date) ==="
echo "=== Running benchmarks ==="

BENCH="/home/hyunho.son/install_files/MNN/build/llm_bench"
for dir in M6_q4_b64_awq M7_q4_b64_smooth M8_q4_b64_hqq; do
    config="$OUTPUT/$dir/config.json"
    if [ -f "$config" ]; then
        echo ""
        echo "=== Benchmark $dir === $(date)"
        $BENCH -m "$config" -a cpu -t 4 -p 64 -n 256 -rep 3 2>&1
        echo ""
        $BENCH -m "$config" -a cpu -t 4 -p 256 -n 256 -rep 3 2>&1
    else
        echo "[SKIP] $dir - no config.json"
    fi
done

echo ""
echo "=== Running PPL evaluation ==="
PPL_SCRIPT="/home/hyunho.son/install_files/MNN/transformers/llm/eval/evaluate_perplexity.py"
RESULTS="/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/results/raw"

for dir in M6_q4_b64_awq M7_q4_b64_smooth M8_q4_b64_hqq; do
    config="$OUTPUT/$dir/config.json"
    if [ -f "$config" ]; then
        echo ""
        echo "=== PPL $dir === $(date)"
        python "$PPL_SCRIPT" -m "$config" -d wikitext/wikitext-2-raw-v1 2>&1 | tee "$RESULTS/ppl_${dir}.log"
        echo "=== PPL $dir DONE === $(date)"
    fi
done

echo ""
echo "=== ALL COMPLETE: $(date) ==="
