#!/bin/bash
# M1-M8 GPU (OpenCL) 벤치마크 스크립트
# Usage: tmux new -s gpu_bench 'bash scripts/run_gpu_bench_all.sh'
set -e

cd /home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench

BENCH="/home/hyunho.son/install_files/MNN/build/llm_bench"
MODEL_BASE="./mnn_models"
OUTPUT="./results/raw"

mkdir -p "$OUTPUT"

echo "=== GPU (OpenCL) Benchmark Start: $(date) ==="

for dir in M1_q4_ch_direct M2_q4_b64_direct M3_q4_b128_direct M4_q8_ch_direct M5_q8_b64_direct M6_q4_b64_awq M7_q4_b64_smooth M8_q4_b64_hqq; do
    config="$MODEL_BASE/$dir/config.json"
    if [ ! -f "$config" ]; then
        echo "[SKIP] $dir - config not found"
        continue
    fi

    echo ""
    echo "=== $dir (OpenCL) === $(date)"
    $BENCH -m "$config" -a opencl -t 4 -p 64,256 -n 256 -rep 5 2>&1 | tee "$OUTPUT/gpu_bench_${dir}.log"
    echo "=== $dir DONE === $(date)"
done

echo ""
echo "=== All GPU Benchmark Done: $(date) ==="
