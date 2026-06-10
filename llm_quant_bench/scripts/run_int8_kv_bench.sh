#!/bin/bash
# MNN quant_qkv 옵션별 CPU decode 비교 (Phase 2 baseline)
# -qatten 0 → quant_qkv=8  (no KV quant, flash-attn)
# -qatten 1 → quant_qkv=9  (Q,K INT8)
# -qatten 2 → quant_qkv=10 (Q,K,V INT8)
# Usage: tmux new -s kv_bench 'bash scripts/run_int8_kv_bench.sh'
set -e

cd /home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench
BENCH="/home/hyunho.son/install_files/MNN/build/llm_bench"
MODEL="./mnn_models/M9_w8a8_ch_direct/config.json"
RESULTS="./results/raw"

mkdir -p "$RESULTS"

echo "=== quant_qkv Comparison on MNN CPU (M9 W8A8) === $(date)"
echo ""

for qatten in 0 1 2; do
  qkv=$((qatten + 8))
  echo "============================================================"
  echo "qatten=$qatten (quant_qkv=$qkv)"
  echo "============================================================"
  for p in 32 128 512 1024 2048 4096; do
    echo "--- prompt=$p, quant_qkv=$qkv ---"
    $BENCH -m "$MODEL" -a cpu -t 4 -p $p -n 64 -rep 3 -qatten $qatten 2>&1 | grep "M9_w8a8" | tee -a "$RESULTS/kv_bench_qkv${qkv}.log"
    echo ""
  done
  echo ""
done

echo "=== ALL DONE === $(date)"
