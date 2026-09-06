#!/bin/bash
# Systematic PPL evaluation for paper comparison
# Each config runs ppl_eval, dumps to CSV

set -e
PPL_BIN=/home/hyunho.son/install_files/MNN/build/ppl_eval
WIKI_DIR=/tmp/wiki_eval
OUT=/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/paper/v9/v9_data/ppl_systematic.csv

CFG_W8=/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/mnn_models/M9_w8a8_ch_direct/config_q0.json
CFG_W4=/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/mnn_models/custom_q4_b128_direct/config.json

echo "config,kv_mode,ppl,notes" > $OUT

run_ppl() {
  local cfg=$1; local label=$2; local env_pref=$3
  echo "[run] $label"
  local raw=$($env_pref $PPL_BIN $cfg $WIKI_DIR 2>&1 | grep "Perplexity" | tail -1)
  local ppl=$(echo "$raw" | grep -oE "[0-9]+\.[0-9]+")
  echo "$label,$ppl,$raw" | tee -a $OUT
  sleep 30  # cool-down
}

run_ppl $CFG_W8 "MNN_W8A8,fp16_KV" ""
run_ppl $CFG_W8 "MNN_W8A8,Stage2_HadamardKV" "MNN_QUANT_HADAMARD=1"
run_ppl $CFG_W4 "MNN_W4,fp16_KV" ""
run_ppl $CFG_W4 "MNN_W4,Stage2_HadamardKV" "MNN_QUANT_HADAMARD=1"

echo "=== Done ==="
cat $OUT
