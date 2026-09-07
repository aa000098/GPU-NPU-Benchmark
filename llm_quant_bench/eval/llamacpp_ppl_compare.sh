#!/bin/bash
# llama.cpp perplexity comparison for fair backend comparison
# Run on same wikitext-2 test data (already in /tmp/wiki_eval/prompt.txt)
# 80K char from same source, but llama.cpp's tokenizer-specific output

set +e
LLAMA_PERP=/home/hyunho.son/install_files/llama.cpp/build/bin/llama-perplexity
LLAMA_DIR=/home/hyunho.son/install_files/llama.cpp/models/Llama-3.2-1B-Instruct
WIKI=/tmp/wiki_eval/prompt.txt
OUT=/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/paper/v9/v9_data/llamacpp_ppl.csv

echo "model,quant,ppl,raw" > $OUT

run() {
  local model=$1; local label=$2
  echo "[run] $label"
  local out=$(taskset -c 4-7 $LLAMA_PERP -m $model -f $WIKI -t 4 --ctx-size 1024 -fa on 2>&1 | tail -20)
  local final=$(echo "$out" | grep -oE "Final estimate: PPL = [0-9.]+ \+/- [0-9.]+" | head -1)
  echo "$label,\"$final\"" | tee -a $OUT
  sleep 30
}

run $LLAMA_DIR/llama32-1b-iq4nl.gguf "llama_iq4nl"
run $LLAMA_DIR/llama32-1b-q4km.gguf "llama_q4km"
run $LLAMA_DIR/llama32-1b-q80.gguf "llama_q80"

cat $OUT
