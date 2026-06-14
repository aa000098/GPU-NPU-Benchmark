#!/bin/bash
# p4096 CPU → GPU → NPU 순차 벤치마크
# Usage: tmux new -s p4096 'bash scripts/run_p4096_bench.sh'
set -e

cd /home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench
BENCH="/home/hyunho.son/install_files/MNN/build/llm_bench"
MNN_MODEL="./mnn_models/M9_w8a8_ch_direct/config.json"
RKLLM_MODEL="./rkllm_models/Llama-3.2-1B-Instruct_W8A8_RK3588.rkllm"

export LD_LIBRARY_PATH=/home/hyunho.son/install_files/rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64:$LD_LIBRARY_PATH

echo "=== p4096 Benchmark Start: $(date) ==="

echo ""
echo "=== [1/3] CPU p4096 === $(date)"
$BENCH -m "$MNN_MODEL" -a cpu -t 4 -p 4096 -n 64 -rep 3 2>&1 | grep "M9_w8a8"
echo "=== CPU DONE: $(date) ==="

echo ""
echo "=== [2/3] GPU p4096 === $(date)"
$BENCH -m "$MNN_MODEL" -a opencl -t 4 -p 4096 -n 64 -rep 3 2>&1 | grep "M9_w8a8"
echo "=== GPU DONE: $(date) ==="

echo ""
echo "=== [3/3] NPU p4096 === $(date)"
python benchmark/bench_rkllm.py \
  --model_path "$RKLLM_MODEL" \
  --output_dir ./results/raw \
  --prompt_lengths 4096 \
  --gen_length 64 \
  --warmup 2 \
  --measure 3 2>&1 | grep -E "TTFT=|Prompt|tok/s"
echo "=== NPU DONE: $(date) ==="

echo ""
echo "=== ALL DONE: $(date) ==="
