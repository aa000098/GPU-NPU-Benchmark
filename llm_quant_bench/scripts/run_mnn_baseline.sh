#!/usr/bin/env bash
# MNN baseline: same Llama-3.2-1B W8A8 model, on CPU (Cortex-A76 4-core) and OpenCL (Mali-G610).
# Same ctx points + n_gen as our NPU measurement, KV cache enabled.
set -uo pipefail

LLM_BENCH=/home/hyunho.son/install_files/MNN/build/llm_bench
CONFIG=/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/mnn_models/M9_w8a8_ch_direct/config.json
OUT=/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/results/raw
mkdir -p "$OUT"

CTX_LIST="15 512 1024 2048 3500"
N_GEN=32
N_THREADS=4
N_REPEAT=3

run_one() {
    local backend=$1
    local ctx=$2
    local outfile=$OUT/mnn_${backend}_ctx${ctx}.txt
    echo
    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] MNN backend=${backend} ctx=${ctx} n_gen=${N_GEN}"
    echo "============================================================"
    "$LLM_BENCH" \
        -m "$CONFIG" \
        -a "$backend" \
        -t "$N_THREADS" \
        -p "${ctx}" \
        -n "${N_GEN}" \
        -kv true \
        -rep "$N_REPEAT" \
        -fp "$outfile" 2>&1 | tee -a "$OUT/mnn_${backend}_summary.log"
    echo "[$(date '+%H:%M:%S')] -> $outfile"
}

# Pin shell to A76 big cores
taskset -cp 4-7 $$ >/dev/null

echo "=== MNN CPU baseline ==="
for ctx in $CTX_LIST; do
    run_one cpu "$ctx"
done

echo
echo "=== MNN OpenCL (Mali-G610) baseline ==="
for ctx in $CTX_LIST; do
    run_one opencl "$ctx"
done

echo
echo "=== ALL DONE ==="
ls -lh $OUT/mnn_*.txt
