#!/usr/bin/env bash
# Async measurement for remaining workloads (short_only first, then mixed).
# Merges into existing v3 jsons. Sequential — NPU 동시 점유 금지.
set -uo pipefail   # set -e 제거: 한 워크로드 실패해도 다음 진행

cd /home/hyunho.son/Desktop/project/GPU-NPU-Benchmark
source ~/local/venv/bin/activate

TARGET=llm_quant_bench/rkllm_models/Llama-3.2-1B-Instruct_W8A8_RK3588.rkllm
DRAFT=llm_quant_bench/draft_models/Llama-3.2-1B-Instruct-Q4_0.gguf
TOKENIZER=llm_quant_bench/models/Llama-3.2-1B-Instruct
OUT=llm_quant_bench/results/raw

run_async_for() {
    local workload=$1
    local existing_json=$OUT/regimespec_${workload}_v3.json
    local outlog=$OUT/async_${workload}_v3.log
    echo
    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] STARTING async measurement: workload=${workload}"
    echo "  merging into: $existing_json"
    echo "============================================================"
    python -u llm_quant_bench/spec_decoding/measure_async_only.py \
        --target_model "$TARGET" \
        --draft_model "$DRAFT" \
        --tokenizer "$TOKENIZER" \
        --workload "$workload" \
        --existing_json "$existing_json" \
        2>&1 | tee "$outlog"
    echo "[$(date '+%H:%M:%S')] DONE: $workload"
}

# 짧은 것 먼저 → 빠르게 short async 결과 보임
run_async_for short_only
run_async_for mixed

echo
echo "============================================================"
echo "[$(date '+%H:%M:%S')] ALL ASYNC MEASUREMENTS COMPLETE"
echo "============================================================"
ls -lh $OUT/regimespec_*_v3.json
