#!/usr/bin/env bash
# RegimeSpec v3 측정: 자연어 wiki long + (자연어 + 코드) short, 반복 클래스 제외.
# 순차 실행 (NPU 동시 점유 금지). 한 단계 실패 시 즉시 중단.
set -euo pipefail

cd /home/hyunho.son/Desktop/project/GPU-NPU-Benchmark
source ~/local/venv/bin/activate

TARGET=llm_quant_bench/rkllm_models/Llama-3.2-1B-Instruct_W8A8_RK3588.rkllm
DRAFT=llm_quant_bench/draft_models/Llama-3.2-1B-Instruct-Q4_0.gguf
TOKENIZER=llm_quant_bench/models/Llama-3.2-1B-Instruct
OUT=llm_quant_bench/results/raw

run_one() {
    local workload=$1
    local outjson=$OUT/regimespec_${workload}_v3.json
    local outlog=$OUT/regimespec_${workload}_v3.log
    echo
    echo "============================================================"
    echo "[$(date '+%H:%M:%S')] STARTING: workload=${workload}"
    echo "============================================================"
    python llm_quant_bench/spec_decoding/regimespec.py \
        --target_model "$TARGET" \
        --draft_model "$DRAFT" \
        --tokenizer "$TOKENIZER" \
        --workload "$workload" \
        --output_json "$outjson" \
        2>&1 | tee "$outlog"
    echo "[$(date '+%H:%M:%S')] DONE: $outjson"
}

run_one mixed
run_one long_only
run_one short_only

echo
echo "============================================================"
echo "[$(date '+%H:%M:%S')] ALL THREE WORKLOADS COMPLETE"
echo "============================================================"
ls -lh $OUT/regimespec_*_v3.json
