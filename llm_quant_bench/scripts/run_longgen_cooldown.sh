#!/usr/bin/env bash
# Long-generation 측정 (rep=3, 모든 backend) with 5-min cooldowns between.
# 의심된 thermal/state 영향을 배제하기 위해 backend 사이 충분한 휴식.
set -uo pipefail
cd /home/hyunho.son/Desktop/project/GPU-NPU-Benchmark
source ~/local/venv/bin/activate

OUT=llm_quant_bench/results/raw

cool_down() {
    local sec=$1
    echo
    echo "=== [$(date '+%H:%M:%S')] cooldown ${sec}s ==="
    for z in /sys/class/thermal/thermal_zone*/temp; do
        name=$(cat ${z%temp}type 2>/dev/null)
        temp=$(cat $z 2>/dev/null)
        [ -n "$temp" ] && printf "  %-25s %d°C\n" "$name" "$((temp/1000))"
    done
    sleep $sec
    echo "=== [$(date '+%H:%M:%S')] cooldown done ==="
    for z in /sys/class/thermal/thermal_zone*/temp; do
        name=$(cat ${z%temp}type 2>/dev/null)
        temp=$(cat $z 2>/dev/null)
        [ -n "$temp" ] && printf "  %-25s %d°C\n" "$name" "$((temp/1000))"
    done
}

# Initial cooldown (in case board is warm from previous runs)
cool_down 300

echo
echo "============================================================"
echo "[$(date '+%H:%M:%S')] (1/3) NPU long-gen (rep=3)"
echo "============================================================"
python -u llm_quant_bench/benchmark/measure_long_generation.py --backend npu \
    2>&1 | tee $OUT/longgen_v4_cool_npu.log

cool_down 300

echo
echo "============================================================"
echo "[$(date '+%H:%M:%S')] (2/3) CPU MNN long-gen (rep=3)"
echo "============================================================"
python -u llm_quant_bench/benchmark/measure_long_generation.py --backend cpu \
    2>&1 | tee $OUT/longgen_v4_cool_cpu.log

cool_down 300

echo
echo "============================================================"
echo "[$(date '+%H:%M:%S')] (3/3) GPU OpenCL long-gen (rep=3)"
echo "============================================================"
python -u llm_quant_bench/benchmark/measure_long_generation_gpu.py \
    2>&1 | tee $OUT/longgen_v4_cool_gpu.log

echo
echo "[$(date '+%H:%M:%S')] ALL DONE"
ls -lh $OUT/long_generation_v4.json
