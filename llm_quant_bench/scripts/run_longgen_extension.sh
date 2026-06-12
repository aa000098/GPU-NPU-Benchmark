#!/usr/bin/env bash
# Long-gen 확장 (n_gen=1024, 2048), 3 backends, rep=3, 5-min cooldown.
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
}

cool_down 300

echo
echo "============================================================"
echo "[$(date '+%H:%M:%S')] (1/3) NPU long-gen ext (n_gen=1024,2048)"
echo "============================================================"
python -u llm_quant_bench/benchmark/measure_long_generation.py --backend npu \
    2>&1 | tee $OUT/longgen_ext_npu.log

cool_down 300

echo
echo "============================================================"
echo "[$(date '+%H:%M:%S')] (2/3) CPU MNN long-gen ext"
echo "============================================================"
python -u llm_quant_bench/benchmark/measure_long_generation.py --backend cpu \
    2>&1 | tee $OUT/longgen_ext_cpu.log

cool_down 300

echo
echo "============================================================"
echo "[$(date '+%H:%M:%S')] (3/3) GPU OpenCL long-gen ext"
echo "============================================================"
python -u llm_quant_bench/benchmark/measure_long_generation_gpu.py \
    2>&1 | tee $OUT/longgen_ext_gpu.log

echo
echo "[$(date '+%H:%M:%S')] DONE"
