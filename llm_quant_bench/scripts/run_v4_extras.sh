#!/usr/bin/env bash
# v4 paper 보강 측정 (모두 -rep 3 통일):
#   1) NPU baseline 5 ctx × 3 repeat (variance)
#   2) Long-gen NPU/CPU/GPU at ctx=3500 × n_gen={32,128,256,512} × rep 3
set -uo pipefail
cd /home/hyunho.son/Desktop/project/GPU-NPU-Benchmark
source ~/local/venv/bin/activate

OUT=llm_quant_bench/results/raw

echo "============================================================"
echo "[$(date '+%H:%M:%S')] (1/3) NPU baseline 5-ctx × 3-rep"
echo "============================================================"
python -u llm_quant_bench/benchmark/measure_npu_kvreuse.py 2>&1 | tee $OUT/npu_only_kvreuse_v4repeat.log

echo
echo "============================================================"
echo "[$(date '+%H:%M:%S')] (2/3) Long-gen NPU + CPU MNN (rep=3)"
echo "============================================================"
python -u llm_quant_bench/benchmark/measure_long_generation.py 2>&1 | tee $OUT/long_generation_v4.log

echo
echo "============================================================"
echo "[$(date '+%H:%M:%S')] (3/3) Long-gen GPU (rep=3)"
echo "============================================================"
python -u llm_quant_bench/benchmark/measure_long_generation_gpu.py 2>&1 | tee $OUT/long_generation_gpu_v4.log

echo
echo "[$(date '+%H:%M:%S')] ALL DONE"
ls -lh $OUT/npu_only_kvreuse_v3.json $OUT/long_generation_v4.json
