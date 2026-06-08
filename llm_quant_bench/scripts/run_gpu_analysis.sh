#!/bin/bash
# GPU vs CPU 성능 차이 원인 분석 실험 스크립트
# Usage: bash scripts/run_gpu_analysis.sh
set -e

cd /home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench

BENCH="/home/hyunho.son/install_files/MNN/build/llm_bench"
MODEL="./mnn_models/M2_q4_b64_direct/config.json"  # 대표 모델
OUTPUT="./results/raw/gpu_analysis"
mkdir -p "$OUTPUT"

echo "=============================================="
echo "실험 1: Prefill 길이별 GPU vs CPU 비교"
echo "  긴 시퀀스에서 GPU 병렬성이 발휘되는지 확인"
echo "=============================================="

echo "--- CPU ---"
for pl in 16 64 128 256 512 1024; do
    echo "  prompt=$pl"
    $BENCH -m "$MODEL" -a cpu -t 4 -p $pl -n 16 -rep 3 2>&1 | grep "pp$pl\|tg16" | tee -a "$OUTPUT/exp1_cpu.log"
done

echo ""
echo "--- GPU (OpenCL) ---"
for pl in 16 64 128 256 512 1024; do
    echo "  prompt=$pl"
    $BENCH -m "$MODEL" -a opencl -t 4 -p $pl -n 16 -rep 3 2>&1 | grep "pp$pl\|tg16" | tee -a "$OUTPUT/exp1_gpu.log"
done

echo ""
echo "=============================================="
echo "실험 2: Decode 길이별 GPU vs CPU 비교"
echo "  decode가 길어질수록 차이가 어떻게 변하는지"
echo "=============================================="

echo "--- CPU ---"
for tg in 16 64 128 256; do
    echo "  gen=$tg"
    $BENCH -m "$MODEL" -a cpu -t 4 -p 64 -n $tg -rep 3 2>&1 | grep "tg$tg" | tee -a "$OUTPUT/exp2_cpu.log"
done

echo ""
echo "--- GPU (OpenCL) ---"
for tg in 16 64 128 256; do
    echo "  gen=$tg"
    $BENCH -m "$MODEL" -a opencl -t 4 -p 64 -n $tg -rep 3 2>&1 | grep "tg$tg" | tee -a "$OUTPUT/exp2_gpu.log"
done

echo ""
echo "=============================================="
echo "실험 3: 8-bit vs 4-bit GPU 성능 비교"
echo "  양자화 비트에 따른 GPU 효율 차이"
echo "=============================================="

echo "--- 4-bit (M2) GPU ---"
$BENCH -m ./mnn_models/M2_q4_b64_direct/config.json -a opencl -t 4 -p 64,256 -n 256 -rep 3 2>&1 | grep -E "pp|tg" | tee "$OUTPUT/exp3_4bit_gpu.log"

echo "--- 8-bit (M4) GPU ---"
$BENCH -m ./mnn_models/M4_q8_ch_direct/config.json -a opencl -t 4 -p 64,256 -n 256 -rep 3 2>&1 | grep -E "pp|tg" | tee "$OUTPUT/exp3_8bit_gpu.log"

echo "--- 4-bit (M2) CPU ---"
$BENCH -m ./mnn_models/M2_q4_b64_direct/config.json -a cpu -t 4 -p 64,256 -n 256 -rep 3 2>&1 | grep -E "pp|tg" | tee "$OUTPUT/exp3_4bit_cpu.log"

echo "--- 8-bit (M4) CPU ---"
$BENCH -m ./mnn_models/M4_q8_ch_direct/config.json -a cpu -t 4 -p 64,256 -n 256 -rep 3 2>&1 | grep -E "pp|tg" | tee "$OUTPUT/exp3_8bit_cpu.log"

echo ""
echo "=============================================="
echo "실험 4: CPU 스레드 수에 따른 비교"
echo "  GPU는 스레드와 무관, CPU는 스레드 비례 → 교차점 확인"
echo "=============================================="

echo "--- CPU 1 thread ---"
$BENCH -m "$MODEL" -a cpu -t 1 -p 64 -n 64 -rep 3 2>&1 | grep -E "pp|tg" | tee "$OUTPUT/exp4_cpu_t1.log"

echo "--- CPU 2 threads ---"
$BENCH -m "$MODEL" -a cpu -t 2 -p 64 -n 64 -rep 3 2>&1 | grep -E "pp|tg" | tee "$OUTPUT/exp4_cpu_t2.log"

echo "--- CPU 4 threads ---"
$BENCH -m "$MODEL" -a cpu -t 4 -p 64 -n 64 -rep 3 2>&1 | grep -E "pp|tg" | tee "$OUTPUT/exp4_cpu_t4.log"

echo "--- CPU 8 threads ---"
$BENCH -m "$MODEL" -a cpu -t 8 -p 64 -n 64 -rep 3 2>&1 | grep -E "pp|tg" | tee "$OUTPUT/exp4_cpu_t8.log"

echo "--- GPU ---"
$BENCH -m "$MODEL" -a opencl -t 4 -p 64 -n 64 -rep 3 2>&1 | grep -E "pp|tg" | tee "$OUTPUT/exp4_gpu.log"

echo ""
echo "=== All GPU Analysis Done: $(date) ==="
