#!/bin/bash
# NPU GEMM Benchmark — rknn_matmul_api
# FP16 matmul (type=1: FP16→FP32, type=4: FP16→FP16)
# core_mask=7: 3 NPU cores
set -e

DEMO="/home/hyunho.son/install_files/rknn-toolkit2/rknpu2/examples/rknn_matmul_api_demo/install/rknn_matmul_api_demo_Linux/rknn_matmul_api_demo"
export LD_LIBRARY_PATH="/home/hyunho.son/install_files/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64:$LD_LIBRARY_PATH"
OUTPUT="/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/results/raw/gpu_analysis"
mkdir -p "$OUTPUT"

LOOP=10

echo "=========================================================================="
echo "NPU GEMM Benchmark (RK3588, 3 NPU cores, rknn_matmul_api)"
echo "loop=$LOOP"
echo "=========================================================================="

echo ""
echo "=== FP16 matmul → FP32 (type=1), 3 NPU cores (core_mask=7) ==="
for S in 32 64 128 256 512 1024 2048; do
    echo "--- ${S}x${S}x${S} ---"
    $DEMO 1 ${S},${S},${S} 0 0 $LOOP 7 0 0 2>&1
    echo ""
done

echo ""
echo "=== INT8 matmul → INT32 (type=2), 3 NPU cores (core_mask=7) ==="
for S in 32 64 128 256 512 1024 2048; do
    echo "--- ${S}x${S}x${S} ---"
    $DEMO 2 ${S},${S},${S} 0 0 $LOOP 7 0 0 2>&1
    echo ""
done

echo ""
echo "=== Done: $(date) ==="
