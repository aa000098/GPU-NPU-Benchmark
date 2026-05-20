#!/bin/bash
# MNN LLM + OpenCL 빌드 스크립트 (RK3588 네이티브)
set -e

MNN_DIR="/home/hyunho.son/install_files/MNN"
BUILD_DIR="${MNN_DIR}/build"

echo "=== Building MNN with LLM + OpenCL support ==="

cd "${BUILD_DIR}"

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNN_OPENCL=ON \
  -DMNN_BUILD_LLM=ON \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_SEP_BUILD=ON \
  -DMNN_LLM_BUILD_DEMO=ON

make -j$(nproc)

echo ""
echo "=== Build complete ==="
echo "Binaries:"
echo "  llm_demo:  $(ls -lh ${BUILD_DIR}/transformer/llm_demo 2>/dev/null || echo 'NOT FOUND')"
echo "  llm_bench: $(ls -lh ${BUILD_DIR}/transformer/tools/llm_bench 2>/dev/null || echo 'NOT FOUND')"
echo "  ppl_eval:  $(ls -lh ${BUILD_DIR}/transformer/tools/ppl_eval 2>/dev/null || echo 'NOT FOUND')"
echo "  MNNConvert: $(ls -lh ${BUILD_DIR}/MNNConvert 2>/dev/null || echo 'NOT FOUND')"
