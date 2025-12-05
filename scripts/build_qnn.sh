#!/bin/bash
#set -e

# ================================
# QNN NPU(HTP) MODEL BUILD SCRIPT
# ================================

PROJECT=$HOME/Desktop/project/GPU-NPU-Benchmark
MODELS=$PROJECT/models
OUTPUT=$PROJECT/output

QNN_ROOT="$QNN_SDK_ROOT"
TOOL_NAME="qnn-model-lib-generator"
TOOL_PATH="$QNN_ROOT/bin/$TOOL_NAME"

MODEL_FILE="$MODELS/mha_qnn.cpp"
BIN_FILE="$MODELS/mha_qnn.bin"
OUT_DIR="$OUTPUT/mha_htp_lib"
BACKEND="htp"
#TARGET="x86_64-linux-clang"
TARGET="aarch64-oe-linux-gcc11.2"

# QNN SDK 환경 확인
if [[ -z "$QNN_SDK_ROOT" ]]; then
    echo "[ERROR] QNN_SDK_ROOT not set. Run:"
    echo "   source <QNN_PATH>/bin/envsetup.sh"
    exit 1
fi

echo "[INFO] Using QNN SDK at: $QNN_SDK_ROOT"

# 출력 디렉터리 준비
if [[ ! -d "$OUT_DIR" ]]; then
    echo "[INFO] Creating output directory: $OUT_DIR"
    mkdir -p "$OUT_DIR"
fi

# 모델 파일 확인
if [[ ! -f "$MODEL_FILE" ]]; then
    echo "[ERROR] Model file '$MODEL_FILE' not found!"
    exit 1
fi

if [[ ! -f "$BIN_FILE" ]]; then
    echo "[ERROR] Binary file '$BIN_FILE' not found!"
    exit 1
fi

# 모델 라이브러리 생성
echo "[INFO] Building QNN HTP model..."
python $TOOL_PATH \
    -c "$MODEL_FILE" \
    -b "$BIN_FILE" \
    -o "$OUT_DIR" \
    -t "$TARGET" \
#    -l "$LIB_NAME" \

echo "[INFO] QNN HTP model build complete!"
echo "[INFO] Output directory: $OUT_DIR"

