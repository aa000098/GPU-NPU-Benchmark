#!/bin/bash
#set -e

#######################################
# QNN net-run input generator
# - 랜덤 float32 텐서를 raw로 저장
# - input_list.txt 생성
#######################################

# 프로젝트 루트 (이 스크립트 기준 상위 폴더)
PROJECT_DIR="$(cd "$(dirname "$0")/.."; pwd)"

# 입력 디렉토리 및 파일 경로
INPUT_DIR="${PROJECT_DIR}/inputs"
INPUT_RAW="${INPUT_DIR}/mha_input_0.raw"
INPUT_LIST="${INPUT_DIR}/input_list.txt"

# === 여기서 모델 입력 shape 맞춰줘야 함 ===
B=1
T=128
D=512
NUM_ELEMS=$((B * T * D))

echo "[INFO] Project dir : ${PROJECT_DIR}"
echo "[INFO] Input dir   : ${INPUT_DIR}"
echo "[INFO] Raw file    : ${INPUT_RAW}"
echo "[INFO] Input list  : ${INPUT_LIST}"
echo "[INFO] Shape       : [${B}, ${T}, ${D}]  (total ${NUM_ELEMS} elements, float32)"

# 입력 디렉토리 생성
mkdir -p "${INPUT_DIR}"

# Python으로 랜덤 float32 텐서 생성 -> raw 저장
python - << EOF
import numpy as np
import os

b, t, d = ${B}, ${T}, ${D}
out_path = r"${INPUT_RAW}"

x = np.random.randn(b, t, d).astype("float32")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
x.tofile(out_path)
print(f"[PY] Saved random input to {out_path}, shape=({b}, {t}, {d}), dtype=float32")
EOF

# input_list.txt 생성
# QNN 예제들처럼 "한 줄에 하나의 raw 파일 경로"만 쓰는 형태
# (모델에 입력이 하나뿐인 경우 이 형식이 일반적임)
echo "${INPUT_RAW}" > "${INPUT_LIST}"

echo "[INFO] Generated input_list.txt:"
cat "${INPUT_LIST}"

echo "[INFO] Done. You can now run qnn-net-run with:"
echo "  qnn-net-run \\"
echo "    --backend      htp \\"
echo "    --model        mha_qnn.cpp \\"
echo "    --input_list   ${INPUT_LIST} \\"
echo "    --output_dir   output_htp"

