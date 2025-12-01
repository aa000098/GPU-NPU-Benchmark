#!/bin/bash
#set -e

###############################
# QNN HTP net-run 실행 스크립트
###############################

# 기본 경로들 (필요하면 수정해서 사용)
PROJECT="${HOME}/Desktop/project/GPU-NPU-Benchmark"
MODELS_DIR="${PROJECT}/models"
INPUT_DIR="$PROJECT/inputs"
OUTPUT_DIR="$PROJECT/outputs"
EXEC_DIR=$QNN_SDK_ROOT/bin/aarch64-oe-linux-gcc11.2
EXEC_FILE=$EXEC_DIR/qnn-net-run
BACKEND_DIR=$QNN_SDK_ROOT/lib

# 기본값: mha_qnn.cpp, input_list.txt, output_htp
#MODEL_CPP="${1:-${MODELS_DIR}/mha_qnn.cpp}"
MODEL_CPP="${1:-${MODELS_DIR}/mha_qnn.bin}"
INPUT_LIST="${2:-${INPUT_DIR}/input_list.txt}"
#BACKEND=$BACKEND_DIR/hexagon-v75/unsigned/libQnnHtpV75.so
#BACKEND=$BACKEND_DIR/aarch64-oe-linux-gcc11.2/libQnnGpu.so
BACKEND=$BACKEND_DIR/aarch64-oe-linux-gcc11.2/libQnnHtp.so

echo "[INFO] PROJECT      : ${PROJECT}"
echo "[INFO] MODEL_CPP    : ${MODEL_CPP}"
echo "[INFO] INPUT_LIST   : ${INPUT_LIST}"
echo "[INFO] OUTPUT_DIR   : ${OUTPUT_DIR}"
echo "[INFO] BACKEND      : ${BACKEND}"

# QNN SDK 환경 체크 (선택적)
if [[ -z "${QNN_SDK_ROOT}" ]]; then
    echo "[WARN] QNN_SDK_ROOT 가 설정되어 있지 않습니다."
    echo "       이미 envsetup.sh 를 source 해서 qnn-net-run 이 PATH 에 있다면 무시해도 됩니다."
else
    ENVSETUP="${QNN_SDK_ROOT}/bin/envsetup.sh"
    if [[ -f "${ENVSETUP}" ]]; then
        echo "[INFO] Sourcing ${ENVSETUP}"
        # 이미 한 번 source 되어 있으면 그냥 다시 불러도 무방
        source "${ENVSETUP}"
    else
        echo "[WARN] ${ENVSETUP} 파일이 없습니다. (QNN_SDK_ROOT=${QNN_SDK_ROOT})"
    fi
fi

# 파일 존재 여부 체크
if [[ ! -f "${MODEL_CPP}" ]]; then
    echo "[ERROR] 모델 CPP 파일을 찾을 수 없습니다: ${MODEL_CPP}"
    exit 1
fi

if [[ ! -f "${INPUT_LIST}" ]]; then
    echo "[ERROR] input_list 파일을 찾을 수 없습니다: ${INPUT_LIST}"
    echo "        형식 예시:"
    echo "          input data/input0.raw"
    exit 1
fi

# 출력 디렉토리 준비
if [[ ! -d "${OUTPUT_DIR}" ]]; then
    echo "[INFO] Creating output directory: ${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
fi

# qnn-net-run 실행
echo "[INFO] Running qnn-net-run..."
#qnn-net-run \
${EXEC_FILE} \
    --model        "${MODEL_CPP}" \
    --input_list   "${INPUT_LIST}" \
    --output_dir   "${OUTPUT_DIR}" \
    --backend      "${BACKEND}" \

RET=$?

if [[ ${RET} -ne 0 ]]; then
    echo "[ERROR] qnn-net-run failed with code ${RET}"
    exit ${RET}
fi

echo "[INFO] qnn-net-run completed successfully!"
echo "[INFO] Outputs are in: ${OUTPUT_DIR}"

