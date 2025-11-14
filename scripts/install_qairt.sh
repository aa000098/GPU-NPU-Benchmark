
#########################################################
# Qualcomm AI Engine Direct (QAIRT / QNN SDK) 2.40.0.251030
# - zip 없으면 다운로드
# - 있으면 재다운로드 스킵
# - qairt/<SDK_VERSION>/bin/envsetup.sh 자동 탐색
# - check-linux-dependency, envcheck까지 실행
#########################################################

VERSION="2.40.0.251030"
ZIP_NAME="${VERSION}.zip"
URL="https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/${VERSION}/${ZIP_NAME}"

INSTALL_BASE="$HOME/install_files"
INSTALL_DIR="$INSTALL_BASE/qairt_${VERSION}"
ROOT_DIR="$INSTALL_DIR/qairt/$VERSION"
ZIP_PATH="$INSTALL_DIR/${ZIP_NAME}"

echo "=== QAIRT / QNN SDK ${VERSION} 설치 스크립트 ==="

echo
echo ">>> [1] 설치 디렉토리 준비: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

#########################################################
# [2] zip 파일 체크 (이미 있으면 다운로드 스킵)
#########################################################
if [[ -f "$ZIP_PATH" ]]; then
    echo "📦 이미 다운로드된 SDK zip 발견: $ZIP_PATH"
    echo "    → 다운로드 스킵"
else
    echo ">>> [2] SDK 다운로드 중..."
    curl -L -o "$ZIP_PATH" "$URL"
fi

#########################################################
# [3] 압축 해제 (이미 qairt 디렉토리가 있으면 스킵)
#########################################################
if find "$INSTALL_DIR" -maxdepth 3 -type d -name "qairt" | grep -q "qairt"; then
    echo "📁 qairt 디렉토리가 이미 존재 → unzip 스킵"
else
    echo ">>> [3] SDK 압축 해제..."
    unzip -o "$ZIP_PATH" > /dev/null
    echo "    압축 해제 완료."
fi

#########################################################
# [4] qairt/<SDK_VERSION>/bin/envsetup.sh 실행
#########################################################
echo
echo ">>> [4] envsetup.sh 스크립트 실행"
cd $ROOT_DIR
source ./bin/envsetup.sh


#########################################################
# [5] check-linux-dependency.sh 실행
#########################################################
CHECK_DEP_SCRIPT="${ROOT_DIR}/bin/check-linux-dependency.sh"

if [[ -x "$CHECK_DEP_SCRIPT" ]]; then
    echo
    echo ">>> [5] Linux dependency 체크 스크립트 실행 (sudo 필요)"
    echo "    $CHECK_DEP_SCRIPT"

    # 1) 이미 root 계정이면 그냥 실행
    if [[ "$EUID" -eq 0 ]]; then
        "$CHECK_DEP_SCRIPT"

    else
        # 2) sudo가 있고, 비밀번호 없이 sudo 가능하면 직접 실행
        if command -v sudo >/dev/null 2>&1; then
            if sudo -n true 2>/dev/null; then
                echo "    (sudo 권한 확인 완료, sudo로 실행합니다)"
                sudo "$CHECK_DEP_SCRIPT"
            else
                # 3) sudo 비밀번호가 필요하거나, sudo 권한이 없는 경우
                echo "⚠️ 현재 사용자에게 비밀번호 없는 sudo 권한이 없거나,"
                echo "   sudo 권한이 없는 계정일 수 있습니다."
                echo
                echo "   아래 명령을 *sudo 권한이 있는 사용자*로 로그인해서 실행해 주세요:"
                echo
                echo "     sudo $CHECK_DEP_SCRIPT"
                echo
            fi
        else
            # 4) sudo 자체가 없는 환경 (예: 일부 minimal 컨테이너)
            echo "⚠️ 시스템에 sudo 명령이 없습니다."
            echo "   root 계정으로 로그인해서 아래 명령을 직접 실행해 주세요:"
            echo
            echo "     $CHECK_DEP_SCRIPT"
            echo
        fi
    fi
else
    echo "⚠️ check-linux-dependency.sh 를 찾지 못했거나 실행 권한이 없습니다:"
    echo "   $CHECK_DEP_SCRIPT"
fi


#########################################################
# [6] envcheck -c 실행 (toolchain 체크)
#########################################################
ENVCHECK_BIN="${ROOT_DIR}/bin/envcheck"

if [[ -x "$ENVCHECK_BIN" ]]; then
    echo
    echo ">>> [6] envcheck -c 실행 (toolchain 검증)"
    "$ENVCHECK_BIN" -c || true
else
    echo "⚠️ envcheck 바이너리를 찾지 못했습니다:"
    echo "   $ENVCHECK_BIN"
fi

#########################################################
# [7] Python 3.10 및 가상환경(qnn_venv) 자동 설정
#########################################################

echo
echo ">>> [7] Python 3.10 및 가상환경(qnn_venv) 설정"

# python3.10 존재 여부 확인
if ! command -v python3.10 >/dev/null 2>&1; then
    echo "❌ python3.10 을 찾을 수 없습니다."
    echo "   먼저 아래와 같이 설치해 주세요 (Ubuntu 기준):"
    echo "     sudo apt-get update"
    echo "     sudo apt-get install -y python3.10 python3.10-venv python3.10-distutils libpython3.10"
    echo
    echo "python3.10 설치 후 이 스크립트를 다시 실행해 주세요."
else
    PY_VER=$(python3.10 - << 'EOF'
import sys
print(".".join(map(str, sys.version_info[:3])))
EOF
)
    echo "    python3.10 버전: $PY_VER"
fi

# venv 디렉토리 경로
VENV_DIR="${ROOT_DIR}/qnn_venv"

if [[ -d "$VENV_DIR" ]]; then
    echo "📁 기존 가상환경 발견: $VENV_DIR"
else
    echo "    새 가상환경 생성: $VENV_DIR"
    python3.10 -m venv "$VENV_DIR" --without-pip
fi

# 가상환경 활성화 (이 스크립트 실행 동안에만 유효)
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo
echo "    활성화된 Python / pip:"
python --version || true
which python || true
which pip || true

#########################################################
# [8] pip 부트스트랩 및 QNN Python 의존성 설치
#########################################################

echo
echo ">>> [8] pip 부트스트랩 및 QNN Python 의존성 설치"

# Debian/Ubuntu 이슈 때문에 venv 생성 후 ensurepip 수행
python -m ensurepip --upgrade
pip install --upgrade pip

CHECK_PY_DEP="${ROOT_DIR}/bin/check-python-dependency"

if [[ -x "$CHECK_PY_DEP" ]]; then
    echo "    check-python-dependency 스크립트 실행:"
    echo "    python \"$CHECK_PY_DEP\""
    python "$CHECK_PY_DEP"
    # 옵션 패키지까지 설치하고 싶으면 아래 줄로 대체 가능:
    # python "$CHECK_PY_DEP" --with-optional
else
    echo "⚠️ $CHECK_PY_DEP 를 찾을 수 없습니다. 경로를 확인하세요."
fi

echo
echo "=== Python 가상환경 및 의존성 설정 완료 ==="
echo
echo "앞으로 이 SDK를 사용할 때는 새 터미널에서 아래 순서대로 실행하면 됩니다:"
echo "------------------------------------------------------"
echo "cd \"$BIN_DIR\""
echo "source ./envsetup.sh        # QAIRT_SDK_ROOT, QNN_SDK_ROOT 설정"
echo "cd \"$ROOT_DIR\""
echo "source qnn_venv/bin/activate  # Python 3.10 가상환경 활성화"
echo "------------------------------------------------------"
echo
echo "이 상태에서 QNN/QAIRT 툴(qnn-onnx-converter, qnn-run 등)을 사용하면 됩니다."
