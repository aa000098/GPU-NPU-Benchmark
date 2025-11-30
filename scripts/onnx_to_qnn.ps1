param(
    [string]$OnnxPath = "..\pytorch\mha.onnx",
    # 출력 베이스 파일 경로 (확장자 X)
    [string]$OutBase  = "..\models\mha_qnn",
    # [string]$Backend  = "gpu"
    [string]$Backend  = "npu"
)

if (-not $env:QNN_SDK_ROOT) {
    Write-Error "환경변수 QNN_SDK_ROOT 가 설정되어 있지 않습니다."
    exit 1
}

$env:PYTHONPATH = "$env:QNN_SDK_ROOT\lib\python;$env:PYTHONPATH"

$converter = Join-Path $env:QNN_SDK_ROOT "bin\x86_64-windows-msvc\qnn-onnx-converter"
if (-not (Test-Path $converter)) {
    Write-Error "qnn-onnx-converter 스크립트를 찾을 수 없습니다: $converter"
    exit 1
}

# 부모 디렉터리(.\\models)만 만들어준다
$OutDir = Split-Path $OutBase -Parent
if ($OutDir -and -not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir | Out-Null
}

# 입력 shape 고정
$InputName = "input"
$Batch  = 32
$SeqLen = 128

& python "$converter" `
    --input_network "$OnnxPath" `
    --output_path   "$OutBase" `
    --input_dim     "$InputName" "$Batch,$SeqLen,512"
    --backend      "$Backend"
