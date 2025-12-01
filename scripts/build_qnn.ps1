param(
    # 프로젝트 루트 (기본: 스크립트 기준 상위 폴더)
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,

    # 모델 베이스 이름
    [string]$ModelBaseName = "mha_qnn",

    # 타겟 (우선 x86_64-windows-msvc로 테스트하고,
    # 나중에 aarch64-android로 바꿔도 됨. NDK 필요)
    # [string]$Target = "x86_64-windows-msvc"
    # [string]$Target = "aarch64-oe-linux"
    [string]$Target = "windows-x86_64"
)

Write-Host "[INFO] Project root : $ProjectRoot"

$modelsDir = Join-Path $ProjectRoot "models"
$outDir    = Join-Path $ProjectRoot ("output\mha_qnn_" + $Target)

# cpp 파일 (mha_qnn.cpp 우선, 없으면 mha_qnn)
$cppFile = Join-Path $modelsDir ($ModelBaseName + ".cpp")
if (-not (Test-Path $cppFile)) {
    $fallback = Join-Path $modelsDir $ModelBaseName
    if (Test-Path $fallback) {
        $cppFile = $fallback
    }
    else {
        Write-Error "[ERROR] CPP file not found: $cppFile or $fallback"
        exit 1
    }
}

# bin 파일
$binFile = Join-Path $modelsDir ($ModelBaseName + ".bin")
if (-not (Test-Path $binFile)) {
    Write-Error "[ERROR] BIN file not found: $binFile"
    exit 1
}

# QNN SDK 환경변수 체크
if (-not $Env:QNN_SDK_ROOT) {
    Write-Error "[ERROR] QNN_SDK_ROOT is not set. 먼저 다음처럼 설정하세요:"
    Write-Error "        `$Env:QNN_SDK_ROOT = 'C:\Qualcomm\AIStack\QAIRT\2.22.6.240515'"
    Write-Error "        & `"$Env:QNN_SDK_ROOT\bin\envsetup.bat`""
    exit 1
}

# qnn-model-lib-generator 실제 위치 (Windows용)
$toolPath = Join-Path $Env:QNN_SDK_ROOT "bin\x86_64-windows-msvc\qnn-model-lib-generator"
if (-not (Test-Path $toolPath)) {
    # 혹시 exe 래퍼가 있을 수도 있으니 한 번 더 체크
    $toolPathExe = "$toolPath.exe"
    if (Test-Path $toolPathExe) {
        $toolPath = $toolPathExe
    }
    else {
        Write-Error "[ERROR] qnn-model-lib-generator not found at:"
        Write-Error "        $toolPath"
        exit 1
    }
}

Write-Host "[INFO] Using QNN SDK at : $Env:QNN_SDK_ROOT"
Write-Host "[INFO] Models dir       : $modelsDir"
Write-Host "[INFO] CPP file         : $cppFile"
Write-Host "[INFO] BIN file         : $binFile"
Write-Host "[INFO] Target           : $Target"
Write-Host "[INFO] Output dir       : $outDir"
Write-Host "[INFO] Tool path        : $toolPath"

# 출력 폴더 준비
if (-not (Test-Path $outDir)) {
    Write-Host "[INFO] Creating output directory: $outDir"
    New-Item -ItemType Directory -Path $outDir | Out-Null
}

# qnn-model-lib-generator 실행 (★ python으로 강제 실행)
Write-Host "[INFO] Running qnn-model-lib-generator via python..."

# 현재 qnn 가상환경의 python 사용 (이미 qnn env에서 실행한다고 가정)
& python `
    $toolPath `
    -c $cppFile `
    -b $binFile `
    -l $ModelBaseName `
    -o $outDir `
    -t $Target `

if ($LASTEXITCODE -ne 0) {
    Write-Error "[ERROR] qnn-model-lib-generator failed with code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "[INFO] QNN model lib build complete!"
Write-Host "[INFO] Output files in: $outDir"
Get-ChildItem $outDir
