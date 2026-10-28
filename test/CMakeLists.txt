cmake_minimum_required(VERSION 3.16)
project(qnn_hello LANGUAGES C CXX)

# QNN SDK 경로 (env에서 받아오기)
if(NOT DEFINED ENV{QNN_SDK_ROOT})
    message(FATAL_ERROR "QNN_SDK_ROOT is not set")
endif()

set(QNN_SDK_ROOT $ENV{QNN_SDK_ROOT})

# QNN 헤더 / 라이브러리 경로 — 실제 SDK 디렉토리 구조에 맞게 조정해야 함
include_directories(
    ${QNN_SDK_ROOT}/include
    ${QNN_SDK_ROOT}/include/QNN
)

# 예: 리눅스 aarch64용 QNN 런타임 라이브러리가 있는 경로
# 실제로는 ${QNN_SDK_ROOT}/lib/aarch64-linux-gnu  또는
# ${QNN_SDK_ROOT}/lib/aarch64-oe-linux-gcc11  이런 식일 가능성이 큼
link_directories(
    ${QNN_SDK_ROOT}/lib/aarch64-oe-linux-gcc11.2
)

add_executable(qnn_hello
    src/main.cpp
)

# 필요한 QNN 라이브러리들 — 네가 실제로 쓰는 백엔드에 맞게 추가
target_link_libraries(qnn_hello
    QnnSystem
    QnnHtp
    # QnnCpu QnnGpu 등 필요한 것들 추가
)

# (선택) 컴파일/링크 옵션 조정
target_compile_features(qnn_hello PRIVATE cxx_std_17)

