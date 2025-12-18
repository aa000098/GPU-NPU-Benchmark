cmake_minimum_required(VERSION 3.10)
project(rknn_matmul CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# rknn api
set(RKNN_ROOT_PATH "$ENV{HOME}/install_files/rknn-toolkit2/rknpu2")
set(RKNN_API_PATH "${RKNN_ROOT_PATH}/runtime//${CMAKE_SYSTEM_NAME}/librknn_api")
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${CMAKE_ANDROID_ARCH_ABI}/librknnrt.so)
else()
  if(CMAKE_C_COMPILER MATCHES "aarch64")
    set(LIB_ARCH aarch64)
  else()
    #    set(LIB_ARCH armhf)
    set(LIB_ARCH aarch64)
  endif()

  set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
endif()

include_directories(${RKNN_API_PATH}/include)
include_directories(${RKNN_ROOT_PATH}/examples/3rdparty)

set(CMAKE_INSTALL_RPATH "lib")

# rknn_matmul_api_demo
add_library(matmul_rknn_f16 SHARED
  matmul_rknn_f16.cpp
  #  src/rknn_matmul_api_demo.cpp
  #  src/matmul_utils.cpp
)

target_link_libraries(matmul_rknn_f16
  ${RKNN_RT_LIB}
)

# install target and libraries
#set(CMAKE_INSTALL_PREFIX ${RKNN_ROOT_PATH}/install/rknn_matmul_api_demo_${CMAKE_SYSTEM_NAME})
#install(TARGETS rknn_matmul_api_demo DESTINATION ./)
#install(TARGETS rknn_matmul_api_dynshape_demo DESTINATION ./)
#install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
