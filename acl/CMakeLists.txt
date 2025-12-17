cmake_minimum_required(VERSION 3.10)
project(acl_matmul CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ACL_ROOT "$ENV{HOME}/install_files/ComputeLibrary")

# OpenCL 찾기(가능하면 find_package가 더 안전)
find_package(OpenCL REQUIRED)

add_library(acl_common INTERFACE)
target_include_directories(acl_common INTERFACE
  ${ACL_ROOT}
  ${ACL_ROOT}/include
)

# 필요하면 경로 추가 (네 환경에 맞게)
target_link_directories(acl_common INTERFACE
  ${ACL_ROOT}/build
  ${ACL_ROOT}/lib
)

target_link_libraries(acl_common INTERFACE
  arm_compute
  #  arm_compute_core
  OpenCL::OpenCL
)

# --- shared libs ---
add_library(matmul_acl_f16 SHARED matmul_acl_f16.cpp)
add_library(matmul_acl_int8 SHARED matmul_acl_int8.cpp)
add_library(softmax_acl_f16 SHARED softmax_acl_f16.cpp)

target_link_libraries(matmul_acl_f16 PRIVATE acl_common)
target_link_libraries(matmul_acl_int8 PRIVATE acl_common)
target_link_libraries(softmax_acl_f16 PRIVATE acl_common)
