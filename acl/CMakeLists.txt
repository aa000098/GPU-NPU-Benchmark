cmake_minimum_required(VERSION 3.10)
project(acl_matmul CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ACL_ROOT "$ENV{HOME}/install_files/ComputeLibrary")

include_directories(
    ${ACL_ROOT}
    ${ACL_ROOT}/include
)

link_directories(
    ${ACL_ROOT}/build
)

add_library(matmul_acl_f16 SHARED matmul_acl_f16.cpp)

target_link_libraries(matmul_acl_f16
    arm_compute
    #    arm_compute_core
    OpenCL
)

