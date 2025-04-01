cmake_minimum_required(VERSION 3.25)
project(Qui1framework VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87 89 90)
endif()

# 添加 fmt header-only 库
add_subdirectory(external/fmt)

# 添加可执行文件并链接 fmt
add_executable(test_fmt tests/test_fmt.cpp)
target_link_libraries(test_fmt PRIVATE fmt::fmt)


# 判断是否有 CUDA 环境
find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA 环境已找到")
else()
    message(STATUS "CUDA 环境未找到")
endif()
