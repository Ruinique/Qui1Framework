#pragma once

#include <cuda_runtime.h>
#include <fmt/core.h>

#include "common/error_check.cuh"  // 包含错误检查宏

namespace qui1 {
namespace common {

/**
 * @brief 打印当前 CUDA 设备的名称。
 *
 * 该函数获取当前活动的 CUDA 设备的属性，并使用 fmt 库打印其名称。
 * 如果获取设备属性或设备名称时发生 CUDA 错误，将调用 CUDA_CHECK 宏进行错误处理。
 */
inline void print_device_info() {
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));  // 获取当前设备 ID
    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));  // 获取设备属性
    fmt::print("Using device: {}\n", device_prop.name);  // 使用 fmt 打印设备名称
}

}  // namespace common
}  // namespace qui1
