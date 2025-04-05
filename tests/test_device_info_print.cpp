#include <gtest/gtest.h>
#include "common/device_info_print.cuh" // 包含我们要测试的头文件
#include "common/error_check.cuh" // 包含 CUDA_CHECK 宏定义

// 测试 print_device_info 函数
TEST(DeviceInfoPrintTest, PrintsWithoutError) {
    // 这个测试的主要目的是确保函数能够成功执行，
    // 调用 CUDA API 并使用 fmt 打印，而不会抛出任何 CUDA 错误或 C++ 异常。
    // 由于设备名称会根据实际硬件变化，我们不检查具体的输出字符串，
    // 只验证函数调用本身是否正常。
    EXPECT_NO_THROW({
        // 将调用放在代码块中，以确保任何潜在的异常都能被 EXPECT_NO_THROW 捕获
        qui1::common::print_device_info();
    });

    // 也可以添加一个检查，确保至少有一个 CUDA 设备可用
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    // 如果没有设备，打印函数内部的 cudaGetDevice 可能会失败，
    // 但 CUDA_CHECK 会处理它。这里我们额外检查一下。
    if (err == cudaErrorNoDevice) {
        GTEST_SKIP() << "Skipping test: No CUDA devices found.";
    }
}
