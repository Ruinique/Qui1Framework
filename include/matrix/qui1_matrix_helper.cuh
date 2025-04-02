#ifndef QUI1_MATRIX_HELPER_CUH
#define QUI1_MATRIX_HELPER_CUH

#include <curand.h>
#include <fmt/core.h>
#include <fmt/ranges.h>  // For printing ranges like vectors

#include <iostream>
#include <stdexcept>
#include <type_traits>  // For std::is_same_v
#include <vector>       // For temporary host storage

#include "../common/error_check.cuh"
#include "qui1_device_matrix.cuh"
#include "qui1_host_matrix.cuh"
#include "qui1_matrix_base.cuh"

namespace qui1 {

class MatrixHelper {
   public:
    // 静态方法，禁止实例化
    MatrixHelper() = delete;

    template <typename T>
    static void printMatrix(const qui1::MatrixBase<T>& matrix,
                            const std::string& title = "Matrix") {
        if (!matrix.hasData()) {
            fmt::print("{}: Matrix is empty.\n", title);
            return;
        }

        const auto rows = matrix.getRows();
        const auto cols = matrix.getCols();
        const auto num_elements = rows * cols;
        fmt::print("{}:\n", title);
        fmt::print("  Location: {}\n",
                   matrix.getLocation() == Location::HOST ? "Host" : "Device");
        fmt::print("  Dimensions: {}x{}\n", rows, cols);
        fmt::print("  Layout: {}\n",
                   matrix.getLayout() == Layout::ROW_MAJOR ? "RowMajor"
                                                            : "ColumnMajor");
        fmt::print("  Data:\n");

        // Prepare data for printing (copy from device if necessary)
        std::vector<T> host_data(num_elements);
        const T* data_ptr = nullptr;

        if (matrix.getLocation() == Location::DEVICE) {
            CUDA_CHECK(cudaMemcpy(host_data.data(), matrix.getData(),
                                  num_elements * sizeof(T),
                                  cudaMemcpyDeviceToHost));
            data_ptr = host_data.data();
        } else {
            data_ptr = matrix.getData(); // Directly use host data
        }

        // Print the matrix data using fmt
        for (size_t i = 0; i < rows; ++i) {
            fmt::print("    [");
            for (size_t j = 0; j < cols; ++j) {
                // Calculate index based on layout
                size_t index = (matrix.getLayout() == Layout::ROW_MAJOR)
                                   ? (i * cols + j)
                                   : (j * rows + i);
                fmt::print("{}{}", data_ptr[index], (j == cols - 1) ? "" : ", ");
            }
            fmt::print("]\n");
        }
        fmt::print("\n");
    }


    template <typename T>
    static void fillWithRandom(qui1::MatrixBase<T>& matrix,
                               unsigned long long seed = 1234ULL) {
        if (!matrix.hasData()) {
            throw std::runtime_error(
                "Matrix must be allocated before filling with random numbers.");
        }

        const auto rows = matrix.getRows();
        const auto cols = matrix.getCols();
        const auto num_elements = rows * cols;
        T* target_data = matrix.getData();  // 获取目标数据的指针

        curandGenerator_t gen;
        CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));

        if (matrix.getLocation() == qui1::Location::DEVICE) {
            // 直接在设备内存上生成随机数
            generateRandomNumbers(gen, target_data, num_elements);
        } else if (matrix.getLocation() == qui1::Location::HOST) {
            // 1. 创建临时的 DeviceMatrix
            DeviceMatrix<T> temp_device_matrix(rows, cols, matrix.getLayout());
            T* device_data = temp_device_matrix.getData();

            // 2. 在临时设备矩阵上生成随机数
            generateRandomNumbers(gen, device_data, num_elements);

            // 3. 将数据从设备复制回主机
            CUDA_CHECK(cudaMemcpy(target_data, device_data, num_elements * sizeof(T),
                                  cudaMemcpyDeviceToHost));
            // temp_device_matrix 会在离开作用域时自动释放设备内存
        } else {
            CURAND_CHECK(curandDestroyGenerator(gen));  // 确保在抛出异常前销毁生成器
            throw std::runtime_error("Unsupported matrix location for random fill.");
        }

        CURAND_CHECK(curandDestroyGenerator(gen));
    }

   private:
    // 辅助函数，根据类型调用相应的 curandGenerate 函数
    template <typename T>
    static void generateRandomNumbers(curandGenerator_t gen, T* data,
                                      size_t num_elements) {
        if constexpr (std::is_same_v<T, float>) {
            CURAND_CHECK(curandGenerateUniform(gen, data, num_elements));
        } else if constexpr (std::is_same_v<T, double>) {
            CURAND_CHECK(curandGenerateUniformDouble(gen, data, num_elements));
        }
        // 可以为其他类型添加更多的 else if 分支，例如 curandGenerateNormal 等
        else {
            // 在编译时报错，如果类型不支持
            static_assert(
                std::is_same_v<T, float> || std::is_same_v<T, double>,
                "Unsupported data type for cuRAND random number generation.");
        }
    }
};
}  // namespace qui1
#endif  // QUI1_MATRIX_HELPER_CUH
