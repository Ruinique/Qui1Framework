#ifndef QUI1_MATRIX_HELPER_CUH
#define QUI1_MATRIX_HELPER_CUH

#include <cuda_runtime.h> // 需要包含 cudaMemcpy2D
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
#include "matrix/view/qui1_matrix_view_base.cuh" // 确保包含视图基类

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
        const auto layout = matrix.getLayout();
        const auto location = matrix.getLocation();

        // 尝试获取视图信息
        const auto* view_ptr = dynamic_cast<const MatrixViewBase<T>*>(&matrix);
        const size_t lda = view_ptr ? view_ptr->getLDA() :
                           (layout == Layout::ROW_MAJOR ? cols : rows);
        // 注意：视图的 offset 是相对于原始数据指针的偏移，cudaMemcpy2D 处理的是指针本身
        // const size_t offset = view_ptr ? view_ptr->getOffset() : 0; // 这个 offset 在这里不直接用于 cudaMemcpy2D

        fmt::print("{}:\n", title);
        fmt::print("  Location: {}\n",
                   location == Location::HOST ? "Host" : "Device");
        fmt::print("  Dimensions: {}x{}\n", rows, cols);
        fmt::print("  Layout: {}\n",
                   layout == Layout::ROW_MAJOR ? "RowMajor" : "ColumnMajor");
        if (view_ptr) {
            fmt::print("  LDA: {}\n", lda);
            // fmt::print("  Offset: {}\n", offset); // Offset 信息对于理解视图有用，但复制时已包含在 getData() 返回的指针中
        }
        fmt::print("  Data:\n");

        std::vector<T> host_data; // 用于存储从设备复制过来的数据或直接引用主机数据
        const T* data_ptr = nullptr; // 指向要打印的数据
        bool host_data_is_contiguous_row_major = false; // 标记 host_data 中的布局是否为连续行主序

        if (location == Location::DEVICE) {
            host_data.resize(num_elements); // 分配足够容纳视图数据的连续主机内存
            data_ptr = host_data.data();
            const T* device_src_ptr = matrix.getData(); // 获取设备端数据指针（可能已包含 offset）

            // 使用 cudaMemcpy2D 处理设备到主机的复制
            size_t src_pitch = lda * sizeof(T); // 源（设备）内存的 pitch
            size_t dst_pitch = 0; // 目标（主机）内存的 pitch
            size_t copy_width_bytes = 0; // 每次复制的宽度（字节）
            size_t copy_height = 0; // 复制的高度（行数或列数）

            if (layout == Layout::ROW_MAJOR) {
                // 行主序：复制 rows 行，每行 cols 个元素
                copy_width_bytes = cols * sizeof(T);
                copy_height = rows;
                dst_pitch = cols * sizeof(T); // 主机端是连续存储的，pitch 等于宽度
                host_data_is_contiguous_row_major = true; // 复制后主机数据是连续行主序
            } else { // Layout::COLUMN_MAJOR
                // 列主序：复制 cols 列，每列 rows 个元素
                copy_width_bytes = rows * sizeof(T); // 复制宽度是行数 * 元素大小
                copy_height = cols; // 复制高度是列数
                dst_pitch = rows * sizeof(T); // 主机端按列复制过来后，每“列”（在主机内存中连续）的 pitch 是行数 * 元素大小
                // 注意：虽然复制过来了，但 host_data 是一维数组，打印时仍需按列主序逻辑访问
                host_data_is_contiguous_row_major = false; // 复制后主机数据不是标准的连续行主序排列
            }

            CUDA_CHECK(cudaMemcpy2D(host_data.data(), dst_pitch, // 目标：主机内存，目标 pitch
                                    device_src_ptr, src_pitch,   // 源：设备内存，源 pitch
                                    copy_width_bytes, copy_height, // 复制宽度（字节），复制高度
                                    cudaMemcpyDeviceToHost));

        } else { // Location::HOST
            data_ptr = matrix.getData(); // 直接使用主机数据指针
            // 主机端视图也可能有 lda，需要按 lda 访问，但不需要复制
            host_data_is_contiguous_row_major = (layout == Layout::ROW_MAJOR && lda == cols) ||
                                                (layout == Layout::COLUMN_MAJOR && lda == rows); // 只有完全连续时才为 true
        }

        // 打印数据
        for (size_t i = 0; i < rows; ++i) {
            fmt::print("    [");
            for (size_t j = 0; j < cols; ++j) {
                size_t index = 0;
                if (location == Location::DEVICE || host_data_is_contiguous_row_major) {
                    // 如果是从设备复制过来的（已变为连续行主序）或主机本身是连续行主序
                    index = i * cols + j; // 直接按行主序访问 host_data
                } else { // 主机端非连续（视图）
                    // 仍需根据原始布局和 LDA 计算索引
                    index = (layout == Layout::ROW_MAJOR)
                                ? (i * lda + j)
                                : (j * lda + i);
                }
                 // 对于列主序，从设备复制到 host_data 后，虽然 host_data 本身是连续内存，
                 // 但数据的逻辑排列仍然反映了列主序。我们需要按列主序的方式从 host_data 中提取元素。
                 // 如果是从设备复制过来的列主序数据
                if (location == Location::DEVICE && layout == Layout::COLUMN_MAJOR) {
                    // host_data 中，数据是按列存储的，每列有 rows 个元素。
                    // 第 j 列的起始位置是 j * rows。第 i 行的元素在该列的第 i 个位置。
                    index = j * rows + i;
                }


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

        // 注意：fillWithRandom 目前不支持直接填充非连续的视图
        // 如果 matrix 是一个视图，这里的实现可能需要调整，
        // 例如，创建一个临时的连续 DeviceMatrix，填充它，然后用 cudaMemcpy2D 复制回视图。
        // 或者，如果 cuRAND 支持带 pitch 的生成，可以直接在视图上操作。
        // 当前实现假设 matrix 是一个完整的（非视图）矩阵，或者视图恰好是连续的。
        const auto* view_ptr = dynamic_cast<const MatrixViewBase<T>*>(&matrix);
        if (view_ptr) {
             // 检查视图是否连续，如果不连续，当前 fillWithRandom 可能行为不正确
             const auto rows = matrix.getRows();
             const auto cols = matrix.getCols();
             const auto lda = view_ptr->getLDA();
             const bool is_contiguous = (matrix.getLayout() == Layout::ROW_MAJOR && lda == cols) ||
                                        (matrix.getLayout() == Layout::COLUMN_MAJOR && lda == rows);
             if (!is_contiguous) {
                 // 可以选择抛出异常或打印警告
                 fmt::print(stderr, "Warning: fillWithRandom called on a non-contiguous device view. "
                                    "The behavior might be incorrect as cuRAND typically operates on contiguous memory blocks.\n");
                 // 或者抛出异常:
                 // throw std::runtime_error("fillWithRandom on non-contiguous device views is not directly supported by this implementation.");
             }
             // 即使连续，getData() 返回的指针可能带有 offset，cuRAND 可能需要从 0 开始的指针。
             // 这是一个复杂的问题，取决于 cuRAND API 的具体行为。
             // 最安全的做法可能是创建一个临时的连续 DeviceMatrix。
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
