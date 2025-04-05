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
#include "qui1_device_matrix.cuh" // 需要 DeviceMatrix 作为返回类型和临时对象

namespace qui1 {

// 定义提取类型
enum class TriangleType { UPPER, LOWER };

#ifdef __CUDACC__
// CUDA 核函数，用于提取上/下三角部分
template <typename T>
__global__ void extractTriangleKernel(const T* input_data, T* output_data,
                                      size_t rows, size_t cols, size_t input_lda, size_t output_lda,
                                      Layout layout, TriangleType triangle_type) {
    // 计算全局线程索引 (i, j)
    const auto j = blockIdx.x * blockDim.x + threadIdx.x; // Column index (使用 auto)
    const auto i = blockIdx.y * blockDim.y + threadIdx.y; // Row index (使用 auto)

    // 检查索引是否越界
    if (i < rows && j < cols) {
        // 根据布局计算输入和输出内存中的线性索引
        const auto input_idx = (layout == Layout::ROW_MAJOR) ? (i * input_lda + j) : (j * input_lda + i); // 使用 auto 和 const
        const auto output_idx = (layout == Layout::ROW_MAJOR) ? (i * output_lda + j) : (j * output_lda + i); // 使用 auto 和 const

        // 判断当前元素是否属于目标三角区域
        const bool condition = (triangle_type == TriangleType::UPPER) // 使用 const
                             ? (j >= i) // 上三角条件 (列索引 >= 行索引)
                             : (j <= i); // 下三角条件 (列索引 <= 行索引)

        // 根据条件复制元素或置零
        if (condition) {
            output_data[output_idx] = input_data[input_idx];
        } else {
            output_data[output_idx] = static_cast<T>(0); // 非目标区域置零
        }
    }
}
#endif // __CUDACC__


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


    /**
     * @brief 从输入矩阵视图中提取上三角或下三角部分。
     *
     * @tparam T 矩阵元素类型。
     * @param input_matrix 输入矩阵视图（可以是 HostMatrixView 或 DeviceMatrixView）。
     * @param triangle_type 要提取的三角类型 (UPPER 或 LOWER)。
     * @return 一个新的 DeviceMatrix，包含提取出的三角部分，其余元素为零。
     *         如果输入是 HostMatrixView，数据会被复制到设备上进行处理。
     */
    template <typename T>
    static DeviceMatrix<T> extractTriangle(
        const MatrixViewBase<T>& input_matrix, // 接受基类引用，兼容 Host/Device 视图
        TriangleType triangle_type)
    {
        const auto rows = input_matrix.getRows();
        const auto cols = input_matrix.getCols();
        const auto layout = input_matrix.getLayout();
        const auto location = input_matrix.getLocation();
        const auto input_lda = input_matrix.getLDA(); // 获取输入视图的 LDA

        // 处理空矩阵情况
        if (rows == 0 || cols == 0) {
            return DeviceMatrix<T>(0, 0, layout); // 返回空的 DeviceMatrix
        }

        // 准备输入数据指针 (指向设备内存)
        const T* device_input_ptr = nullptr;
        size_t kernel_input_lda = 0; // 核函数实际读取数据的 LDA
        DeviceMatrix<T> temp_device_input; // RAII 管理临时设备内存

        if (location == Location::HOST) {
            // 1. 输入在主机：创建临时设备矩阵并将数据从主机视图复制过去
            temp_device_input = DeviceMatrix<T>(rows, cols, layout); // 分配设备内存
            const T* host_src_ptr = input_matrix.getData(); // 主机视图的数据指针
            const size_t temp_device_lda = temp_device_input.getLeadingDimension(); // 临时设备矩阵的 LDA

            // 2. 使用 cudaMemcpy2D 处理可能非连续的主机视图到连续设备内存的复制
            const size_t src_pitch = input_lda * sizeof(T); // 源 (主机) pitch
            const size_t dst_pitch = temp_device_lda * sizeof(T); // 目标 (设备) pitch
            size_t copy_width_bytes = 0;
            size_t copy_height = 0;

            // 根据布局确定复制的宽度和高度
            if (layout == Layout::ROW_MAJOR) {
                copy_width_bytes = cols * sizeof(T);
                copy_height = rows;
            } else { // COLUMN_MAJOR
                copy_width_bytes = rows * sizeof(T); // 列主序时，宽度是行数
                copy_height = cols; // 列主序时，高度是列数
            }

            CUDA_CHECK(cudaMemcpy2D(temp_device_input.getData(), dst_pitch, // 目标设备指针和 pitch
                                    host_src_ptr, src_pitch,               // 源主机指针和 pitch
                                    copy_width_bytes, copy_height,         // 复制尺寸
                                    cudaMemcpyHostToDevice));              // 传输方向

            device_input_ptr = temp_device_input.getData(); // 核函数使用临时设备矩阵的数据
            kernel_input_lda = temp_device_lda; // 核函数读取数据的 LDA 是临时设备矩阵的 LDA
        } else { // Location::DEVICE
            // 输入已在设备：直接使用设备视图的数据指针和 LDA
            device_input_ptr = input_matrix.getData();
            kernel_input_lda = input_lda;
        }

        // 3. 分配输出设备矩阵
        DeviceMatrix<T> output_matrix(rows, cols, layout);
        T* device_output_ptr = output_matrix.getData();
        const auto output_lda = output_matrix.getLeadingDimension(); // 输出矩阵的 LDA

        // 4. 配置并启动 CUDA 核函数
        // 使用 16x16 的线程块大小（可根据 GPU 架构调整）
        constexpr int BLOCK_DIM_X = 16;
        constexpr int BLOCK_DIM_Y = 16;
        const dim3 threadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
        // 计算覆盖整个矩阵所需的线程块网格大小
        const dim3 blocksPerGrid(
            (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

#ifdef __CUDACC__
        // 启动核函数
        extractTriangleKernel<<<blocksPerGrid, threadsPerBlock>>>(
            device_input_ptr,   // 输入设备数据
            device_output_ptr,  // 输出设备数据
            rows, cols,         // 矩阵维度
            kernel_input_lda,   // 输入数据的 LDA
            output_lda,         // 输出数据的 LDA
            layout,             // 内存布局
            triangle_type);     // 提取类型 (UPPER/LOWER)

        // 检查核函数启动和执行过程中是否发生错误
        CUDA_CHECK(cudaGetLastError());
        // 可选：如果需要立即在主机端访问结果，可以取消注释下一行
        // CUDA_CHECK(cudaDeviceSynchronize()); // 也应在 #ifdef 内
#else
        // 如果从非 nvcc 编译的代码调用，则抛出错误
        throw std::runtime_error("extractTriangle can only be called from code compiled with nvcc.");
        // 注意：如果此函数旨在在主机编译器编译时具有不同的行为，
        // 则应在此处添加替代逻辑。
#endif // __CUDACC__

        // 5. 返回包含结果的设备矩阵
        // 此 return 语句假定函数要么完成 CUDA 部分，
        // 要么在从错误的上下文调用时抛出异常。
        return output_matrix;
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
