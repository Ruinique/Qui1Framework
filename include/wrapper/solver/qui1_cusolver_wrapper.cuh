#ifndef QUI1_CUSOLVER_WRAPPER_H
#define QUI1_CUSOLVER_WRAPPER_H

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <stdexcept>
#include <type_traits>
#include <vector>  // 包含 vector 以便将来可能使用

#include "common/error_check.cuh"
#include "fmt/format.h"
#include "matrix/view/qui1_device_matrix_view.cuh"

namespace qui1 {

class CusolverWrapper {
   public:
    // 构造函数：创建 cuSOLVER 句柄
    CusolverWrapper() { CUSOLVER_CHECK(cusolverDnCreate(&handle_)); }

    // 析构函数：销毁 cuSOLVER 句柄
    ~CusolverWrapper() { cleanup(); }

    // 禁止拷贝构造函数
    CusolverWrapper(const CusolverWrapper&) = delete;
    // 禁止拷贝赋值运算符
    CusolverWrapper& operator=(const CusolverWrapper&) = delete;

    // 移动构造函数
    CusolverWrapper(CusolverWrapper&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    // 移动赋值运算符
    CusolverWrapper& operator=(CusolverWrapper&& other) noexcept {
        if (this != &other) {
            cleanup();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief 使用部分主元法 (Partial Pivoting) 计算通用 M x N 矩阵 A 的 LU 分解。
     * @warning
     * 此函数始终执行部分主元选取以保证数值稳定性，无法执行无主元选取的分解。
     *
     * 分解形式为 A = P * L * U，其中 P 是置换矩阵，L 是单位对角线的下三角矩阵
     * (如果 m > n，则为下梯形)，U 是上三角矩阵 (如果 m < n，则为上梯形)。
     *
     * @tparam T 数据类型 (float 或 double)。
     * @param A 输入/输出: 输入时为 M x N 矩阵 A。输出时为因子 L 和 U (A = P*L*U)；
     *          L 的单位对角线元素不存储。假定为列主序 (Column Major) 格式。
     * @param devIpiv (可选) 输出: 指向设备端整数数组的指针，维度至少为 min(M,N)。
     *                如果提供，将存储主元索引。第 j 个主元索引表示第 j 行与第
     * IPIV(j) 行 进行了交换 (使用 1-based 索引)。调用者负责分配和释放此内存。
     *                如果传入 `nullptr` (默认)，则函数内部会处理主元信息，但不返回。
     * @throws std::runtime_error 如果分解失败 (例如，矩阵奇异，U(i,i) == 0)。
     * @throws std::invalid_argument 如果矩阵不是列主序。
     */
    template <typename T>
    void getrf(DeviceMatrixView<T>& A, int* devIpiv = nullptr) {
        if (A.getLayout() != qui1::Layout::COLUMN_MAJOR) {
            throw std::invalid_argument(
                fmt::format("cusolverDnGetrf (with pivoting) currently requires "
                            "Column Major layout."));
        }

        auto m = static_cast<int>(A.getRows());
        auto n = static_cast<int>(A.getCols());
        auto lda = static_cast<int>(A.getLDA());
        T* data_ptr = A.getData();

        // --- 内部管理 info ---
        int* internal_devInfo = nullptr;
        CUDA_CHECK(
            cudaMalloc(reinterpret_cast<void**>(&internal_devInfo), sizeof(int)));

        // --- 处理 devIpiv ---
        int* actual_devIpiv = devIpiv == nullptr ? NULL : devIpiv;
        int lwork = 0;  // 工作空间大小

        // 1. 计算工作空间大小
        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_CHECK(
                cusolverDnSgetrf_bufferSize(handle_, m, n, data_ptr, lda, &lwork));
        } else if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_CHECK(
                cusolverDnDgetrf_bufferSize(handle_, m, n, data_ptr, lda, &lwork));
        } else {
            // 使用 fmt::format 输出错误信息
            throw std::runtime_error(
                fmt::format("getrf_with_pivoting: Unsupported data type. Only float "
                            "and double are supported."));
        }

        // 2. 分配工作空间
        T* devWork = nullptr;
        // 检查 lwork 是否大于 0，避免分配 0 字节内存（虽然 cudaMalloc
        // 可能处理，但显式检查更安全）
        if (lwork > 0) {
            CUDA_CHECK(
                cudaMalloc(reinterpret_cast<void**>(&devWork), sizeof(T) * lwork));
        }

        // 3. 执行 LU 分解
        // 确保即使 m 或 n 为 0，actual_devIpiv 也是一个有效（可能是 nullptr）或
        // cusolver 能接受的值
        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_CHECK(cusolverDnSgetrf(handle_, m, n, data_ptr, lda, devWork,
                                            actual_devIpiv, internal_devInfo));
        } else if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_CHECK(cusolverDnDgetrf(handle_, m, n, data_ptr, lda, devWork,
                                            actual_devIpiv, internal_devInfo));
        }
        // 类型检查已在 bufferSize 时完成

        // 4. 检查分解结果 (info)
        int hostInfo = 0;
        CUDA_CHECK(cudaMemcpy(&hostInfo, internal_devInfo, sizeof(int),
                              cudaMemcpyDeviceToHost));

        // 5. 清理资源
        if (devWork) {
            CUDA_CHECK(cudaFree(devWork));
        }
        CUDA_CHECK(cudaFree(internal_devInfo));  // 释放内部 info

        // 6. 如果分解失败，抛出异常
        if (hostInfo > 0) {
            throw std::runtime_error(
                fmt::format("LU decomposition failed: U({},{}) is exactly zero. The "
                            "matrix is singular.",
                            hostInfo, hostInfo));
        } else if (hostInfo < 0) {
            // CUSOLVER_CHECK 应该已经捕获了 < 0 的情况并退出，但以防万一
            throw std::runtime_error(
                fmt::format("LU decomposition failed: Invalid argument at position "
                            "{}. This should have been caught by CUSOLVER_CHECK.",
                            -hostInfo));
        }
        // hostInfo == 0 表示成功
    }

   private:
    cusolverDnHandle_t handle_{nullptr};

    // 清理句柄的辅助函数
    void cleanup() {
        if (handle_) {
            // 检查销毁操作的状态，虽然通常不处理这里的错误，但加上以保持一致性
            /* cusolverStatus_t status = */ cusolverDnDestroy(handle_);
            // CUSOLVER_CHECK(status); // 在析构函数中抛出异常通常是不推荐的
            handle_ = nullptr;
        }
    }
};

}  // namespace qui1

#endif  // QUI1_CUSOLVER_WRAPPER_H
