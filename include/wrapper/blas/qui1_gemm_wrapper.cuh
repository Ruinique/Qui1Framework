#ifndef QUI1_GEMM_WRAPPER_H
#define QUI1_GEMM_WRAPPER_H

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <type_traits>

#include "common/error_check.cuh"
#include "fmt/format.h"
#include "matrix/view/qui1_device_matrix_view.cuh"

namespace qui1 {

class GemmWrapper {
   public:
    // 默认构造函数，初始化 cuBLAS 句柄
    GemmWrapper() { CUBLAS_CHECK(cublasCreate(&handle_)); }

    // 禁止拷贝构造函数
    GemmWrapper(const GemmWrapper&) = delete;

    // 禁止拷贝赋值运算符
    GemmWrapper& operator=(const GemmWrapper&) = delete;

    // 移动构造函数，转移句柄所有权
    GemmWrapper(GemmWrapper&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    // 移动赋值运算符，转移句柄所有权
    GemmWrapper& operator=(GemmWrapper&& other) noexcept {
        if (this != &other) {
            cleanup();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    // 析构函数，销毁 cuBLAS 句柄
    ~GemmWrapper() { cleanup(); }

    template <typename T>
    void gemm(DeviceMatrixView<T>& A, DeviceMatrixView<T>& B,
              DeviceMatrixView<T>& C) {
        if (A.getCols() != B.getRows()) {
            throw std::invalid_argument(fmt::format("A cols != B rows"));
        }
        auto k = A.getCols();
        auto m = A.getRows();
        auto n = B.getCols();
        auto aLayout = A.getLayout();
        auto bLayout = B.getLayout();
        auto alpha = T{1};
        auto beta = T{0};
        cublasOperation_t transa = CUBLAS_OP_N;
        cublasOperation_t transb = CUBLAS_OP_N;

        if (aLayout != bLayout) {
            throw std::invalid_argument(
                fmt::format("A layout and B layout must be the same"));
        }

        if (aLayout == qui1::Layout::ROW_MAJOR) {
            transa = CUBLAS_OP_T;
            transb = CUBLAS_OP_T;
        }

        if constexpr (std::is_same_v<T, float>) {
            CUBLAS_CHECK(cublasSgemm_v2(
                handle_, transa, transb, static_cast<int>(m), static_cast<int>(n),
                static_cast<int>(k), &alpha, A.getData(), static_cast<int>(A.getLDA()),
                B.getData(), static_cast<int>(B.getLDA()), &beta, C.getData(),
                static_cast<int>(C.getLDA())));
        } else if constexpr (std::is_same_v<T, double>) {
            CUBLAS_CHECK(cublasDgemm_v2(
                handle_, transa, transb, static_cast<int>(m), static_cast<int>(n),
                static_cast<int>(k), &alpha, A.getData(), static_cast<int>(A.getLDA()),
                B.getData(), static_cast<int>(B.getLDA()), &beta, C.getData(),
                static_cast<int>(C.getLDA())));
        } else {
            throw std::runtime_error(fmt::format("Unsupported data type"));
        }
    }

   private:
    cublasHandle_t handle_{nullptr};

    // 清理句柄的辅助函数
    void cleanup() {
        if (handle_) {
            cublasDestroy(handle_);
            handle_ = nullptr;
        }
    }
};
}  // namespace qui1

#endif
