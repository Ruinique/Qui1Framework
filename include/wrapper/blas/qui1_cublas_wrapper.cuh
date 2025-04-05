#ifndef QUI1_CUBLAS_WRAPPER_H
#define QUI1_CUBLAS_WRAPPER_H

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <type_traits>

#include "common/error_check.cuh"
#include "fmt/format.h"
#include "matrix/view/qui1_device_matrix_view.cuh"

namespace qui1 {

class CublasWrapper {
   public:
    // 默认构造函数，初始化 cuBLAS 句柄
    CublasWrapper() { CUBLAS_CHECK(cublasCreate(&handle_)); }

    // 禁止拷贝构造函数
    CublasWrapper(const CublasWrapper&) = delete;

    // 禁止拷贝赋值运算符
    CublasWrapper& operator=(const CublasWrapper&) = delete;

    // 移动构造函数，转移句柄所有权
    CublasWrapper(CublasWrapper&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    // 移动赋值运算符，转移句柄所有权
    CublasWrapper& operator=(CublasWrapper&& other) noexcept {
        if (this != &other) {
            cleanup();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    // 析构函数，销毁 cuBLAS 句柄
    ~CublasWrapper() { cleanup(); }

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
            CUBLAS_CHECK(cublasSgemm_v2(handle_, transa, transb, static_cast<int>(m),
                                        static_cast<int>(n), static_cast<int>(k),
                                        &alpha, A.getData(),
                                        static_cast<int>(A.getLDA()), B.getData(),
                                        static_cast<int>(B.getLDA()), &beta,
                                        C.getData(), static_cast<int>(C.getLDA())));
        } else if constexpr (std::is_same_v<T, double>) {
            CUBLAS_CHECK(cublasDgemm_v2(handle_, transa, transb, static_cast<int>(m),
                                        static_cast<int>(n), static_cast<int>(k),
                                        &alpha, A.getData(),
                                        static_cast<int>(A.getLDA()), B.getData(),
                                        static_cast<int>(B.getLDA()), &beta,
                                        C.getData(), static_cast<int>(C.getLDA())));
        } else {
            throw std::runtime_error(fmt::format("Unsupported data type"));
        }
    }

    // Solves op(A) * X = alpha * B or X * op(A) = alpha * B
    template <typename T>
    void trsm(DeviceMatrixView<T>& A, DeviceMatrixView<T>& B,
              cublasSideMode_t side = CUBLAS_SIDE_LEFT,
              cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER,
              cublasOperation_t trans = CUBLAS_OP_N,
              cublasDiagType_t diag = CUBLAS_DIAG_UNIT, T alpha =1) {
        // Determine matrix dimensions based on 'side'
        int m, n;
        if (side == CUBLAS_SIDE_LEFT) {
            // op(A) is m x m, B is m x n
            m = static_cast<int>(A.getRows());  // A must be square
            n = static_cast<int>(B.getCols());
            if (A.getRows() != A.getCols() || A.getRows() != B.getRows()) {
                throw std::invalid_argument(fmt::format(
                    "Invalid dimensions for TRSM with SIDE_LEFT: A({}x{}), B({}x{})",
                    A.getRows(), A.getCols(), B.getRows(), B.getCols()));
            }
        } else {  // CUBLAS_SIDE_RIGHT
            // op(A) is n x n, B is m x n
            m = static_cast<int>(B.getRows());
            n = static_cast<int>(A.getRows());  // A must be square
            if (A.getRows() != A.getCols() || A.getRows() != B.getCols()) {
                throw std::invalid_argument(
                    fmt::format("Invalid dimensions for TRSM with SIDE_RIGHT: "
                                "A({}x{}), B({}x{})",
                                A.getRows(), A.getCols(), B.getRows(), B.getCols()));
            }
        }

        auto lda = static_cast<int>(A.getLDA());
        auto ldb = static_cast<int>(B.getLDA());
        auto A_ptr = A.getData();
        auto B_ptr = B.getData();  // B is input and output

        if constexpr (std::is_same_v<T, float>) {
            CUBLAS_CHECK(cublasStrsm(handle_, side, uplo, trans, diag, m, n, &alpha,
                                     A_ptr, lda, B_ptr, ldb));
        } else if constexpr (std::is_same_v<T, double>) {
            CUBLAS_CHECK(cublasDtrsm(handle_, side, uplo, trans, diag, m, n, &alpha,
                                     A_ptr, lda, B_ptr, ldb));
        } else {
            throw std::runtime_error(fmt::format("Unsupported data type"));
        }
    }

    /**
     * Only used by a entire matrix stored contiguously
     */
    template <typename T>
    T nrm(const DeviceMatrixView<T>& A) {
        auto x = A.getData();
        auto n = A.getCols() * A.getRows();
        T result = 0;
        if constexpr (std::is_same_v<T, float>) {
            CUBLAS_CHECK(cublasSnrm2_v2(handle_, n, x, 1, &result));
        } else if constexpr (std::is_same_v<T, double>) {
            CUBLAS_CHECK(cublasDnrm2_v2(handle_, n, x, 1, &result));
        }
        return result;
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
