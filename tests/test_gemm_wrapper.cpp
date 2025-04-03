#include <gtest/gtest.h>

#include <limits>  // For std::numeric_limits
#include <vector>

#include "matrix/qui1_device_matrix.cuh"
#include "matrix/qui1_matrix_helper.cuh"
#include "wrapper/blas/qui1_gemm_wrapper.cuh"

// Test fixture for GemmWrapper tests
class GemmWrapperTest : public ::testing::Test {
   protected:
    // Using float for simplicity in tests
    using DataType = float;
    static constexpr size_t rows_A = 3;
    static constexpr size_t cols_A = 2;
    static constexpr size_t cols_B = 4;

    qui1::DeviceMatrix<DataType> A;
    qui1::DeviceMatrix<DataType> B;
    qui1::DeviceMatrix<DataType> C;

    void SetUp() override {
        A = qui1::DeviceMatrix<DataType>(rows_A, cols_A);
        B = qui1::DeviceMatrix<DataType>(cols_A, cols_B);
        C = qui1::DeviceMatrix<DataType>(rows_A, cols_B);

        // Fill matrices A and B with known values
        std::vector<DataType> host_data_A(rows_A * cols_A, 1.0f);
        std::vector<DataType> host_data_B(cols_A * cols_B, 2.0f);
        CUDA_CHECK(cudaMemcpy(A.getData(), host_data_A.data(),
                              rows_A * cols_A * sizeof(DataType),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(B.getData(), host_data_B.data(),
                              cols_A * cols_B * sizeof(DataType),
                              cudaMemcpyHostToDevice));
    }
};

TEST_F(GemmWrapperTest, floatAmultiplyB) {
    auto A_view = A.getView(A.getRows(), A.getCols());
    auto B_view = B.getView(B.getRows(), B.getCols());
    auto C_view = C.getView(C.getRows(), C.getCols());

    // 调用 GEMM 操作
    qui1::GemmWrapper gemm_wrapper;
    EXPECT_NO_THROW(gemm_wrapper.gemm(A_view, B_view, C_view));

    qui1::MatrixHelper::printMatrix(C_view);
}

TEST_F(GemmWrapperTest, doubleAmultiplyB) {
    using DataType = double;
    qui1::DeviceMatrix<DataType> A;
    qui1::DeviceMatrix<DataType> B;
    qui1::DeviceMatrix<DataType> C;
    A = qui1::DeviceMatrix<DataType>(rows_A, cols_A);
    B = qui1::DeviceMatrix<DataType>(cols_A, cols_B);
    C = qui1::DeviceMatrix<DataType>(rows_A, cols_B);

    // Fill matrices A and B with known values
    std::vector<DataType> host_data_A(rows_A * cols_A, 1.0f);
    std::vector<DataType> host_data_B(cols_A * cols_B, 2.0f);
    CUDA_CHECK(cudaMemcpy(A.getData(), host_data_A.data(),
                          rows_A * cols_A * sizeof(DataType),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B.getData(), host_data_B.data(),
                          cols_A * cols_B * sizeof(DataType),
                          cudaMemcpyHostToDevice));

    auto A_view = A.getView(A.getRows(), A.getCols());
    auto B_view = B.getView(B.getRows(), B.getCols());
    auto C_view = C.getView(C.getRows(), C.getCols());

    // 调用 GEMM 操作
    qui1::GemmWrapper gemm_wrapper;
    EXPECT_NO_THROW(gemm_wrapper.gemm(A_view, B_view, C_view));

    qui1::MatrixHelper::printMatrix(C_view);
}

TEST_F(GemmWrapperTest, floatAmultiplyBView) {
    auto A_view = A.getView(2, 2, 0, 0);
    auto B_view = B.getView(2, 2, 0, 0);
    auto C_view = C.getView(2, 2, 1, 1);

    // 调用 GEMM 操作
    qui1::GemmWrapper gemm_wrapper;
    EXPECT_NO_THROW(gemm_wrapper.gemm(A_view, B_view, C_view));

    qui1::MatrixHelper::printMatrix(C_view);
    qui1::MatrixHelper::printMatrix(C.getView(C.getRows(), C.getCols()));
    
}