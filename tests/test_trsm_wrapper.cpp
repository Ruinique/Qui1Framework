#include <gtest/gtest.h>

#include <limits>  // For std::numeric_limits
#include <vector>
#include <cmath> // For std::abs
#include <fmt/core.h> // For fmt::print

#include "matrix/qui1_device_matrix.cuh"
#include "matrix/qui1_matrix_helper.cuh"
#include "matrix/qui1_host_matrix.cuh" // For HostMatrix used in verification
#include "wrapper/blas/qui1_cublas_wrapper.cuh"
#include "common/error_check.cuh" // For CUDA_CHECK

// Test fixture for CublasWrapper TRSM tests
class CublasTrsmWrapperTest : public ::testing::Test {
   protected:
    // Using float for simplicity in tests
    using DataType = float;
    // For op(A) * X = alpha * B, A is m x m, B is m x n, X is m x n
    // For X * op(A) = alpha * B, A is n x n, B is m x n, X is m x n
    // Let's test op(A) * X = alpha * B with A being 3x3, B being 3x4
    static constexpr size_t m = 3;
    static constexpr size_t n = 4;
    static constexpr float alpha = 1.0f;

    qui1::DeviceMatrix<DataType> A; // Square matrix (m x m)
    qui1::DeviceMatrix<DataType> B; // Result matrix (m x n), overwritten by X
    qui1::DeviceMatrix<DataType> B_original; // To store the original B for verification

    qui1::CublasWrapper cublas_wrapper; // Instantiate wrapper

    void SetUp() override {
        A = qui1::DeviceMatrix<DataType>(m, m);
        B = qui1::DeviceMatrix<DataType>(m, n);
        B_original = qui1::DeviceMatrix<DataType>(m, n);

        // Fill matrix A with known upper triangular values
        // A = [ 1 2 3 ]
        //     [ 0 4 5 ]
        //     [ 0 0 6 ]
        std::vector<DataType> host_data_A = {1.0f, 0.0f, 0.0f,
                                             2.0f, 4.0f, 0.0f,
                                             3.0f, 5.0f, 6.0f};
        CUDA_CHECK(cudaMemcpy(A.getData(), host_data_A.data(),
                              m * m * sizeof(DataType),
                              cudaMemcpyHostToDevice));

        // Fill matrix B with known values
        // B = [ 1 1 1 1 ]
        //     [ 1 1 1 1 ]
        //     [ 1 1 1 1 ]
        std::vector<DataType> host_data_B(m * n, 1.0f);
        CUDA_CHECK(cudaMemcpy(B.getData(), host_data_B.data(),
                              m * n * sizeof(DataType),
                              cudaMemcpyHostToDevice));
        // Copy B to B_original
        CUDA_CHECK(cudaMemcpy(B_original.getData(), B.getData(),
                              m * n * sizeof(DataType),
                              cudaMemcpyDeviceToDevice));

        fmt::print("Matrix A ({}x{}):\n", A.getRows(), A.getCols());
        qui1::MatrixHelper::printMatrix(A.getView(A.getRows(), A.getCols()));
        fmt::print("Matrix B_original ({}x{}):\n", B_original.getRows(), B_original.getCols());
        qui1::MatrixHelper::printMatrix(B_original.getView(B_original.getRows(), B_original.getCols()));
    }

    void TearDown() override {
        // Optional: Add cleanup code if needed
    }
};

// Test case for float trsm: op(A) * X = alpha * B
// Using A as upper triangular, no transpose, unit diagonal = false
TEST_F(CublasTrsmWrapperTest, floatTrsmLeftUpperNoTrans) {
    auto A_view = A.getView(m, m);
    auto B_view = B.getView(m, n); // B will be overwritten with solution X

    fmt::print("Matrix B before trsm ({}x{}):\n", B_view.getRows(), B_view.getCols());
    qui1::MatrixHelper::printMatrix(B_view);

    // Call trsm: Solve A * X = alpha * B for X
    // side = CUBLAS_SIDE_LEFT, uplo = CUBLAS_FILL_MODE_UPPER,
    // trans = CUBLAS_OP_N, diag = CUBLAS_DIAG_NON_UNIT
    // Use the updated wrapper signature
    cublas_wrapper.trsm(CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                        alpha, A_view, B_view); // Removed EXPECT_NO_THROW

    fmt::print("Matrix B after trsm (Solution X) ({}x{}):\n", B_view.getRows(), B_view.getCols());
    qui1::MatrixHelper::printMatrix(B_view);

    // Verification: Calculate A * X and compare with alpha * B_original
    qui1::DeviceMatrix<DataType> C_verification(m, n);
    auto C_verification_view = C_verification.getView(m, n);
    // Note: B_view now contains X (the solution)
    cublas_wrapper.gemm(A_view, B_view, C_verification_view); // Removed EXPECT_NO_THROW

    fmt::print("Verification: A * X ({}x{}):\n", C_verification_view.getRows(), C_verification_view.getCols());
    qui1::MatrixHelper::printMatrix(C_verification_view);

    // Copy results back to host for comparison
    qui1::HostMatrix<DataType> host_C_verification(m, n);
    qui1::HostMatrix<DataType> host_B_original(m, n);

    CUDA_CHECK(cudaMemcpy(host_C_verification.getData(), C_verification.getData(),
                          m * n * sizeof(DataType), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_B_original.getData(), B_original.getData(),
                          m * n * sizeof(DataType), cudaMemcpyDeviceToHost));

    // Compare C_verification (A*X) with B_original (alpha * B) element-wise
    const DataType tolerance = 1e-5f; // Tolerance for float comparison
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            DataType expected_value = alpha * host_B_original(i, j);
            DataType actual_value = host_C_verification(i, j);
            EXPECT_NEAR(actual_value, expected_value, tolerance)
                << "Mismatch at (" << i << ", " << j << ")";
        }
    }
}

// TODO: Add more test cases for different parameters (double, side, uplo, trans, diag)
// Example: Lower triangular, Transposed, Double precision, Right side etc.
