#include <gtest/gtest.h>

#include <limits>  // For std::numeric_limits
#include <vector>

#include "matrix/qui1_device_matrix.cuh"
#include "matrix/qui1_matrix_helper.cuh"
#include "wrapper/blas/qui1_cublas_wrapper.cuh"

// Test fixture for CublasWrapper TRSM tests
class CublasTrsmWrapperTest : public ::testing::Test {
   protected:
    // Using float for simplicity in tests
    using DataType = float;
    // A is m x m, B is m x n (for side = left)
    static constexpr size_t m = 3;
    static constexpr size_t n = 2;
    static constexpr DataType alpha = 1.0f;

    qui1::DeviceMatrix<DataType> A;
    qui1::DeviceMatrix<DataType> B; // B will be overwritten by X

    void SetUp() override {
        A = qui1::DeviceMatrix<DataType>(m, m);
        B = qui1::DeviceMatrix<DataType>(m, n);

        // Initialize A as an upper triangular matrix (example)
        // 1 2 3
        // 0 4 5
        // 0 0 6
        std::vector<DataType> host_data_A = {1.0f, 0.0f, 0.0f,  // Column 0
                                             2.0f, 4.0f, 0.0f,  // Column 1
                                             3.0f, 5.0f, 6.0f}; // Column 2
        CUDA_CHECK(cudaMemcpy(A.getData(), host_data_A.data(),
                              m * m * sizeof(DataType), cudaMemcpyHostToDevice));

        // Initialize B with some values
        // 7  8
        // 9 10
        //11 12
        std::vector<DataType> host_data_B = {7.0f, 9.0f, 11.0f, // Column 0
                                             8.0f, 10.0f, 12.0f}; // Column 1
        CUDA_CHECK(cudaMemcpy(B.getData(), host_data_B.data(),
                              m * n * sizeof(DataType), cudaMemcpyHostToDevice));
    }
};

// Basic test case for float TRSM (Left, Upper, No Transpose, Non-Unit)
TEST_F(CublasTrsmWrapperTest, floatLeftUpperNoTransNonUnit) {
    // Directly construct the views instead of using getView
    qui1::DeviceMatrixView<DataType> A_view(A.getData(), m, m, A.getLayout(),
                                            A.getLeadingDimension());
    qui1::DeviceMatrixView<DataType> B_view(B.getData(), m, n, B.getLayout(),
                                            B.getLeadingDimension());

    qui1::CublasWrapper trsm_wrapper;
    // Correct signature: trsm(A, B, side, uplo, trans, diag, alpha)
    EXPECT_NO_THROW(trsm_wrapper.trsm(A_view, B_view,
                                      CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                      CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                      alpha));

    fmt::print("Result matrix B (X) after TRSM (Left, Upper, NoTrans, NonUnit):\n");
    qui1::MatrixHelper::printMatrix(B_view);

    // Verification step
    std::vector<DataType> host_result(m * n);
    CUDA_CHECK(cudaMemcpy(host_result.data(), B_view.getData(),
                          m * n * sizeof(DataType), cudaMemcpyDeviceToHost));

    std::vector<DataType> expected_result = {
        19.0f / 12.0f, -1.0f / 24.0f, 11.0f / 6.0f, // Column 0
        2.0f, 0.0f, 2.0f                           // Column 1
    };

    ASSERT_EQ(host_result.size(), expected_result.size());
    for (size_t i = 0; i < host_result.size(); ++i) {
        // Use ASSERT_NEAR for floating-point comparisons
        ASSERT_NEAR(host_result[i], expected_result[i], 1e-5f);
    }
}

// Test case for double TRSM (Left, Upper, No Transpose, Non-Unit)
TEST_F(CublasTrsmWrapperTest, doubleLeftUpperNoTransNonUnit) {
    using DoubleType = double;
    // Need to create double matrices A_d, B_d
    qui1::DeviceMatrix<DoubleType> A_d(m, m);
    qui1::DeviceMatrix<DoubleType> B_d(m, n);

    // Initialize A_d as an upper triangular matrix
    std::vector<DoubleType> host_data_A_d = {1.0, 0.0, 0.0,  // Column 0
                                             2.0, 4.0, 0.0,  // Column 1
                                             3.0, 5.0, 6.0}; // Column 2
    CUDA_CHECK(cudaMemcpy(A_d.getData(), host_data_A_d.data(),
                          m * m * sizeof(DoubleType), cudaMemcpyHostToDevice));

    // Initialize B_d with some values
    std::vector<DoubleType> host_data_B_d = {7.0, 9.0, 11.0, // Column 0
                                             8.0, 10.0, 12.0}; // Column 1
    CUDA_CHECK(cudaMemcpy(B_d.getData(), host_data_B_d.data(),
                          m * n * sizeof(DoubleType), cudaMemcpyHostToDevice));

    // Directly construct the views
    qui1::DeviceMatrixView<DoubleType> A_view_d(A_d.getData(), m, m, A_d.getLayout(),
                                                A_d.getLeadingDimension());
    qui1::DeviceMatrixView<DoubleType> B_view_d(B_d.getData(), m, n, B_d.getLayout(),
                                                B_d.getLeadingDimension());

    qui1::CublasWrapper trsm_wrapper;
    DoubleType alpha_d = 1.0;
    EXPECT_NO_THROW(trsm_wrapper.trsm(A_view_d, B_view_d,
                                      CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                      CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                      alpha_d));

    fmt::print("Result matrix B (X) after TRSM (Double, Left, Upper, NoTrans, NonUnit):\n");
    qui1::MatrixHelper::printMatrix(B_view_d);

    // Verification step
    std::vector<DoubleType> host_result_d(m * n);
    CUDA_CHECK(cudaMemcpy(host_result_d.data(), B_view_d.getData(),
                          m * n * sizeof(DoubleType), cudaMemcpyDeviceToHost));

    std::vector<DoubleType> expected_result_d = {
        19.0 / 12.0, -1.0 / 24.0, 11.0 / 6.0, // Column 0
        2.0, 0.0, 2.0                           // Column 1
    };

    ASSERT_EQ(host_result_d.size(), expected_result_d.size());
    for (size_t i = 0; i < host_result_d.size(); ++i) {
        ASSERT_NEAR(host_result_d[i], expected_result_d[i], 1e-9); // Smaller tolerance for double
    }
}


// Test case for float TRSM (Left, Lower, No Transpose, Non-Unit)
TEST_F(CublasTrsmWrapperTest, floatLeftLowerNoTransNonUnit) {
    // Re-initialize A as a lower triangular matrix for this test
    // 1 0 0
    // 2 4 0
    // 3 5 6
    std::vector<DataType> host_data_A_lower = {1.0f, 2.0f, 3.0f,  // Column 0
                                               0.0f, 4.0f, 5.0f,  // Column 1
                                               0.0f, 0.0f, 6.0f}; // Column 2
    CUDA_CHECK(cudaMemcpy(A.getData(), host_data_A_lower.data(),
                          m * m * sizeof(DataType), cudaMemcpyHostToDevice));

    // B is initialized fresh by the fixture for each test

    qui1::DeviceMatrixView<DataType> A_view(A.getData(), m, m, A.getLayout(),
                                            A.getLeadingDimension());
    qui1::DeviceMatrixView<DataType> B_view(B.getData(), m, n, B.getLayout(),
                                            B.getLeadingDimension());

    qui1::CublasWrapper trsm_wrapper;
    EXPECT_NO_THROW(trsm_wrapper.trsm(A_view, B_view,
                                      CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, // Changed uplo
                                      CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                      alpha));

    fmt::print("Result matrix B (X) after TRSM (Left, Lower, NoTrans, NonUnit):\n");
    qui1::MatrixHelper::printMatrix(B_view);

    // Verification step
    std::vector<DataType> host_result(m * n);
    CUDA_CHECK(cudaMemcpy(host_result.data(), B_view.getData(),
                          m * n * sizeof(DataType), cudaMemcpyDeviceToHost));

    // Corrected expected result
    std::vector<DataType> expected_result = {
        7.0f, -1.25f, -0.625f, // Column 0
        8.0f, -1.5f,  -0.75f   // Column 1
    };

    ASSERT_EQ(host_result.size(), expected_result.size());
    for (size_t i = 0; i < host_result.size(); ++i) {
        ASSERT_NEAR(host_result[i], expected_result[i], 1e-5f);
    }
}

// Test case for float TRSM using sub-views (Left, Upper, No Transpose, Non-Unit)
TEST_F(CublasTrsmWrapperTest, floatLeftUpperNoTransNonUnitView) {
    // Use 2x2 subviews
    // A_sub = [[1, 2], [0, 4]] (from original A upper left)
    // B_sub = [[7, 8], [9, 10]] (from original B upper left)
    // Solve A_sub * X_sub = B_sub, result X_sub overwrites B_sub

    // Ensure A is upper triangular for this test
    std::vector<DataType> host_data_A_upper = {1.0f, 0.0f, 0.0f,  // Column 0
                                               2.0f, 4.0f, 0.0f,  // Column 1
                                               3.0f, 5.0f, 6.0f}; // Column 2
    CUDA_CHECK(cudaMemcpy(A.getData(), host_data_A_upper.data(),
                          m * m * sizeof(DataType), cudaMemcpyHostToDevice));
    // Re-initialize B as well, since the view operation modifies it in place
    std::vector<DataType> host_data_B_orig = {7.0f, 9.0f, 11.0f, // Column 0
                                              8.0f, 10.0f, 12.0f}; // Column 1
    CUDA_CHECK(cudaMemcpy(B.getData(), host_data_B_orig.data(),
                          m * n * sizeof(DataType), cudaMemcpyHostToDevice));


    // Create 2x2 views starting at (0,0)
    // Note: DeviceMatrixView constructor takes offset in elements, not rows/cols
    // For column-major A (3x3, ld=3): offset for (0,0) is 0*3 + 0 = 0
    // For column-major B (3x2, ld=3): offset for (0,0) is 0*3 + 0 = 0
    qui1::DeviceMatrixView<DataType> A_view(A.getData(), 2, 2, A.getLayout(),
                                            A.getLeadingDimension(), 0);
    // B_view represents the 2x2 submatrix of B where the result X_sub will be stored
    qui1::DeviceMatrixView<DataType> B_view(B.getData(), 2, 2, B.getLayout(),
                                            B.getLeadingDimension(), 0);

    qui1::CublasWrapper trsm_wrapper;
    EXPECT_NO_THROW(trsm_wrapper.trsm(A_view, B_view,
                                      CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                                      CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                                      alpha));

    fmt::print("Result matrix B after TRSM on View (Original B):\n");
    // Print the whole B matrix to see the changes in the top-left 2x2 part
    qui1::MatrixHelper::printMatrix(qui1::DeviceMatrixView<DataType>(B.getData(), m, n, B.getLayout(), B.getLeadingDimension()));

    // Verification step: Check the modified 2x2 part of B
    std::vector<DataType> host_result_full(m * n);
    CUDA_CHECK(cudaMemcpy(host_result_full.data(), B.getData(),
                          m * n * sizeof(DataType), cudaMemcpyDeviceToHost));

    // Expected result for the 2x2 subproblem X_sub:
    // A_sub * X_sub = B_sub
    // [[1, 2], [0, 4]] * [[x00, x01], [x10, x11]] = [[7, 8], [9, 10]]
    // Col 0: 1*x00 + 2*x10 = 7; 0*x00 + 4*x10 = 9 => x10 = 9/4; x00 = 7 - 2*(9/4) = 7 - 9/2 = 5/2
    // Col 1: 1*x01 + 2*x11 = 8; 0*x01 + 4*x11 = 10 => x11 = 10/4 = 5/2; x01 = 8 - 2*(5/2) = 8 - 5 = 3
    // X_sub (ColMajor) = {5/2, 9/4, 3, 5/2} = {2.5f, 2.25f, 3.0f, 2.5f}

    std::vector<DataType> expected_sub_result = { 2.5f, 2.25f, 3.0f, 2.5f };
    std::vector<DataType> actual_sub_result = { host_result_full[0], host_result_full[1], // Col 0 of subview
                                                host_result_full[3], host_result_full[4] }; // Col 1 of subview

    ASSERT_EQ(actual_sub_result.size(), expected_sub_result.size());
    for (size_t i = 0; i < actual_sub_result.size(); ++i) {
        ASSERT_NEAR(actual_sub_result[i], expected_sub_result[i], 1e-5f);
    }

    // Also check that the rest of B remains unchanged
    ASSERT_EQ(host_result_full[2], host_data_B_orig[2]); // Element (2,0)
    ASSERT_EQ(host_result_full[5], host_data_B_orig[5]); // Element (2,1)
}
