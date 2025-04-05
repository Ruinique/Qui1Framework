#include <cmath> // For std::sqrt
#include <gtest/gtest.h>

#include "matrix/qui1_device_matrix.cuh"
#include "wrapper/blas/qui1_cublas_wrapper.cuh"

class NrmWrapperTest : public ::testing::Test {
   protected:
    // Using float for simplicity in tests
    using DataType = float;
    static constexpr size_t rows = 2;
    static constexpr size_t cols = 3;
};

TEST_F(NrmWrapperTest, ComputeNormalizationFloat) {
    qui1::DeviceMatrix<DataType> A;
    A = qui1::DeviceMatrix<DataType>(rows, cols);
    std::vector<DataType> host_data(rows * cols, 2.0f);
    CUDA_CHECK(cudaMemcpy(A.getData(), host_data.data(),
                          rows * cols * sizeof(DataType), cudaMemcpyHostToDevice));
    qui1::CublasWrapper nrm_wrapper;
    auto result = nrm_wrapper.nrm(A.getView(A.getRows(), A.getCols())); 
    // Calculate expected norm: sqrt(6 * (2.0f * 2.0f)) = sqrt(24.0f)
    const DataType expected_norm = std::sqrt(static_cast<DataType>(rows * cols * 4.0f));
    ASSERT_NEAR(result, expected_norm, 1e-6); // Assert the result
}
