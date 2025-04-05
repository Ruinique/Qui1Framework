#include <gtest/gtest.h>
#include <numeric> // For std::accumulate
#include <vector>
#include <limits> // For std::numeric_limits

#include "matrix/qui1_device_matrix.cuh"
#include "matrix/qui1_host_matrix.cuh"
#include "matrix/qui1_matrix_helper.cuh"

// Test fixture for MatrixHelper tests
class MatrixHelperFillRandomTest : public ::testing::Test {
protected:
    // Using float for simplicity in tests
    using DataType = float;
    static constexpr size_t rows = 5;
    static constexpr size_t cols = 4;
};

// Test fillWithRandom for HostMatrix
TEST_F(MatrixHelperFillRandomTest, FillHostMatrix) {
    qui1::HostMatrix<DataType> host_matrix(rows, cols);
    EXPECT_TRUE(host_matrix.hasData()); // Ensure allocation succeeded

    // Fill with random numbers
    qui1::MatrixHelper::fillWithRandom(host_matrix);

    // Verify: Check if the data is actually filled (not all zeros)
    // We copy data to a vector for easier inspection on the host.
    std::vector<DataType> data_vec(rows * cols);
    // HostMatrix data is already on the host, just copy it
    std::copy(host_matrix.getData(), host_matrix.getData() + rows * cols, data_vec.begin());

    // Check if at least one element is non-zero (a basic check for randomness)
    // Or calculate sum/average if more robustness is needed.
    double sum = std::accumulate(data_vec.begin(), data_vec.end(), 0.0);

    // Random numbers are uniform between 0.0 and 1.0.
    // The sum should be roughly (rows * cols) / 2.
    // We check if the sum is greater than a small epsilon, indicating it's not all zeros.
    EXPECT_GT(sum, std::numeric_limits<double>::epsilon());
    // Also check it's not excessively large (sanity check)
    EXPECT_LT(sum, static_cast<double>(rows * cols));
}

// Test fillWithRandom for DeviceMatrix
TEST_F(MatrixHelperFillRandomTest, FillDeviceMatrix) {
    qui1::DeviceMatrix<DataType> device_matrix(rows, cols);
    EXPECT_TRUE(device_matrix.hasData()); // Ensure allocation succeeded

    // Fill with random numbers
    qui1::MatrixHelper::fillWithRandom(device_matrix);

    // Verify: Copy data back to host and check if it's filled
    std::vector<DataType> host_data(rows * cols);
    CUDA_CHECK(cudaMemcpy(host_data.data(), device_matrix.getData(),
                          rows * cols * sizeof(DataType), cudaMemcpyDeviceToHost));

    // Check if at least one element is non-zero
    double sum = std::accumulate(host_data.begin(), host_data.end(), 0.0);
    EXPECT_GT(sum, std::numeric_limits<double>::epsilon());
    EXPECT_LT(sum, static_cast<double>(rows * cols));
}

// Test fillWithRandom with a specific seed
TEST_F(MatrixHelperFillRandomTest, FillWithSeed) {
    qui1::HostMatrix<DataType> matrix1(rows, cols);
    qui1::HostMatrix<DataType> matrix2(rows, cols);
    unsigned long long seed = 98765ULL;

    qui1::MatrixHelper::fillWithRandom(matrix1, seed);
    qui1::MatrixHelper::fillWithRandom(matrix2, seed);

    // Verify that matrices filled with the same seed have the same data
    std::vector<DataType> data1(rows * cols);
    std::vector<DataType> data2(rows * cols);
    std::copy(matrix1.getData(), matrix1.getData() + rows * cols, data1.begin());
    std::copy(matrix2.getData(), matrix2.getData() + rows * cols, data2.begin());

    ASSERT_EQ(data1.size(), data2.size());
    for (size_t i = 0; i < data1.size(); ++i) {
        EXPECT_EQ(data1[i], data2[i]) << "Mismatch at index " << i;
    }
}

// Test attempting to fill an unallocated matrix (should throw)
TEST_F(MatrixHelperFillRandomTest, FillUnallocatedMatrix) {
    qui1::HostMatrix<DataType> unallocated_matrix; // Default constructor, no allocation
    EXPECT_FALSE(unallocated_matrix.hasData());
    EXPECT_THROW(qui1::MatrixHelper::fillWithRandom(unallocated_matrix), std::runtime_error);

    // Also test with a moved-from matrix
    qui1::HostMatrix<DataType> source_matrix(rows, cols);
    qui1::HostMatrix<DataType> moved_matrix = std::move(source_matrix);
    EXPECT_FALSE(source_matrix.hasData()); // source_matrix should be empty now
    EXPECT_THROW(qui1::MatrixHelper::fillWithRandom(source_matrix), std::runtime_error);
}
