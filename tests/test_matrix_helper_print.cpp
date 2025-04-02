#include <gtest/gtest.h>
#include <sstream> // To potentially capture output, though not strictly verifying format here
#include <vector>
#include <iostream> // For std::cout redirection (optional)

#include "matrix/qui1_device_matrix.cuh"
#include "matrix/qui1_host_matrix.cuh"
#include "matrix/qui1_matrix_helper.cuh"

// Test fixture for MatrixHelper print tests
class MatrixHelperPrintTest : public ::testing::Test {
protected:
    // Using float for simplicity in tests
    using DataType = float;
    static constexpr size_t rows = 2;
    static constexpr size_t cols = 3;
};

// Test printMatrix for an empty HostMatrix
TEST_F(MatrixHelperPrintTest, PrintEmptyHostMatrix) {
    qui1::HostMatrix<DataType> host_matrix; // Default constructor, empty
    EXPECT_NO_THROW(qui1::MatrixHelper::printMatrix(host_matrix, "Empty Host Matrix Test"));
    // We expect it to print a message indicating it's empty, but primarily check it doesn't crash.
}

// Test printMatrix for an empty DeviceMatrix
TEST_F(MatrixHelperPrintTest, PrintEmptyDeviceMatrix) {
    qui1::DeviceMatrix<DataType> device_matrix; // Default constructor, empty
    EXPECT_NO_THROW(qui1::MatrixHelper::printMatrix(device_matrix, "Empty Device Matrix Test"));
}

// Test printMatrix for a filled HostMatrix
TEST_F(MatrixHelperPrintTest, PrintFilledHostMatrix) {
    qui1::HostMatrix<DataType> host_matrix(rows, cols);
    // Fill with some simple data
    std::vector<DataType> data = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f};
    std::copy(data.begin(), data.end(), host_matrix.getData());

    // Redirect cout to check if *something* is printed (optional, basic check)
    // std::stringstream buffer;
    // std::streambuf* old_cout = std::cout.rdbuf(buffer.rdbuf());

    EXPECT_NO_THROW(qui1::MatrixHelper::printMatrix(host_matrix, "Filled Host Matrix Test"));

    // std::cout.rdbuf(old_cout); // Restore cout
    // EXPECT_FALSE(buffer.str().empty()); // Check if output was generated
    // EXPECT_NE(buffer.str().find("1.1"), std::string::npos); // Basic check for content
}

// Test printMatrix for a filled DeviceMatrix
TEST_F(MatrixHelperPrintTest, PrintFilledDeviceMatrix) {
    qui1::DeviceMatrix<DataType> device_matrix(rows, cols);
    // Fill with some simple data on the host first
    std::vector<DataType> host_data = {10.1f, 20.2f, 30.3f, 40.4f, 50.5f, 60.6f};
    CUDA_CHECK(cudaMemcpy(device_matrix.getData(), host_data.data(),
                          rows * cols * sizeof(DataType), cudaMemcpyHostToDevice));

    EXPECT_NO_THROW(qui1::MatrixHelper::printMatrix(device_matrix, "Filled Device Matrix Test"));
    // Similar to the host test, we primarily check for no exceptions.
    // Verifying exact output after device-to-host copy within printMatrix is complex for a simple test.
}

// Test printMatrix with different layouts (RowMajor vs ColumnMajor)
TEST_F(MatrixHelperPrintTest, PrintDifferentLayouts) {
    // Row Major
    qui1::HostMatrix<DataType> row_major_matrix(rows, cols, qui1::Layout::ROW_MAJOR);
    std::vector<DataType> data_rm = {1, 2, 3, 4, 5, 6};
    std::copy(data_rm.begin(), data_rm.end(), row_major_matrix.getData());
    EXPECT_NO_THROW(qui1::MatrixHelper::printMatrix(row_major_matrix, "Row Major Host"));

    // Column Major
    qui1::HostMatrix<DataType> col_major_matrix(rows, cols, qui1::Layout::COLUMN_MAJOR);
     // Data order for column major: [1, 4] [2, 5] [3, 6] -> stored as [1, 4, 2, 5, 3, 6]
    std::vector<DataType> data_cm = {1, 4, 2, 5, 3, 6};
    std::copy(data_cm.begin(), data_cm.end(), col_major_matrix.getData());
    EXPECT_NO_THROW(qui1::MatrixHelper::printMatrix(col_major_matrix, "Column Major Host"));

    // Device versions
    qui1::DeviceMatrix<DataType> device_rm_matrix(rows, cols, qui1::Layout::ROW_MAJOR);
    CUDA_CHECK(cudaMemcpy(device_rm_matrix.getData(), data_rm.data(), data_rm.size() * sizeof(DataType), cudaMemcpyHostToDevice));
    EXPECT_NO_THROW(qui1::MatrixHelper::printMatrix(device_rm_matrix, "Row Major Device"));

    qui1::DeviceMatrix<DataType> device_cm_matrix(rows, cols, qui1::Layout::COLUMN_MAJOR);
    CUDA_CHECK(cudaMemcpy(device_cm_matrix.getData(), data_cm.data(), data_cm.size() * sizeof(DataType), cudaMemcpyHostToDevice));
    EXPECT_NO_THROW(qui1::MatrixHelper::printMatrix(device_cm_matrix, "Column Major Device"));
}
