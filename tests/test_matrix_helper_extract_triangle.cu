#include <gtest/gtest.h>
#include <vector>
#include <fmt/core.h>

#include "matrix/qui1_host_matrix.cuh"
#include "matrix/qui1_device_matrix.cuh"
#include "matrix/view/qui1_host_matrix_view.cuh"
#include "matrix/view/qui1_device_matrix_view.cuh"
#include "matrix/qui1_matrix_helper.cuh"
#include "common/error_check.cuh" // For CUDA_CHECK

using namespace qui1;

// Helper function to compare two host matrices (vectors)
template <typename T>
void compareMatrices(const std::vector<T>& mat1, const std::vector<T>& mat2, size_t rows, size_t cols, Layout layout) {
    ASSERT_EQ(mat1.size(), mat2.size());
    ASSERT_EQ(mat1.size(), rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            size_t index = (layout == Layout::ROW_MAJOR) ? (i * cols + j) : (j * rows + i);
            EXPECT_NEAR(mat1[index], mat2[index], 1e-6) // Use EXPECT_NEAR for floating point types
                << "Mismatch at (" << i << ", " << j << ") index " << index;
        }
    }
}

// Test fixture for ExtractTriangle tests
class ExtractTriangleTest : public ::testing::Test {
protected:
    const size_t rows = 4;
    const size_t cols = 4;
    std::vector<float> host_data_row_major;
    std::vector<float> host_data_col_major;
    std::vector<float> expected_upper_row_major;
    std::vector<float> expected_lower_row_major;
    std::vector<float> expected_upper_col_major;
    std::vector<float> expected_lower_col_major;

    void SetUp() override {
        // Initialize input data (Row Major)
        host_data_row_major = {
             1.0f,  2.0f,  3.0f,  4.0f,
             5.0f,  6.0f,  7.0f,  8.0f,
             9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f
        };
        // Expected Upper Triangle (Row Major)
        expected_upper_row_major = {
             1.0f,  2.0f,  3.0f,  4.0f,
             0.0f,  6.0f,  7.0f,  8.0f,
             0.0f,  0.0f, 11.0f, 12.0f,
             0.0f,  0.0f,  0.0f, 16.0f
        };
        // Expected Lower Triangle (Row Major)
        expected_lower_row_major = {
             1.0f,  0.0f,  0.0f,  0.0f,
             5.0f,  6.0f,  0.0f,  0.0f,
             9.0f, 10.0f, 11.0f,  0.0f,
            13.0f, 14.0f, 15.0f, 16.0f
        };

        // Initialize input data (Column Major) - Transpose of row major for clarity
         host_data_col_major = {
             1.0f,  5.0f,  9.0f, 13.0f,
             2.0f,  6.0f, 10.0f, 14.0f,
             3.0f,  7.0f, 11.0f, 15.0f,
             4.0f,  8.0f, 12.0f, 16.0f
         };
        // Expected Upper Triangle (Column Major) - Stored linearly column by column
        expected_upper_col_major = {
             1.0f,  0.0f,  0.0f,  0.0f, // Column 0 (Elements (0,0), (1,0), (2,0), (3,0))
             2.0f,  6.0f,  0.0f,  0.0f, // Column 1 (Elements (0,1), (1,1), (2,1), (3,1))
             3.0f,  7.0f, 11.0f,  0.0f, // Column 2 (Elements (0,2), (1,2), (2,2), (3,2))
             4.0f,  8.0f, 12.0f, 16.0f  // Column 3 (Elements (0,3), (1,3), (2,3), (3,3))
        };
         // Expected Lower Triangle (Column Major) - Stored linearly column by column
        expected_lower_col_major = {
             1.0f,  5.0f,  9.0f, 13.0f, // Column 0
             0.0f,  6.0f, 10.0f, 14.0f, // Column 1
             0.0f,  0.0f, 11.0f, 15.0f, // Column 2
             0.0f,  0.0f,  0.0f, 16.0f  // Column 3
        };
    }

    // Helper to run the test logic for a given layout and input location
    template<Layout L, Location Loc>
    void runTest(TriangleType type) {
        const auto& host_data = (L == Layout::ROW_MAJOR) ? host_data_row_major : host_data_col_major;
        const auto& expected_data = (L == Layout::ROW_MAJOR) ?
                                        ((type == TriangleType::UPPER) ? expected_upper_row_major : expected_lower_row_major) :
                                        ((type == TriangleType::UPPER) ? expected_upper_col_major : expected_lower_col_major);

        DeviceMatrix<float> result_device_matrix;

        if constexpr (Loc == Location::HOST) {
            // Create Host Matrix and View
            HostMatrix<float> host_matrix(rows, cols, L);
            std::copy(host_data.begin(), host_data.end(), host_matrix.getData());
            // Construct view using parameters from the matrix, calculating natural LDA
            const size_t host_lda = (host_matrix.getLayout() == Layout::ROW_MAJOR) ? host_matrix.getCols() : host_matrix.getRows();
            HostMatrixView<float> host_view(host_matrix.getData(), host_matrix.getRows(), host_matrix.getCols(), host_matrix.getLayout(), host_lda);

            // Call extractTriangle with Host View
            result_device_matrix = MatrixHelper::extractTriangle(host_view, type);
        } else { // Location::DEVICE
            // Create Device Matrix and View
            DeviceMatrix<float> device_matrix(rows, cols, L);
            CUDA_CHECK(cudaMemcpy(device_matrix.getData(), host_data.data(), host_data.size() * sizeof(float), cudaMemcpyHostToDevice));
            // Construct view using parameters from the matrix, calculating natural LDA
            const size_t device_lda = (device_matrix.getLayout() == Layout::ROW_MAJOR) ? device_matrix.getCols() : device_matrix.getRows();
            DeviceMatrixView<float> device_view(device_matrix.getData(), device_matrix.getRows(), device_matrix.getCols(), device_matrix.getLayout(), device_lda);

            // Call extractTriangle with Device View
            result_device_matrix = MatrixHelper::extractTriangle(device_view, type);
        }

        // Verify layout and dimensions
        ASSERT_EQ(result_device_matrix.getRows(), rows);
        ASSERT_EQ(result_device_matrix.getCols(), cols);
        ASSERT_EQ(result_device_matrix.getLayout(), L);

        // Copy result back to host
        std::vector<float> result_host_data(rows * cols);
        CUDA_CHECK(cudaMemcpy(result_host_data.data(), result_device_matrix.getData(), result_host_data.size() * sizeof(float), cudaMemcpyDeviceToHost));

        // Compare with expected result
        compareMatrices(result_host_data, expected_data, rows, cols, L);
    }
};

// --- Test Cases ---

TEST_F(ExtractTriangleTest, HostInputRowMajorUpper) {
    runTest<Layout::ROW_MAJOR, Location::HOST>(TriangleType::UPPER);
}

TEST_F(ExtractTriangleTest, HostInputRowMajorLower) {
    runTest<Layout::ROW_MAJOR, Location::HOST>(TriangleType::LOWER);
}

TEST_F(ExtractTriangleTest, HostInputColMajorUpper) {
    runTest<Layout::COLUMN_MAJOR, Location::HOST>(TriangleType::UPPER);
}

TEST_F(ExtractTriangleTest, HostInputColMajorLower) {
    runTest<Layout::COLUMN_MAJOR, Location::HOST>(TriangleType::LOWER);
}

TEST_F(ExtractTriangleTest, DeviceInputRowMajorUpper) {
    runTest<Layout::ROW_MAJOR, Location::DEVICE>(TriangleType::UPPER);
}

TEST_F(ExtractTriangleTest, DeviceInputRowMajorLower) {
    runTest<Layout::ROW_MAJOR, Location::DEVICE>(TriangleType::LOWER);
}

TEST_F(ExtractTriangleTest, DeviceInputColMajorUpper) {
    runTest<Layout::COLUMN_MAJOR, Location::DEVICE>(TriangleType::UPPER);
}

TEST_F(ExtractTriangleTest, DeviceInputColMajorLower) {
    runTest<Layout::COLUMN_MAJOR, Location::DEVICE>(TriangleType::LOWER);
}

// Test with a non-square matrix (e.g., 3x5)
TEST(ExtractTriangleNonSquareTest, RowMajorUpper3x5) {
    const size_t rows = 3;
    const size_t cols = 5;
    const Layout layout = Layout::ROW_MAJOR;
    const TriangleType type = TriangleType::UPPER;

    std::vector<float> host_data = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15
    };
    std::vector<float> expected_data = {
        1, 2, 3, 4, 5,
        0, 7, 8, 9, 10,
        0, 0, 13, 14, 15
    };

    HostMatrix<float> host_matrix(rows, cols, layout);
    std::copy(host_data.begin(), host_data.end(), host_matrix.getData());
    // Construct view using parameters from the matrix, calculating natural LDA
    const size_t host_lda = (host_matrix.getLayout() == Layout::ROW_MAJOR) ? host_matrix.getCols() : host_matrix.getRows();
    HostMatrixView<float> host_view(host_matrix.getData(), host_matrix.getRows(), host_matrix.getCols(), host_matrix.getLayout(), host_lda);

    auto result_device_matrix = MatrixHelper::extractTriangle(host_view, type);

    ASSERT_EQ(result_device_matrix.getRows(), rows);
    ASSERT_EQ(result_device_matrix.getCols(), cols);
    ASSERT_EQ(result_device_matrix.getLayout(), layout);

    std::vector<float> result_host_data(rows * cols);
    CUDA_CHECK(cudaMemcpy(result_host_data.data(), result_device_matrix.getData(), result_host_data.size() * sizeof(float), cudaMemcpyDeviceToHost));

    compareMatrices(result_host_data, expected_data, rows, cols, layout);
}

TEST(ExtractTriangleNonSquareTest, ColMajorLower5x3) {
    const size_t rows = 5;
    const size_t cols = 3;
    const Layout layout = Layout::COLUMN_MAJOR;
    const TriangleType type = TriangleType::LOWER;

    // Input data (Column Major)
    std::vector<float> host_data = {
        1, 6, 11, // Col 0
        2, 7, 12, // Col 1
        3, 8, 13, // Col 2
        4, 9, 14, // Col 3
        5, 10, 15 // Element (4,2)
    };
     // Expected data (Lower Triangle of 5x3 matrix, stored Column Major)
     // Matrix:
     // 1  12   9
     // 6   3  14
     // 11  8   5
     // 2  13  10
     // 7   4  15
     // Lower Triangle (j <= i):
     // 1   0   0
     // 6   3   0
     // 11  8   5
     // 2  13  10
     // 7   4  15
    std::vector<float> expected_data = {
        1.0f, 6.0f, 11.0f, 2.0f, 7.0f,    // Column 0
        0.0f, 3.0f, 8.0f, 13.0f, 4.0f,    // Column 1
        0.0f, 0.0f, 5.0f, 10.0f, 15.0f    // Column 2
    };


    DeviceMatrix<float> device_matrix(rows, cols, layout);
    CUDA_CHECK(cudaMemcpy(device_matrix.getData(), host_data.data(), host_data.size() * sizeof(float), cudaMemcpyHostToDevice));
    // Construct view using parameters from the matrix, calculating natural LDA
    const size_t device_lda = (device_matrix.getLayout() == Layout::ROW_MAJOR) ? device_matrix.getCols() : device_matrix.getRows();
    DeviceMatrixView<float> device_view(device_matrix.getData(), device_matrix.getRows(), device_matrix.getCols(), device_matrix.getLayout(), device_lda);

    auto result_device_matrix = MatrixHelper::extractTriangle(device_view, type);

    ASSERT_EQ(result_device_matrix.getRows(), rows);
    ASSERT_EQ(result_device_matrix.getCols(), cols);
    ASSERT_EQ(result_device_matrix.getLayout(), layout);

    std::vector<float> result_host_data(rows * cols);
    CUDA_CHECK(cudaMemcpy(result_host_data.data(), result_device_matrix.getData(), result_host_data.size() * sizeof(float), cudaMemcpyDeviceToHost));

    compareMatrices(result_host_data, expected_data, rows, cols, layout);
}
