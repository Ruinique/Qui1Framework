#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath> // For std::abs
#include <numeric> // For std::iota
#include <algorithm> // For std::swap

#include "common/error_check.cuh"
#include "matrix/qui1_device_matrix.cuh"
#include "matrix/qui1_host_matrix.cuh"
#include "matrix/qui1_matrix_helper.cuh"
#include "wrapper/solver/qui1_cusolver_wrapper.cuh"
#include "wrapper/blas/qui1_cublas_wrapper.cuh" // Include for potential GEMM use
#include "fmt/format.h" // For potential debug output

// Helper function to check if a matrix is lower triangular (unit diagonal)
template <typename T>
bool isUnitLowerTriangular(const qui1::HostMatrix<T>& matrix, T tolerance = 1e-6) {
    if (matrix.getRows() != matrix.getCols()) return false; // Must be square
    const auto rows = matrix.getRows(); // Use auto
    const auto cols = matrix.getCols(); // Use auto
    const auto layout = matrix.getLayout();
    const auto lda = matrix.getLeadingDimension();
    const T* data = matrix.getData();

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const auto index = (layout == qui1::Layout::COLUMN_MAJOR) ? (j * lda + i) : (i * lda + j); // Use auto and const
            if (i == j) { // Diagonal
                if (std::abs(data[index] - static_cast<T>(1.0)) > tolerance) {
                    // fmt::print("FAIL Diagonal: L({},{}) = {}\n", i, j, data[index]);
                    return false;
                }
            } else if (j > i) { // Upper part
                if (std::abs(data[index] - static_cast<T>(0.0)) > tolerance) {
                     // fmt::print("FAIL Upper: L({},{}) = {}\n", i, j, data[index]);
                    return false;
                }
            }
            // Lower part can be anything
        }
    }
    return true;
}

// Helper function to check if a matrix is upper triangular
template <typename T>
bool isUpperTriangular(const qui1::HostMatrix<T>& matrix, T tolerance = 1e-6) {
     if (matrix.getRows() != matrix.getCols()) return false; // Must be square
    const auto rows = matrix.getRows(); // Use auto
    const auto cols = matrix.getCols(); // Use auto
    const auto layout = matrix.getLayout();
    const auto lda = matrix.getLeadingDimension();
    const T* data = matrix.getData();

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
             const auto index = (layout == qui1::Layout::COLUMN_MAJOR) ? (j * lda + i) : (i * lda + j); // Use auto and const
            if (i > j) { // Lower part
                if (std::abs(data[index] - static_cast<T>(0.0)) > tolerance) {
                    // fmt::print("FAIL Lower: U({},{}) = {}\n", i, j, data[index]);
                    return false;
                }
            }
            // Upper part and diagonal can be anything
        }
    }
    return true;
}

// Helper function to compare two matrices (element-wise)
template <typename T>
bool areMatricesClose(const qui1::HostMatrix<T>& A, const qui1::HostMatrix<T>& B, T tolerance = 1e-5) {
    if (A.getRows() != B.getRows() || A.getCols() != B.getCols() || A.getLayout() != B.getLayout()) {
        fmt::print("Matrix dimension or layout mismatch for comparison.\n");
        return false;
    }
    const auto rows = A.getRows();
    const auto cols = A.getCols();
    const auto layout = A.getLayout(); // Assume A and B have same layout checked above
    const auto lda_A = A.getLeadingDimension();
    const auto lda_B = B.getLeadingDimension();
    const T* data_A = A.getData();
    const T* data_B = B.getData();

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const auto index_A = (layout == qui1::Layout::COLUMN_MAJOR) ? (j * lda_A + i) : (i * lda_A + j);
            const auto index_B = (layout == qui1::Layout::COLUMN_MAJOR) ? (j * lda_B + i) : (i * lda_B + j);
            if (std::abs(data_A[index_A] - data_B[index_B]) > tolerance) {
                fmt::print("Mismatch at ({}, {}): A = {}, B = {}\n", i, j, data_A[index_A], data_B[index_B]);
                return false;
            }
        }
    }
    return true;
}

// Helper function for basic host matrix multiplication (C = A * B)
// Assumes COLUMN_MAJOR for simplicity matching cuSolver/BLAS defaults often
template <typename T>
qui1::HostMatrix<T> multiplyHostMatrices(const qui1::HostMatrix<T>& A, const qui1::HostMatrix<T>& B) {
    // Use COLUMN_MAJOR consistently
    if (A.getCols() != B.getRows() || A.getLayout() != qui1::Layout::COLUMN_MAJOR || B.getLayout() != qui1::Layout::COLUMN_MAJOR) {
        throw std::invalid_argument("Invalid dimensions or layout for host matrix multiplication (COLUMN_MAJOR required).");
    }
    const auto M = A.getRows(); // Use auto
    const auto N = B.getCols(); // Use auto
    const auto K = A.getCols(); // == B.getRows() // Use auto
    qui1::HostMatrix<T> C(M, N, qui1::Layout::COLUMN_MAJOR); // Initialize with zeros

    const T* a_ptr = A.getData();
    const T* b_ptr = B.getData();
    T* c_ptr = C.getData();
    const auto lda = A.getLeadingDimension(); // Use auto and const
    const auto ldb = B.getLeadingDimension(); // Use auto and const
    const auto ldc = C.getLeadingDimension(); // Use auto and const

    // Standard GEMM loop for Column Major C = A * B
    for (size_t j = 0; j < N; ++j) {      // Iterate over columns of C and B
        for (size_t i = 0; i < M; ++i) {  // Iterate over rows of C and A
            T sum = static_cast<T>(0.0);
            for (size_t k = 0; k < K; ++k) { // Iterate over columns of A and rows of B
                // C(i, j) += A(i, k) * B(k, j)
                // Indexing for Column Major: element(row, col) = data[row + col * lda]
                sum += a_ptr[i + k * lda] * b_ptr[k + j * ldb];
            }
            c_ptr[i + j * ldc] = sum; // Correct indexing for C
        }
    }
    return C;
}

// Helper function to apply permutations P (from ipiv) to a matrix A -> P*A
// Modifies A in place. Assumes COLUMN_MAJOR.
template <typename T>
void applyPermutationRows(const std::vector<int>& ipiv, qui1::HostMatrix<T>& A) {
    // Use COLUMN_MAJOR consistently
    if (A.getLayout() != qui1::Layout::COLUMN_MAJOR) {
         throw std::invalid_argument("applyPermutationRows currently assumes COLUMN_MAJOR.");
    }
    const auto m = static_cast<int>(A.getRows()); // Use auto and const
    const auto n = static_cast<int>(A.getCols()); // Use auto and const
    const auto lda = static_cast<int>(A.getLeadingDimension()); // Use auto and const
    T* data = A.getData();

    // ipiv uses 1-based indexing from cuSolver/LAPACK
    // Apply swaps in reverse order to correctly reconstruct P*A from A and ipiv
    for (int i = static_cast<int>(ipiv.size()) - 1; i >= 0; --i) {
        const auto pivot_row = ipiv[i] - 1; // Convert to 0-based index, use auto and const
        if (pivot_row != i) {
            // Swap row i and row pivot_row
            for (int j = 0; j < n; ++j) { // Iterate through columns
                // Correct indexing for Column Major swap
                std::swap(data[i + j * lda], data[pivot_row + j * lda]);
            }
        }
    }
}


#include "matrix/view/qui1_device_matrix_view.cuh" // Need this include for DeviceMatrixView

TEST(CusolverWrapperTest, GetrfFloatColMajorAndVerify) {
    using DataType = float;
    const size_t M = 4;
    const size_t N = 4; // Use square matrix for simplicity now
    const qui1::Layout layout = qui1::Layout::COLUMN_MAJOR; // Fix: Use COLUMN_MAJOR

    // 1. Prepare Host Matrix A (Column Major) - Keep original for verification
    // Example matrix:
    //  2  3  1  5
    //  6  8  2  9
    //  4 10  7  1
    // 12 14  3  11
    std::vector<DataType> h_A_data_orig = {
        2.0f, 6.0f,  4.0f, 12.0f, // Col 0
        3.0f, 8.0f, 10.0f, 14.0f, // Col 1
        1.0f, 2.0f,  7.0f,  3.0f, // Col 2
        5.0f, 9.0f,  1.0f, 11.0f  // Col 3
    };
    // Fix: Create HostMatrix first, then copy data
    qui1::HostMatrix<DataType> h_A_orig(M, N, layout);
    CUDA_CHECK(cudaMemcpy(h_A_orig.getData(), h_A_data_orig.data(), h_A_data_orig.size() * sizeof(DataType), cudaMemcpyHostToHost));

    // Fix: Manually create copy for decomposition
    qui1::HostMatrix<DataType> h_A_for_decomp(M, N, layout);
    CUDA_CHECK(cudaMemcpy(h_A_for_decomp.getData(), h_A_orig.getData(), M * N * sizeof(DataType), cudaMemcpyHostToHost));


    // 2. Prepare Device Matrices
    qui1::DeviceMatrix<DataType> d_LU(M, N, layout); // Matrix for LU decomposition result
    // Fix: Use cudaMemcpy instead of CopyToDevice
    CUDA_CHECK(cudaMemcpy(d_LU.getData(), h_A_for_decomp.getData(), M * N * sizeof(DataType), cudaMemcpyHostToDevice));


    // 3. Prepare Pivot array
    int* d_Ipiv = nullptr;
    const int min_mn = static_cast<int>(std::min(M, N));
    CUDA_CHECK(cudaMalloc(&d_Ipiv, sizeof(int) * min_mn));

    // 4. Perform LU decomposition
    qui1::CusolverWrapper cusolver;
    // Fix: Manually construct the DeviceMatrixView
    qui1::DeviceMatrixView<DataType> d_LU_view(d_LU.getData(), M, N, layout, d_LU.getLeadingDimension());
    ASSERT_NO_THROW(cusolver.getrf(d_LU_view, d_Ipiv)); // Pass the view

    // 5. Extract L and U matrices using MatrixHelper::extractTriangle
    // Fix: extractTriangle returns a new DeviceMatrix and takes a view
    // Fix: Remove the last boolean argument from extractTriangle call
    // Fix: Reuse the existing d_LU_view. extractTriangle takes const&, so this is safe.
    qui1::DeviceMatrix<DataType> d_L_extracted = qui1::MatrixHelper::extractTriangle(d_LU_view, qui1::TriangleType::LOWER);
    qui1::DeviceMatrix<DataType> d_U_extracted = qui1::MatrixHelper::extractTriangle(d_LU_view, qui1::TriangleType::UPPER);


    // 6. Copy results back to host
    qui1::HostMatrix<DataType> h_L(M, N, layout);
    qui1::HostMatrix<DataType> h_U(M, N, layout);
    std::vector<int> h_Ipiv(min_mn);

    // Fix: Use cudaMemcpy instead of CopyToHost
    CUDA_CHECK(cudaMemcpy(h_L.getData(), d_L_extracted.getData(), M * N * sizeof(DataType), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_U.getData(), d_U_extracted.getData(), M * N * sizeof(DataType), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_Ipiv.data(), d_Ipiv, sizeof(int) * min_mn, cudaMemcpyDeviceToHost));

    // 7. Verification on Host
    // Manually adjust h_L to be unit lower triangular based on the extracted lower part
    // Fix: Use getData() and manual indexing
    auto* h_L_data = h_L.getData();
    const auto lda_L = h_L.getLeadingDimension();
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            const auto index = j * lda_L + i; // COLUMN_MAJOR index
            if (i == j) {
                h_L_data[index] = 1.0f; // Set diagonal to 1
            } else if (j > i) {
                h_L_data[index] = 0.0f; // Zero out upper part
            }
            // Keep the lower part as extracted
        }
    }
     // Manually adjust h_U to be upper triangular based on the extracted upper part
     // Fix: Use getData() and manual indexing
    auto* h_U_data = h_U.getData();
    const auto lda_U = h_U.getLeadingDimension();
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
             const auto index = j * lda_U + i; // COLUMN_MAJOR index
             if (i > j) {
                h_U_data[index] = 0.0f; // Zero out lower part
            }
            // Keep the upper part and diagonal as extracted
        }
    }

    // Check properties of L and U
    EXPECT_TRUE(isUnitLowerTriangular(h_L));
    EXPECT_TRUE(isUpperTriangular(h_U));

    // --- Rigorous check: Reconstruct P*L*U and compare to A_orig ---
    // Calculate L * U on host
    qui1::HostMatrix<DataType> h_LU_product = multiplyHostMatrices(h_L, h_U);

    // Apply permutations P to the L*U product (A = P*L*U -> P^-1 * A = L*U)
    // It's easier to apply P to A_orig and compare with L*U
    // Fix: Manually create copy for permutation
    qui1::HostMatrix<DataType> h_A_permuted(M, N, layout);
    CUDA_CHECK(cudaMemcpy(h_A_permuted.getData(), h_A_orig.getData(), M * N * sizeof(DataType), cudaMemcpyHostToHost));
    applyPermutationRows(h_Ipiv, h_A_permuted); // Apply P to the copy

    // Compare P*A_orig (which is now h_A_permuted) with L*U
    qui1::MatrixHelper::printMatrix(h_A_permuted, "P*A_orig");
    qui1::MatrixHelper::printMatrix(h_LU_product, "L*U");
    EXPECT_TRUE(areMatricesClose(h_A_permuted, h_LU_product, 1e-5f));


    // 8. Cleanup
    CUDA_CHECK(cudaFree(d_Ipiv));
}

// TODO: Add tests for double precision
// TODO: Add tests for non-square matrices (M > N and M < N)
