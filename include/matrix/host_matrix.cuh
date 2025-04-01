#ifndef HOST_MATRIX_CUH
#define HOST_MATRIX_CUH

#include <cuda_runtime.h> // CUDA runtime API
#include <cstddef>        // For size_t
#include <stdexcept>      // For runtime_error (optional, for error handling)
#include "../common/error_check.cuh" // Assuming CUDA_CHECK macro is defined here
#include "common_matrix.cuh"         // Include the base Matrix class

template <typename T>
class HostMatrix : public Matrix<T> {
public:
    // Type aliases
    using value_type = T;
    using size_type = std::size_t;
    using pointer = T*;
    using const_pointer = const T*;

    // Default constructor (empty matrix)
    HostMatrix() noexcept = default;

    // Constructor: Allocates host memory using cudaMallocHost
    explicit HostMatrix(size_type rows, size_type cols, typename Matrix<T>::Layout layout)
        : rows_(rows), cols_(cols), layout_(layout) {
        if (rows == 0 || cols == 0) {
            // Handle zero-sized matrix allocation gracefully
            rows_ = 0;
            cols_ = 0;
            data_ = nullptr; // No allocation needed
            return;
        }
        size_type num_elements = rows * cols;
        CUDA_CHECK(cudaMallocHost(&data_, num_elements * sizeof(value_type)));
    }

    // Destructor: Frees host memory
    ~HostMatrix() {
        if (data_) {
            CUDA_CHECK(cudaFreeHost(data_));
            data_ = nullptr; // Prevent double-free
        }
    }

    // Copy constructor: Performs deep copy
    HostMatrix(const HostMatrix& other)
        : rows_(other.rows_), cols_(other.cols_), data_(nullptr), layout_(other.layout_) {
        if (!other.data_) { // Handle copying an empty matrix
            return;
        }
        size_type num_elements = rows_ * cols_;
        if (num_elements > 0) {
            CUDA_CHECK(cudaMallocHost(&data_, num_elements * sizeof(value_type)));
            std::copy(other.data_, other.data_ + num_elements, data_);
        }
    }

    // Copy assignment operator
    HostMatrix& operator=(const HostMatrix& other) {
        if (this == &other) {
            return *this; // Handle self-assignment
        }

        // Release existing resource
        if (data_) {
            CUDA_CHECK(cudaFreeHost(data_));
            data_ = nullptr;
        }

        // Copy data and dimensions from other
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = nullptr; // Ensure data_ is null if allocation fails or size is zero
        layout_ = other.layout_;

        if (other.data_ && rows_ > 0 && cols_ > 0) {
            size_type num_elements = rows_ * cols_;
            CUDA_CHECK(cudaMallocHost(&data_, num_elements * sizeof(value_type)));
            std::copy(other.data_, other.data_ + num_elements, data_);
        }
        return *this;
    }

    // Move constructor: Transfers ownership
    HostMatrix(HostMatrix&& other) noexcept
        : rows_(other.rows_), cols_(other.cols_), data_(other.data_), layout_(other.layout_) {
        // Leave other in a valid, destructible state (empty)
        other.rows_ = 0;
        other.cols_ = 0;
        other.data_ = nullptr;
        other.layout_ = typename Matrix<T>::Layout::RowMajor;
    }

    // Move assignment operator
    HostMatrix& operator=(HostMatrix&& other) noexcept {
        if (this == &other) {
            return *this; // Handle self-assignment
        }

        // Release existing resource
        if (data_) {
            CUDA_CHECK(cudaFreeHost(data_));
        }

        // Transfer ownership from other
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = other.data_;
        layout_ = other.layout_;

        // Leave other in a valid, destructible state
        other.rows_ = 0;
        other.cols_ = 0;
        other.data_ = nullptr;
        other.layout_ = typename Matrix<T>::Layout::RowMajor;

        return *this;
    }

    // --- Accessors ---

    [[nodiscard]] pointer data() noexcept { return data_; }
    [[nodiscard]] const_pointer data() const noexcept { return data_; }
    [[nodiscard]] typename Matrix<T>::Layout layout() const noexcept { return layout_; }
    [[nodiscard]] size_type rows() const noexcept override { return rows_; }
    [[nodiscard]] size_type cols() const noexcept override { return cols_; }
    [[nodiscard]] size_type size() const noexcept override { return rows_ * cols_; } // Total elements

private:
    pointer data_ = nullptr;
    size_type rows_ = 0;
    size_type cols_ = 0;
    typename Matrix<T>::Layout layout_;
};

#endif // HOST_MATRIX_CUH
