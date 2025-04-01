#ifndef MATRIX_VIEW_CUH
#define MATRIX_VIEW_CUH

#include <cstddef> // For size_t
#include "common_matrix.cuh" // Base Matrix class

template <typename T>
class MatrixView {
public:
    // Type aliases
    using value_type = T;
    using size_type = std::size_t;
    using pointer = T*;
    using const_pointer = const T*;

    // Rule of Zero - Views are typically cheap to copy, default members are fine

    // Default constructor (empty view)
    MatrixView() noexcept = default;

    // Constructor from raw pointer and dimensions
    // Assumes column-major layout if leading_dimension is not provided
    MatrixView(pointer data, size_type rows, size_type cols, size_type leading_dimension, typename Matrix<T>::Layout layout) noexcept
        : data_(data), rows_(rows), cols_(cols), leading_dimension_(leading_dimension), layout_(layout) {}

    // Constructor assuming contiguous column-major (leading_dimension = rows)
    MatrixView(pointer data, size_type rows, size_type cols, typename Matrix<T>::Layout layout) noexcept
        : MatrixView(data, rows, cols, rows, layout) {} // Delegate constructor (C++11)

    // Constructor from a DeviceMatrix (creates a view of the whole matrix)
    template <typename MatrixType>
    explicit MatrixView(MatrixType& matrix) noexcept
        : data_(matrix.data()),
          rows_(matrix.rows()),
          cols_(matrix.cols()),
          leading_dimension_(matrix.leading_dimension()),
          layout_(matrix.layout()) {}

    // Const overload for creating a view from a const Matrix
    template <typename MatrixType>
    explicit MatrixView(const MatrixType& matrix) noexcept
        : data_(const_cast<pointer>(matrix.data())), // Const cast needed for pointer type
          rows_(matrix.rows()),
          cols_(matrix.cols()),
          leading_dimension_(matrix.leading_dimension()),
          layout_(matrix.layout()) {
        // Note: A const MatrixView might be more appropriate here,
        // or enforce const-correctness via methods if T is non-const.
        // For now, this creates a non-const view of potentially const data.
        // Consider adding a ConstMatrixView class later.
    }

    // --- Accessors ---

    [[nodiscard]] pointer data() const noexcept { return data_; }
    [[nodiscard]] size_type rows() const noexcept { return rows_; }
    [[nodiscard]] size_type cols() const noexcept { return cols_; }
    [[nodiscard]] size_type size() const noexcept { return rows_ * cols_; } // Logical size
    [[nodiscard]] size_type leading_dimension() const noexcept { return leading_dimension_; }
    [[nodiscard]] typename Matrix<T>::Layout layout() const noexcept { return layout_; }

private:
    pointer data_ = nullptr; // Non-owning pointer
    size_type rows_ = 0;
    size_type cols_ = 0;
    size_type leading_dimension_ = 0; // Stride between columns (in elements)
    typename Matrix<T>::Layout layout_;
};

#endif // MATRIX_VIEW_CUH
