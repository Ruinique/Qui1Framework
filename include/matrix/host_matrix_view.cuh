#ifndef HOST_MATRIX_VIEW_CUH
#define HOST_MATRIX_VIEW_CUH

#include "matrix_view.cuh"
#include "host_matrix.cuh"

template <typename T>
class HostMatrixView : public MatrixView<T> {
public:
    // Type aliases
    using value_type = T;
    using size_type = std::size_t;
    using pointer = T*;
    using const_pointer = const T*;

    // Default constructor (empty view)
    HostMatrixView() noexcept = default;

    // Constructor from a HostMatrix (creates a view of the whole matrix)
    explicit HostMatrixView(HostMatrix<T>& matrix) noexcept
        : MatrixView<T>(matrix.data(), matrix.rows(), matrix.cols(), matrix.leading_dimension(), matrix.layout()) {}

    // Const overload for creating a view from a const HostMatrix
    explicit HostMatrixView(const HostMatrix<T>& matrix) noexcept
        : MatrixView<T>(const_cast<typename HostMatrix<T>::pointer>(matrix.data()), matrix.rows(), matrix.cols(), matrix.leading_dimension(), matrix.layout()) {}

    // Const overload for creating a view from a const HostMatrix
    explicit HostMatrixView(const HostMatrix<T>& matrix) noexcept
        : MatrixView<T>(matrix) {}
};

#endif // HOST_MATRIX_VIEW_CUH
