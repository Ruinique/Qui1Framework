#ifndef DEVICE_MATRIX_VIEW_CUH
#define DEVICE_MATRIX_VIEW_CUH

#include "matrix_view.cuh"
#include "device_matrix.cuh"

template <typename T>
class DeviceMatrixView : public MatrixView<T> {
public:
    // Type aliases
    using value_type = T;
    using size_type = std::size_t;
    using pointer = T*;
    using const_pointer = const T*;

    // Default constructor (empty view)
    DeviceMatrixView() noexcept = default;

    // Constructor from a DeviceMatrix (creates a view of the whole matrix)
    explicit DeviceMatrixView(DeviceMatrix<T>& matrix) noexcept
        : MatrixView<T>(matrix.data(), matrix.rows(), matrix.cols(), matrix.leading_dimension(), matrix.layout()) {}

    // Const overload for creating a view from a const DeviceMatrix
    explicit DeviceMatrixView(const DeviceMatrix<T>& matrix) noexcept
        : MatrixView<T>(const_cast<typename DeviceMatrix<T>::pointer>(matrix.data()), matrix.rows(), matrix.cols(), matrix.leading_dimension(), matrix.layout()) {}

    // Const overload for creating a view from a const DeviceMatrix
    explicit DeviceMatrixView(const DeviceMatrix<T>& matrix) noexcept
        : MatrixView<T>(matrix) {}
};

#endif // DEVICE_MATRIX_VIEW_CUH
