#ifndef QUI1_DEVICE_MATRIX_VIEW_CUH
#define QUI1_DEVICE_MATRIX_VIEW_CUH

#include "qui1_matrix_view_base.cuh"

namespace qui1 {

template <typename T>
class DeviceMatrixView : public MatrixViewBase<T> {
public:
    DeviceMatrixView(T* data, size_t rows, size_t cols, Layout layout,
                   size_t lda, size_t offset = 0)
        : MatrixViewBase<T>(data, rows, cols, layout, Location::DEVICE, lda, offset) {}

    // 禁止拷贝和移动
    DeviceMatrixView(const DeviceMatrixView&) = delete;
    DeviceMatrixView& operator=(const DeviceMatrixView&) = delete;
    DeviceMatrixView(DeviceMatrixView&&) = delete;
    DeviceMatrixView& operator=(DeviceMatrixView&&) = delete;
};

} // namespace qui1

#endif // QUI1_DEVICE_MATRIX_VIEW_CUH
