#ifndef QUI1_HOST_MATRIX_VIEW_CUH
#define QUI1_HOST_MATRIX_VIEW_CUH

#include "qui1_matrix_view_base.cuh"

namespace qui1 {

template <typename T>
class HostMatrixView : public MatrixViewBase<T> {
public:
    HostMatrixView(T* data, size_t rows, size_t cols, Layout layout, 
                 size_t lda, size_t offset = 0)
        : MatrixViewBase<T>(data, rows, cols, layout, Location::HOST, lda, offset) {}
    
    // 禁止拷贝和移动
    HostMatrixView(const HostMatrixView&) = delete;
    HostMatrixView& operator=(const HostMatrixView&) = delete;
    HostMatrixView(HostMatrixView&&) = delete;
    HostMatrixView& operator=(HostMatrixView&&) = delete;
};

} // namespace qui1

#endif // QUI1_HOST_MATRIX_VIEW_CUH
