#ifndef QUI1_MATRIX_VIEW_BASE_CUH
#define QUI1_MATRIX_VIEW_BASE_CUH

#include "../qui1_matrix_base.cuh"

namespace qui1 {

template <typename T>
class MatrixViewBase : public MatrixBase<T> {
public:
    // 构造函数：视图不管理内存，直接使用外部数据
    MatrixViewBase(T* data, size_t rows, size_t cols, Layout layout, 
                  Location loc, size_t lda, size_t offset = 0)
        : data_(data + offset), rows_(rows), cols_(cols), 
          layout_(layout), location_(loc), lda_(lda), offset_(offset) {}

    // 禁止拷贝和移动
    MatrixViewBase(const MatrixViewBase&) = delete;
    MatrixViewBase& operator=(const MatrixViewBase&) = delete;
    MatrixViewBase(MatrixViewBase&&) = delete;
    MatrixViewBase& operator=(MatrixViewBase&&) = delete;

    // 实现基类纯虚函数
    auto getRows() const -> size_t override { return rows_; }
    auto getCols() const -> size_t override { return cols_; }
    auto getLayout() const -> Layout override { return layout_; }
    auto getLocation() const -> Location override { return location_; }
    
    auto getData() -> T* override { return data_; }
    auto getData() const -> const T* override { return data_; }
    
    auto hasData() const -> bool override { return data_ != nullptr; }

    // 新增视图特有方法
    auto getLDA() const -> size_t { return lda_; }
    auto getOffset() const -> size_t { return offset_; }

    // 覆盖基类的leading dimension计算
    size_t getLeadingDimension() const override {
        return lda_;
    }

private:
    T* data_;         // 数据指针（包含offset）
    size_t rows_;     // 视图行数
    size_t cols_;     // 视图列数
    Layout layout_;   // 内存布局
    Location location_; // 数据位置
    size_t lda_;      // 指定的leading dimension
    size_t offset_;   // 原始数据偏移量
};

} // namespace qui1

#endif // QUI1_MATRIX_VIEW_BASE_CUH
