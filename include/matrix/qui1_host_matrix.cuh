#ifndef QUI1_HOST_MATRIX_CUH
#define QUI1_HOST_MATRIX_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>

#include "../common/error_check.cuh"
#include "fmt/format.h"
#include "matrix/view/qui1_host_matrix_view.cuh"
#include "qui1_matrix_base.cuh"

namespace qui1 {
template <typename T>
class HostMatrix : public MatrixBase<T> {
   public:
    // 默认构造函数，提供一个空的 HostMatrix 对象
    HostMatrix()
        : data_(nullptr), rows_(0), cols_(0), layout_(qui1::Layout::COLUMN_MAJOR) {}

    // 有参构造函数，默认构造一个列优先的 rows * cols 大小的矩阵
    HostMatrix(size_t rows, size_t cols,
               qui1::Layout layout = qui1::Layout::COLUMN_MAJOR) {
        if ((rows == 0 && cols != 0) || (rows != 0 && cols == 0)) {
            throw std::invalid_argument(fmt::format(
                "rows = {} and cols = {} must be both non-zero", rows, cols));
        } else {
            allocate(rows, cols, layout);
        }
    }

    // 虚析构函数，确保派生类正确释放资源
    ~HostMatrix() { clear(); }

    // getter
    size_t getRows() const override { return rows_; }
    size_t getCols() const override { return cols_; }
    size_t getLeadingDimension() const override {
        return layout_ == qui1::Layout::COLUMN_MAJOR ? rows_ : cols_;
    }
    qui1::Layout getLayout() const override { return layout_; }
    // 返回非 const 指针，允许修改数据
    T* getData() override { return data_; }
    // 返回 const 指针，用于 const 对象
    const T* getData() const override { return data_; }
    bool hasData() const override { return data_ != nullptr; }

    auto getView(size_t r, size_t c, size_t offset_m = 0, size_t offset_n = 0,
                 size_t ld = 0) -> HostMatrixView<T> {
        if (ld == 0) {
            ld = getLeadingDimension();
        }
        if (layout_ == Layout::COLUMN_MAJOR) {
            return HostMatrixView<T>(data_, r, c, layout_, ld,
                                    offset_n * ld + offset_m);
        }
        if (layout_ == Layout::ROW_MAJOR) {
            return HostMatrixView<T>(data_, r, c, layout_, ld,
                                    offset_m * ld + offset_n);
        }
        throw std::invalid_argument(fmt::format("get view failed"));
    }

    auto getLocation() const -> qui1::Location override {
        return qui1::Location::HOST;
    }

    void clear() {
        if (data_ != nullptr) {
            CUDA_CHECK(cudaFreeHost(data_));
            data_ = nullptr;
        }
        rows_ = 0;
        cols_ = 0;
    }

    // 删除拷贝构造函数，禁止拷贝
    HostMatrix(const HostMatrix&) = delete;
    // 删除拷贝赋值运算符，禁止拷贝
    HostMatrix& operator=(const HostMatrix&) = delete;

    // 移动构造函数
    HostMatrix(HostMatrix&& other) noexcept
        : data_(other.data_),
          rows_(other.rows_),
          cols_(other.cols_),
          layout_(other.layout_) {
        // 将 other 置于有效但空的状态，防止其析构函数释放资源
        other.data_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
    }

    // 移动赋值运算符
    HostMatrix& operator=(HostMatrix&& other) noexcept {
        if (this != &other) {
            // 释放当前对象的资源
            clear();
            // 窃取 other 的资源
            data_ = other.data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            layout_ = other.layout_;
            // 将 other 置于有效但空的状态
            other.data_ = nullptr;
            other.rows_ = 0;
            other.cols_ = 0;
        }
        return *this;
    }

    void resize(size_t rows, size_t cols, qui1::Layout layout) {
        if (rows == rows_ && cols == cols_ && layout == layout_) {
            return;
        }
        if ((rows == 0 && cols != 0) || (rows != 0 && cols == 0)) {
            throw std::invalid_argument(fmt::format(
                "rows = {} and cols = {} must be both non-zero", rows, cols));
        }
        if (data_) {
            clear();
        }
        allocate(rows, cols, layout);
    }

   private:
    T* data_;
    size_t rows_;
    size_t cols_;
    qui1::Layout layout_;

    void allocate(size_t r, size_t c, qui1::Layout layout) {
        rows_ = r;
        cols_ = c;
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data_),
                                  rows_ * cols_ * sizeof(T)));
        layout_ = layout;
    }
};
}  // namespace qui1
#endif
