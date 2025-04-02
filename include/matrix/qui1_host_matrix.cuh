#ifndef QUI1_HOST_MATRIX_CUH
#define QUI1_HOST_MATRIX_CUH

#include <cuda_runtime.h>

#include <optional>
#include <stdexcept>

#include "error_check.cuh"
#include "fmt/format.h"
#include "qui1_matrix_base.cuh"

template <typename T>
class HostMatrix : public MatrixBase<T> {
   public:
    // 默认构造函数，提供一个空的 HostMatrix 对象
    HostMatrix()
        : data_(nullptr), rows_(0), cols_(0), layout_(qui1::Layout::COLUMN_MAJOR) {}

    // 有参构造函数，默认构造一个列优先的 rows * cols 大小的矩阵
    HostMatrix(size_t rows, size_t cols,
               qui1::Layout layout = qui1::Layout::COLUMN_MAJOR)
        : data_(new T[rows * cols]), rows_(rows), cols_(cols), layout_(layout) {
        if ((rows == 0 && cols != 0) || (rows != 0 && cols == 0)) {
            throw std::invalid_argument(fmt::format(
                "rows = {} and cols = {} must be both non-zero", rows, cols));
        }
    }

    // 虚析构函数，确保派生类正确释放资源
    ~HostMatrix() { clear(); }

    // getter
    size_t getRows() const override { return rows_; }
    size_t getCols() const override { return cols_; }
    qui1::Layout getLayout() const override { return layout_; }
    T* getData() const override { return data_; }
    // 为了从 const 函数中获取数据，需要使用 const T*
    const T* getData() const override { return data_; }
    bool hasData() const override { return data_ != nullptr; }

    auto getLocation() const -> qui1::Location override {
        return qui1::Location::HOST;
    }

    void clear() {
        if (data_) {
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

    // 删除移动构造函数，禁止移动
    HostMatrix(HostMatrix&&) = delete;
    // 删除移动赋值运算符，禁止移动
    HostMatrix& operator=(HostMatrix&&) = delete;

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
        if (data_) {
            CUDA_CHECK(cudaFreeHost(data_));
        }
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data_),
                                  rows_ * cols_ * sizeof(T)));
        rows_ = r;
        cols_ = c;
        layout_ = layout;
    }
};

#endif
