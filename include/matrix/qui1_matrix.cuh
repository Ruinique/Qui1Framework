#ifndef QUI1_MATRIX_CUH
#define QUI1_MATRIX_CUH
namespace qui1 {
template <typename T>
class Matrix {
   public:
    // 存储布局
    enum class Layout { RowMajor, ColMajor };

   private:
    // 矩阵数据
    T* data_;
    // 矩阵尺寸
    int rows_;
    int cols_;
    // 矩阵存储布局
    Layout layout_;
};
}  // namespace qui1

#endif