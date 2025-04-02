#ifndef QUI1_MATRIX_BASE_CUH
#define QUI1_MATRIX_BASE_CUH
namespace qui1 {

enum class Layout { ROW_MAJOR, COLUMN_MAJOR };

enum class Location { HOST, DEVICE };

template <typename T>
class MatrixBase {
   public:
    // 虚析构函数，确保派生类正确释放资源
    virtual ~MatrixBase() = default;

    // 纯虚函数，获取矩阵的行数
    virtual auto getRows() const -> size_t = 0;
    // 纯虚函数，获取矩阵的列数
    virtual auto getCols() const -> size_t = 0;
    // 纯虚函数，获取矩阵的布局方式
    virtual auto getLayout() const -> Layout = 0;
    // 纯虚函数，获取矩阵数据的位置（主机或设备）
    virtual auto getLocation() const -> Location = 0;
    // 纯虚函数，获取矩阵数据指针
    virtual auto getData() -> T* = 0;
    virtual auto getData() const -> const T* = 0;
    // 纯虚函数，判断矩阵是否有数据
    virtual auto hasData() const -> bool = 0;

    virtual size_t getNumElements() const { return getRows() * getCols(); }
    virtual size_t getLeadingDimension() const {
        size_t rows = getRows();
        size_t cols = getCols();
        if (rows == 0 || cols == 0) return 0;
        return (getLayout() == Layout::ROW_MAJOR) ? cols : rows;
    }

    // 删除拷贝构造函数，禁止拷贝
    MatrixBase(const MatrixBase&) = delete;
    // 删除拷贝赋值运算符，禁止拷贝
    MatrixBase& operator=(const MatrixBase&) = delete;

    // 删除移动构造函数，禁止移动
    MatrixBase(MatrixBase&&) = delete;
    // 删除移动赋值运算符，禁止移动
    MatrixBase& operator=(MatrixBase&&) = delete;

   protected:
    MatrixBase() = default;
};
}  // namespace qui1

#endif
