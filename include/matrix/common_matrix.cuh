#ifndef COMMON_MATRIX_CUH
#define COMMON_MATRIX_CUH

#include <cstddef> // For size_t

template <typename T>
class Matrix {
public:
    // Type aliases
    using value_type = T;
    using size_type = std::size_t;

    // Virtual destructor for proper cleanup in derived classes
    virtual ~Matrix() = default;

    // Pure virtual methods to be implemented by derived classes
    [[nodiscard]] virtual size_type rows() const noexcept = 0;
    [[nodiscard]] virtual size_type cols() const noexcept = 0;
    [[nodiscard]] virtual size_type size() const noexcept = 0;

    // Layout enum class
    enum class Layout { RowMajor, ColumnMajor };

    // New member variable for layout
    virtual Layout layout() const noexcept = 0;

    // Optional: Additional common interface methods can be added here
    // For example:
    // virtual void resize(size_type new_rows, size_type new_cols) = 0;
};

#endif // COMMON_MATRIX_CUH
