cmake_minimum_required(VERSION 3.25)
project(Qui1framework VERSION 0.1.0 LANGUAGES C CXX CUDA)

# --- Standard Settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- Include Directory ---
# Add the project's include directory globally for convenience
include_directories(include)

# --- CUDA Architectures ---
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Default architectures if not specified by the user/environment
    set(CMAKE_CUDA_ARCHITECTURES 87 89 90) # Example: Ampere, Hopper. Adjust as needed.
endif()
message(STATUS "Targeting CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# --- Dependencies ---
# Add fmt library (assuming it's header-only or built by add_subdirectory)
add_subdirectory(external/fmt)

# Add GoogleTest using FetchContent
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip # Or specify a commit hash
)
# Make GoogleTest available (set QUIET to suppress output)
FetchContent_MakeAvailable(googletest)

# --- Testing Setup ---
enable_testing()

# --- Test Executables ---

# Example test (kept for reference, modify or remove as needed)
add_executable(test_fmt tests/test_fmt.cpp)
target_link_libraries(test_fmt PRIVATE fmt::fmt)
# If test_fmt uses gtest:
# target_link_libraries(test_fmt PRIVATE fmt::fmt GTest::gtest_main)
# add_test(NAME TestFmt COMMAND test_fmt)


# Test for MatrixHelper::fillWithRandom
# Note: Even though the .cpp file might not directly use CUDA API, it includes headers (.cuh) that do.
# Linking against CUDA::cudart ensures the necessary include paths and runtime libs are available.
add_executable(test_matrix_helper_fill_random tests/test_matrix_helper_fill_random.cpp)
# Explicitly add CUDA include directories for the C++ compiler
target_include_directories(test_matrix_helper_fill_random PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(test_matrix_helper_fill_random PRIVATE fmt::fmt GTest::gtest_main CUDA::cudart CUDA::curand)
add_test(NAME TestMatrixHelperFillRandom COMMAND test_matrix_helper_fill_random)

# Test for MatrixHelper::printMatrix
add_executable(test_matrix_helper_print tests/test_matrix_helper_print.cpp)
# Explicitly add CUDA include directories for the C++ compiler
target_include_directories(test_matrix_helper_print PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(test_matrix_helper_print PRIVATE fmt::fmt GTest::gtest_main CUDA::cudart CUDA::curand)
add_test(NAME TestMatrixHelperPrint COMMAND test_matrix_helper_print)

add_executable(test_gemm_wrapper tests/test_gemm_wrapper.cpp)
# Explicitly add CUDA include directories for the C++ compiler
target_include_directories(test_gemm_wrapper PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(test_gemm_wrapper PRIVATE fmt::fmt GTest::gtest_main CUDA::cudart CUDA::curand CUDA::cublas)
add_test(NAME TestGemmWrapper COMMAND test_gemm_wrapper)

# Test for NrmWrapper
add_executable(test_nrm_wrapper tests/test_nrm_wrapper.cpp)
# Explicitly add CUDA include directories for the C++ compiler
target_include_directories(test_nrm_wrapper PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(test_nrm_wrapper PRIVATE fmt::fmt GTest::gtest_main CUDA::cudart CUDA::curand CUDA::cublas)
add_test(NAME TestNrmWrapper COMMAND test_nrm_wrapper)


# --- CUDA Check (Optional, informational) ---
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found (Version: ${CUDAToolkit_VERSION})")
else()
    message(WARNING "CUDA Toolkit not found by find_package. Compilation might still work if nvcc is in PATH.")
endif()
