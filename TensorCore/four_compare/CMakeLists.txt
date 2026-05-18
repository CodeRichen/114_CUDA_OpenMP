cmake_minimum_required(VERSION 3.18)
project(gemm_compare LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ⭐ 加這行可讓 CUTLASS 編譯更乾淨
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -diag-suppress=20013")

set(CMAKE_CUDA_ARCHITECTURES 87)

find_package(CUDAToolkit REQUIRED)

add_executable(gemm_compare gemm_compare.cu)

target_link_libraries(gemm_compare
    PRIVATE CUDA::cudart CUDA::cublas
)

if(EXISTS "$ENV{HOME}/cutlass/include")
    target_include_directories(gemm_compare PRIVATE
        $ENV{HOME}/cutlass/include
        $ENV{HOME}/cutlass/tools/util/include
    )
endif()
