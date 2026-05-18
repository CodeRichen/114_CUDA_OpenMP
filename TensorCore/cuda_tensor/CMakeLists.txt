cmake_minimum_required(VERSION 3.18)
project(gemm_compare LANGUAGES CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 87)    # Orin Nano = Ampere SM87
set(CMAKE_CUDA_STANDARD 14)

add_executable(gemm_compare
    main.cu
    gemm_cuda_core.cu
    gemm_tensor_core.cu
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(gemm_compare PRIVATE CUDA::cudart)
