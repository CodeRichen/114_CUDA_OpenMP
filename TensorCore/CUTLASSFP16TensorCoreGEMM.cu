#include "cutlass/gemm/device/gemm.h"
#include <cuda_fp16.h>
#include <cstdio>

void cutlass_tensorcore_gemm(int N)
{
    using Gemm = cutlass::gemm::device::Gemm<
        half, cutlass::layout::RowMajor,
        half, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor>;

    Gemm gemm_op;

    int n = N;
    int size = n * n;

    half *A, *B;
    float *C;

    cudaMallocManaged(&A, size * sizeof(half));
    cudaMallocManaged(&B, size * sizeof(half));
    cudaMallocManaged(&C, size * sizeof(float));

    for (int i = 0; i < size; i++) {
        A[i] = __float2half(1.0f);
        B[i] = __float2half(1.0f);
    }

    typename Gemm::Arguments args(
        {n, n, n},
        {A, n},
        {B, n},
        {C, n},
        {C, n},
        {1.0f, 0.0f});

    gemm_op(args);
    cudaDeviceSynchronize();

    printf("CUTLASS C[0] = %f\n", C[0]);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}
