#include <cstdio>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void gemm_tensor_core(const half* A, const half* B, float* C, int N)
{
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 32;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    if (warpM * WMMA_M < N && warpN * WMMA_N < N)
    {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int k = 0; k < N; k += WMMA_K)
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

            const half* tileA = A + warpM * WMMA_M * N + k;
            const half* tileB = B + k * N + warpN * WMMA_N;

            wmma::load_matrix_sync(a_frag, tileA, N);
            wmma::load_matrix_sync(b_frag, tileB, N);

            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        float* tileC = C + warpM * WMMA_M * N + warpN * WMMA_N;
        wmma::store_matrix_sync(tileC, c_frag, N, wmma::mem_row_major);
    }
}

void gemm_tensor_core_run(int N)
{
    size_t bytes_f16 = N * N * sizeof(half);
    size_t bytes_f32 = N * N * sizeof(float);

    half *A, *B;
    float *C;

    cudaMallocManaged(&A, bytes_f16);
    cudaMallocManaged(&B, bytes_f16);
    cudaMallocManaged(&C, bytes_f32);

    for (int i = 0; i < N * N; i++) {
        A[i] = __float2half(1.0f);
        B[i] = __float2half(1.0f);
    }

    dim3 block(128);
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (N + WMMA_M - 1) / WMMA_M);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    gemm_tensor_core<<<grid, block>>>(A, B, C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("[TENSOR CORE] N=%d time: %.3f ms\n", N, ms);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}
