#include <cstdio>
#include <cuda_runtime.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

__global__ void gemm_cuda_core(const float* A, const float* B, float* C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[IDX2C(row, k, N)] * B[IDX2C(k, col, N)];
        }
        C[IDX2C(row, col, N)] = sum;
    }
}

void gemm_cuda_core_run(int N)
{
    size_t bytes = N * N * sizeof(float);

    float *A, *B, *C;
    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    gemm_cuda_core<<<grid, block>>>(A, B, C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("[CUDA CORE] N=%d time: %.3f ms\n", N, ms);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}
