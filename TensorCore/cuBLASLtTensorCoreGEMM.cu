#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cstdio>

void cublaslt_tensorcore_gemm(int N)
{
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

    cublasLtHandle_t handle;
    cublasLtCreate(&handle);

    float alpha = 1.0f;
    float beta  = 0.0f;

    cublasOperation_t op = CUBLAS_OP_N;

    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t aDesc, bDesc, cDesc;

    cublasLtMatmulDescCreate(&operationDesc, CUDA_R_32F);

    // 啟用 Tensor Core
    cublasLtMatmulDescSetAttribute(
        operationDesc,
        CUBLASLT_MATMUL_DESC_MATH_MODE,
        (void*)&CUBLAS_TF32_TENSOR_OP_MATH,
        sizeof(int));

    cublasLtMatrixLayoutCreate(&aDesc, CUDA_R_16F, n, n, n);
    cublasLtMatrixLayoutCreate(&bDesc, CUDA_R_16F, n, n, n);
    cublasLtMatrixLayoutCreate(&cDesc, CUDA_R_32F, n, n, n);

    cublasLtMatmul(handle,
                   operationDesc,
                   &alpha, A, aDesc,
                            B, bDesc,
                   &beta,  C, cDesc,
                           C, cDesc,
                   nullptr, 0, 0);

    cudaDeviceSynchronize();

    printf("C[0] = %f\n", C[0]);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cublasLtDestroy(handle);
}
