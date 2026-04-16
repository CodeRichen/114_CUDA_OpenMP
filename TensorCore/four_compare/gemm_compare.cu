#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <functional>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>            // WMMA
#include <cublas_v2.h>

// 如果有裝 CUTLASS，保留這行；如果還沒裝或不想用，先註解掉
#define USE_CUTLASS

#ifdef USE_CUTLASS
#include <cutlass/gemm/device/gemm.h>
#endif

using namespace nvcuda;

// ====================== 公用工具 ========================

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                    cudaGetErrorString(err));                                    \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

#define CHECK_CUBLAS(call)                                                       \
    do {                                                                         \
        cublasStatus_t status = call;                                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,      \
                    (int)status);                                                \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

// 簡單計時工具：回傳毫秒
float time_kernel_ms(std::function<void()> func, int repeat = 10) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 熱身一次
    func();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeat; ++i) {
        func();
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    ms /= repeat;  // 平均
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return ms;
}

// ====================== CUDA Core 版本 ========================

__global__ void gemm_cuda_core_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

float run_gemm_cuda_core(int N) {
    size_t bytes = N * N * sizeof(float);
    float *A, *B, *C;

    CHECK_CUDA(cudaMallocManaged(&A, bytes));
    CHECK_CUDA(cudaMallocManaged(&B, bytes));
    CHECK_CUDA(cudaMallocManaged(&C, bytes));

    for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0f;
        B[i] = 1.0f;
        C[i] = 0.0f;
    }

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    auto launcher = [&]() {
        gemm_cuda_core_kernel<<<grid, block>>>(A, B, C, N);
    };

    float ms = time_kernel_ms(launcher, 3);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("[CUDA Core]  N=%d  time = %.3f ms  C[0]=%f\n", N, ms, C[0]);

    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(B));
    CHECK_CUDA(cudaFree(C));
    return ms;
}

// ====================== WMMA Tensor Core 版本 ========================
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
__global__ void gemm_wmma_kernel(const half* A, const half* B, float* C, int N) {
#if __CUDA_ARCH__ < 700
    return; // 沒有 Tensor Core 的 GPU 不支援
#else
    int tileRow = blockIdx.y;
    int tileCol = blockIdx.x;

    if (tileRow * WMMA_M >= N || tileCol * WMMA_N >= N) return;

    // accumulator fragment
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int tileK = 0; tileK < N; tileK += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       half, wmma::row_major> b_frag;

        const half* tileA = A + (tileRow * WMMA_M * N + tileK);
        const half* tileB = B + (tileK * N + tileCol * WMMA_N);

        wmma::load_matrix_sync(a_frag, tileA, N);
        wmma::load_matrix_sync(b_frag, tileB, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* tileC = C + tileRow * WMMA_M * N + tileCol * WMMA_N;
    wmma::store_matrix_sync(tileC, c_frag, N, wmma::mem_row_major);
#endif
}

float run_gemm_wmma(int N) {
    if (N % 16 != 0) {
        printf("[WMMA] N 必須為 16 的倍數，略過 WMMA 測試\n");
        return NAN;
    }

    size_t bytes_half = N * N * sizeof(half);
    size_t bytes_float = N * N * sizeof(float);

    half* A;
    half* B;
    float* C;

    CHECK_CUDA(cudaMallocManaged(&A, bytes_half));
    CHECK_CUDA(cudaMallocManaged(&B, bytes_half));
    CHECK_CUDA(cudaMallocManaged(&C, bytes_float));

    for (int i = 0; i < N * N; ++i) {
        A[i] = __float2half(1.0f);
        B[i] = __float2half(1.0f);
        C[i] = 0.0f;
    }

    dim3 grid(N / WMMA_M, N / WMMA_N);
    dim3 block(32, 1);  // 一個 warp

    auto launcher = [&]() {
        gemm_wmma_kernel<<<grid, block>>>(A, B, C, N);
    };

    float ms = time_kernel_ms(launcher, 3);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("[WMMA]       N=%d  time = %.3f ms  C[0]=%f\n", N, ms, C[0]);

    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(B));
    CHECK_CUDA(cudaFree(C));
    return ms;
}

// ====================== cuBLAS (Tensor Core) 版本 ========================
// 使用 cublasGemmEx + CUBLAS_TENSOR_OP_MATH，在 Jetson 上會用 Tensor Core

float run_gemm_cublas_tc(int N) {
    size_t bytes_half = N * N * sizeof(half);
    size_t bytes_float = N * N * sizeof(float);

    half *A, *B;
    float *C;

    CHECK_CUDA(cudaMallocManaged(&A, bytes_half));
    CHECK_CUDA(cudaMallocManaged(&B, bytes_half));
    CHECK_CUDA(cudaMallocManaged(&C, bytes_float));

    for (int i = 0; i < N * N; ++i) {
        A[i] = __float2half(1.0f);
        B[i] = __float2half(1.0f);
        C[i] = 0.0f;
    }

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 啟用 Tensor Core
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    float alpha = 1.0f;
    float beta  = 0.0f;

    int lda = N;
    int ldb = N;
    int ldc = N;

    auto launcher = [&]() {
        CHECK_CUBLAS(
            cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N,
                &alpha,
                A, CUDA_R_16F, lda,
                B, CUDA_R_16F, ldb,
                &beta,
                C, CUDA_R_32F, ldc,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            )
        );
    };

    float ms = time_kernel_ms(launcher, 5);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("[cuBLAS TC]  N=%d  time = %.3f ms  C[0]=%f\n", N, ms, C[0]);

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(B));
    CHECK_CUDA(cudaFree(C));
    return ms;
}

// ====================== CUTLASS Tensor Core 版本 ========================
#ifdef USE_CUTLASS

float run_gemm_cutlass(int N) {
    using Gemm = cutlass::gemm::device::Gemm<
        half,  cutlass::layout::RowMajor,
        half,  cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor>;

    size_t bytes_half = N * N * sizeof(half);
    size_t bytes_float = N * N * sizeof(float);

    half *A, *B;
    float *C;

    CHECK_CUDA(cudaMallocManaged(&A, bytes_half));
    CHECK_CUDA(cudaMallocManaged(&B, bytes_half));
    CHECK_CUDA(cudaMallocManaged(&C, bytes_float));

    for (int i = 0; i < N * N; ++i) {
        A[i] = __float2half(1.0f);
        B[i] = __float2half(1.0f);
        C[i] = 0.0f;
    }

    Gemm gemm_op;

    Gemm::Arguments args(
        {N, N, N},           // problem size (M, N, K)
        {A, N},              // A pointer, lda
        {B, N},              // B pointer, ldb
        {C, N},              // C pointer, ldc
        {C, N},              // D pointer (output), ldd
        {1.0f, 0.0f}         // alpha, beta
    );

    auto launcher = [&]() {
        cutlass::Status status = gemm_op(args);
        if (status != cutlass::Status::kSuccess) {
            printf("CUTLASS GEMM failed\n");
        }
    };

    float ms = time_kernel_ms(launcher, 3);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("[CUTLASS]    N=%d  time = %.3f ms  C[0]=%f\n", N, ms, C[0]);

    CHECK_CUDA(cudaFree(A));
    CHECK_CUDA(cudaFree(B));
    CHECK_CUDA(cudaFree(C));
    return ms;
}
#endif // USE_CUTLASS

// ============================= main ==============================

int main(int argc, char** argv) {
    int N = 512;  // 預設矩陣大小
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }
    printf("GEMM size N = %d (N x N)\n", N);
    printf("注意：WMMA / CUTLASS 版本假設 N 為 16 的倍數\n\n");

    // CUDA Core
    float t_cuda = run_gemm_cuda_core(N);

    // WMMA
    float t_wmma = run_gemm_wmma(N);

    // cuBLAS Tensor Core
    float t_cublas_tc = run_gemm_cublas_tc(N);

#ifdef USE_CUTLASS
    // CUTLASS
    float t_cutlass = run_gemm_cutlass(N);
#endif

    printf("\n=== Summary (ms, smaller is faster) ===\n");
    printf("CUDA Core :   %.3f ms\n", t_cuda);
    printf("WMMA      :   %.3f ms\n", t_wmma);
    printf("cuBLAS TC :   %.3f ms\n", t_cublas_tc);
#ifdef USE_CUTLASS
    printf("CUTLASS   :   %.3f ms\n", t_cutlass);
#endif

    printf("\n=== Speedup (vs CUDA Core) ===\n");
    auto speedup = [&](float t) {
        return std::isnan(t) ? 0.0f : t_cuda / t;
    };
    printf("WMMA      :   %.2fx\n", speedup(t_wmma));
    printf("cuBLAS TC :   %.2fx\n", speedup(t_cublas_tc));
#ifdef USE_CUTLASS
    printf("CUTLASS   :   %.2fx\n", speedup(t_cutlass));
#endif

    return 0;
}

