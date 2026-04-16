#include <stdio.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

__global__ void wmma_example(half *a, half *b, float *c) {
    // 定義 WMMA tile fragment
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half,  wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half,  wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // 將 accumulator 初始化為 0
    wmma::fill_fragment(c_frag, 0.0f);

    // 載入 A、B 進 fragment（Tensor Core 的資料 tile）
    wmma::load_matrix_sync(a_frag, a, 16);  
    wmma::load_matrix_sync(b_frag, b, 16);

    // Tensor Core 的矩陣乘運算
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 將結果存回全球記憶體
    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}

int main() {
    const int MATRIX_SIZE = 16 * 16;
    
    half *a, *b;
    float *c;

    cudaMallocManaged(&a, MATRIX_SIZE * sizeof(half));
    cudaMallocManaged(&b, MATRIX_SIZE * sizeof(half));
    cudaMallocManaged(&c, MATRIX_SIZE * sizeof(float));

    // 初始化 A, B
    for (int i = 0; i < MATRIX_SIZE; i++) {
        a[i] = __float2half(1.0f);
        b[i] = __float2half(1.0f);
    }

    // 啟動 kernel
    wmma_example<<<1, 32>>>(a, b, c);
    cudaDeviceSynchronize();

    // 印出部分結果
    printf("C[0] = %f\n", c[0]);
    printf("C[1] = %f\n", c[1]);
    printf("C[10] = %f\n", c[10]);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}
