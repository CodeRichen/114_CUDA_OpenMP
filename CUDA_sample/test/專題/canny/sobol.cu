#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel: 邊緣偵測
__global__ void sobel_kernel(const unsigned char* __restrict__ input,
                              unsigned char* output, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < w - 1 && y > 0 && y < h - 1) {
        // 用 __ldg() 走 read-only texture cache 載入 3x3 鄰域
        unsigned char p00 = __ldg(&input[(y-1)*w + (x-1)]);
        unsigned char p01 = __ldg(&input[(y-1)*w +  x   ]);
        unsigned char p02 = __ldg(&input[(y-1)*w + (x+1)]);
        unsigned char p10 = __ldg(&input[ y   *w + (x-1)]);
        unsigned char p12 = __ldg(&input[ y   *w + (x+1)]);
        unsigned char p20 = __ldg(&input[(y+1)*w + (x-1)]);
        unsigned char p21 = __ldg(&input[(y+1)*w +  x   ]);
        unsigned char p22 = __ldg(&input[(y+1)*w + (x+1)]);

        float dx = (-1.0f * p00) + (1.0f * p02)
                 + (-2.0f * p10) + (2.0f * p12)
                 + (-1.0f * p20) + (1.0f * p22);

        float dy = (-1.0f * p00) + (-2.0f * p01) + (-1.0f * p02)
                 + ( 1.0f * p20) + ( 2.0f * p21) + ( 1.0f * p22);

        float grad = sqrtf(dx*dx + dy*dy);
        output[y*w + x] = (grad > 255.0f) ? 255 : (unsigned char)grad;
    }
}

int main() {
    int width, height, channels;
    // 1. 讀取圖片 (強制轉為灰階 1 channel)
    unsigned char *h_in = stbi_load("input2.jpg", &width, &height, &channels, 1);
    if (!h_in) {
        printf("找不到 input.jpg 檔案！\n");
        return -1;
    }

    size_t size = width * height * sizeof(unsigned char);
    unsigned char *h_out = (unsigned char*)malloc(size);
    unsigned char *d_in, *d_out;

    // 2. 分配 GPU 記憶體並拷貝
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // 3. 配置 Thread 塊 (16x16)
    dim3 blockSize(16, 8);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 4. 執行與計時
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    sobel_kernel<<<gridSize, blockSize>>>(d_in, d_out, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU 處理時間: %f ms\n", ms);

    // 5. 拷貝回結果並存檔
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    stbi_write_jpg("output.jpg", width, height, 1, h_out, 0);

    printf("結果已存至 output.jpg\n");

    stbi_image_free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}