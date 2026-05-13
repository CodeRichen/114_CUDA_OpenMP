#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>

#define BATCH_SIZE 4 // 一次處理的幀數

// CUDA Kernel: 增加 z 軸維度來識別是哪一幀
__global__ void chroma_key_multi_kernel(unsigned char* d_frames, unsigned char* d_bg, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z; // 第幾幀 (0 ~ BATCH_SIZE-1)

    if (x < width && y < height) {
        // 計算該像素在整塊大記憶體中的位置
        long frame_offset = (long)z * width * height * 3;
        long pixel_idx = frame_offset + (y * width + x) * 3;

        unsigned char r = d_frames[pixel_idx];
        unsigned char g = d_frames[pixel_idx + 1];
        unsigned char b = d_frames[pixel_idx + 2];

        // 綠幕判定 (每個 Thread 處理不同幀的相同位置像素)
        if (g > 100 && g > r * 1.1f && g > b * 1.1f) {
            d_frames[pixel_idx]     = d_bg[(y * width + x) * 3];
            d_frames[pixel_idx + 1] = d_bg[(y * width + x) * 3 + 1];
            d_frames[pixel_idx + 2] = d_bg[(y * width + x) * 3 + 2];
        }
    }
}

int main(int argc, char** argv) {
    int width = (argc > 1) ? atoi(argv[1]) : 896;
    int height = (argc > 2) ? atoi(argv[2]) : 504;
    
    const size_t single_frame_bytes = width * height * 3 * sizeof(unsigned char);
    const size_t batch_bytes = single_frame_bytes * BATCH_SIZE;

    unsigned char* h_batch = new unsigned char[width * height * 3 * BATCH_SIZE];
    unsigned char* d_batch, *d_bg;

    cudaMalloc(&d_batch, batch_bytes);
    cudaMalloc(&d_bg, single_frame_bytes);

    // 初始化背景 (深藍色)
    unsigned char* h_bg = new unsigned char[width * height * 3];
    for(int i=0; i<width*height*3; i+=3) { h_bg[i]=0; h_bg[i+1]=30; h_bg[i+2]=60; }
    cudaMemcpy(d_bg, h_bg, single_frame_bytes, cudaMemcpyHostToDevice);

    // 設定三維 Grid: Z 軸大小為 BATCH_SIZE
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y, 
                  BATCH_SIZE);

    fprintf(stderr, "========================================\n");
    fprintf(stderr, "模式: 多幀並行 (Batch Size = %d)\n", BATCH_SIZE);
    fprintf(stderr, "啟動總執行緒: %ld\n", (long)gridSize.x * gridSize.y * BATCH_SIZE * 256);
    fprintf(stderr, "========================================\n");

    int total_frames = 0;
    // 一次讀取 BATCH_SIZE 份資料
    while (fread(h_batch, 1, batch_bytes, stdin) > 0) {
        // 1. 一次搬運 4 幀到 GPU (減少 PCIe 喚醒次數)
        cudaMemcpy(d_batch, h_batch, batch_bytes, cudaMemcpyHostToDevice);

        // 2. 執行 Kernel
        chroma_key_multi_kernel<<<gridSize, blockSize>>>(d_batch, d_bg, width, height);

        // 3. 一次搬運回 CPU
        cudaMemcpy(h_batch, d_batch, batch_bytes, cudaMemcpyDeviceToHost);
        
        fwrite(h_batch, 1, batch_bytes, stdout);
        total_frames += BATCH_SIZE;
        
        if (total_frames % 40 == 0) {
            fprintf(stderr, "\r已處理幀數: %d", total_frames);
            fflush(stderr);
        }
    }

    fprintf(stderr, "\n處理完成！\n");
    cudaFree(d_batch); cudaFree(d_bg);
    delete[] h_batch; delete[] h_bg;
    return 0;
}