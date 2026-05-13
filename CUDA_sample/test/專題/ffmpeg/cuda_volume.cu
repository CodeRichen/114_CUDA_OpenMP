#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>

__global__ void chroma_key_kernel(unsigned char* d_frame, unsigned char* d_bg, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int i = (y * width + x) * 3;
        unsigned char r = d_frame[i];
        unsigned char g = d_frame[i + 1];
        unsigned char b = d_frame[i + 2];

        // 綠幕去背景判定
        if (g > 100 && g > r * 1.1f && g > b * 1.1f) {
            d_frame[i]     = d_bg[i];
            d_frame[i + 1] = d_bg[i + 1];
            d_frame[i + 2] = d_bg[i + 2];
        }
    }
}

int main(int argc, char** argv) {
    // 從參數讀取解析度，若無則預設為 896x504
    int width = (argc > 1) ? atoi(argv[1]) : 896;
    int height = (argc > 2) ? atoi(argv[2]) : 504;
    const size_t frame_size = width * height * 3 * sizeof(unsigned char);

    unsigned char* h_frame = new unsigned char[width * height * 3];
    unsigned char* d_frame, *d_bg;
    cudaMalloc(&d_frame, frame_size);
    cudaMalloc(&d_bg, frame_size);
    cudaMemset(d_bg, 50, frame_size); // 預設背景設為深色

    // --- 執行緒資訊 ---
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    // 計算總執行緒數量
    long total_threads = (long)gridSize.x * gridSize.y * blockSize.x * blockSize.y;
    
    // 輸出到 stderr 避免干擾 stdout 的影片串流
    fprintf(stderr, "========================================\n");
    fprintf(stderr, "影像解析度: %d x %d\n", width, height);
    fprintf(stderr, "CUDA Block 設定: %d x %d\n", blockSize.x, blockSize.y);
    fprintf(stderr, "CUDA Grid  設定: %d x %d\n", gridSize.x, gridSize.y);
    fprintf(stderr, "每一幀啟動的總執行緒數: %ld 個\n", total_threads);
    fprintf(stderr, "========================================\n");

    int frame_count = 0;
    // 讀取 stdin
    while (fread(h_frame, 1, width * height * 3, stdin) > 0) {
        cudaMemcpy(d_frame, h_frame, frame_size, cudaMemcpyHostToDevice);

        chroma_key_kernel<<<gridSize, blockSize>>>(d_frame, d_bg, width, height);

        cudaMemcpy(h_frame, d_frame, frame_size, cudaMemcpyDeviceToHost);
        fwrite(h_frame, 1, width * height * 3, stdout);

        // --- 顯示進度 ---
        frame_count++;
        if (frame_count % 30 == 0) { // 每 30 幀顯示一次
            fprintf(stderr, "\r正在處理第 %d 幀...", frame_count);
            fflush(stderr);
        }
    }

    fprintf(stderr, "\n處理完成！總計幀數: %d\n", frame_count);

    cudaFree(d_frame); cudaFree(d_bg);
    delete[] h_frame;
    return 0;
}