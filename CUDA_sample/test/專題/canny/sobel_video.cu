#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// CUDA Kernel：邊緣偵測（漸進版）
// 只有「已經揭露」的 block 會執行 Sobel，其餘的 block 維持原圖
// 這樣每存一張圖，畫面就會多一塊區域變成邊緣偵測結果
__global__ void sobel_kernel_progressive(unsigned char* input, unsigned char* output,
                                          int w, int h, int revealedBlocks, int blocksPerRow) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int blockLinearIdx = by * blocksPerRow + bx; // grid 內的 block 順序（左到右、上到下）

    int x = bx * blockDim.x + threadIdx.x;
    int y = by * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    // 還沒輪到的 block：先顯示原圖
    if (blockLinearIdx > revealedBlocks) {
        output[y * w + x] = input[y * w + x];
        return;
    }

    // 已經輪到的 block：執行 Sobel 計算
    if (x > 0 && x < w - 1 && y > 0 && y < h - 1) {
        float dx = (-1 * input[(y-1)*w + (x-1)]) + (1 * input[(y-1)*w + (x+1)]) +
                   (-2 * input[(y)*w   + (x-1)]) + (2 * input[(y)*w   + (x+1)]) +
                   (-1 * input[(y+1)*w + (x-1)]) + (1 * input[(y+1)*w + (x+1)]);

        float dy = (-1 * input[(y-1)*w + (x-1)]) + (-2 * input[(y-1)*w + x]) + (-1 * input[(y-1)*w + (x+1)]) +
                   ( 1 * input[(y+1)*w + (x-1)]) + ( 2 * input[(y+1)*w + x]) + ( 1 * input[(y+1)*w + (x+1)]);

        float grad = sqrtf(dx*dx + dy*dy);
        output[y*w + x] = (grad > 255) ? 255 : (unsigned char)grad;
    } else {
        // 邊界像素直接複製原圖（避免讀越界，也避免未初始化垃圾值）
        output[y * w + x] = input[y * w + x];
    }
}

int main(int argc, char** argv) {
    // 用法： ./sobel_video [輸入jpg] [輸出mp4]
    // 沒給參數的話預設 input.jpg / output.mp4
    const char* inputJpg    = (argc > 1) ? argv[1] : "input.jpg";
    const char* outputVideo = (argc > 2) ? argv[2] : "output.mp4";

    int width, height, channels;
    unsigned char *h_in = stbi_load(inputJpg, &width, &height, &channels, 1);
    if (!h_in) {
        printf("找不到 %s 檔案！\n", inputJpg);
        return -1;
    }

    size_t size = (size_t)width * height * sizeof(unsigned char);
    unsigned char *h_out = (unsigned char*)malloc(size);
    unsigned char *d_in, *d_out;

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    int totalBlocks = gridSize.x * gridSize.y;

    // ---- 影片參數，可自行調整 ----
    const int NUM_FRAMES  = 90;  // 揭露過程的張數
    const int HOLD_FRAMES = 30;  // 結尾停留在完成畫面的張數（避免影片戛然而止）
    const int FPS         = 30;  // 輸出影片的 fps
    int step = (totalBlocks + NUM_FRAMES - 1) / NUM_FRAMES;
    if (step < 1) step = 1;

    system("rm -rf frames");
    system("mkdir -p frames");

    int frameIdx = 0;
    char filename[256];

    for (int revealed = 0; revealed <= totalBlocks; revealed += step) {
        sobel_kernel_progressive<<<gridSize, blockSize>>>(d_in, d_out, width, height, revealed, gridSize.x);
        cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

        snprintf(filename, sizeof(filename), "frames/frame_%05d.jpg", frameIdx++);
        stbi_write_jpg(filename, width, height, 1, h_out, 90);
    }

    // 確保最後一張一定是「完全處理完」的畫面
    sobel_kernel_progressive<<<gridSize, blockSize>>>(d_in, d_out, width, height, totalBlocks, gridSize.x);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // 結尾多停留幾張，讓影片不會突然結束
    for (int i = 0; i < HOLD_FRAMES; i++) {
        snprintf(filename, sizeof(filename), "frames/frame_%05d.jpg", frameIdx++);
        stbi_write_jpg(filename, width, height, 1, h_out, 90);
    }

    printf("共輸出 %d 張影格，開始用 ffmpeg 合成影片...\n", frameIdx);

    // ---- 自動呼叫 ffmpeg 合成影片，不用再手動貼指令 ----
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -y -hide_banner -loglevel error -framerate %d -i frames/frame_%%05d.jpg "
             "-c:v libx264 -pix_fmt yuv420p \"%s\"",
             FPS, outputVideo);
    system(cmd);

    printf("完成！輸出影片：%s\n", outputVideo);

    free(h_out);
    stbi_image_free(h_in);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
