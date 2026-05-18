#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// CUDA Kernel: 去除綠背景與白背景
// 影像格式為 RGBA (4 channels)
__global__ void remove_bg_kernel(unsigned char* img, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        int idx = (y * w + x) * 4;
        unsigned char r = img[idx];
        unsigned char g = img[idx + 1];
        unsigned char b = img[idx + 2];
        // unsigned char a = img[idx + 3];

        // 1. 綠色背景偵測 (G 遠大於 R, B)
        bool isGreen = (g > 100 && g > r * 1.4 && g > b * 1.4);
        
        // 2. 白色背景偵測 (RGB 皆高且數值相近)
        bool isWhite = (r > 200 && g > 200 && b > 200 && 
                        abs(r - g) < 20 && abs(r - b) < 20 && abs(g - b) < 20);

        if (isGreen || isWhite) {
            // 背景變為透明黑色 (Alpha = 0)
            img[idx] = 0;     // R
            img[idx + 1] = 0; // G
            img[idx + 2] = 0; // B
            img[idx + 3] = 0; // A
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("用法: %s <輸入影片>.mp4 <寬度> <高度>\n", argv[0]);
        printf("範例: %s input.mp4 1920 1080\n", argv[0]);
        return -1;
    }

    const char* input_file = argv[1];
    int width = atoi(argv[2]);
    int height = atoi(argv[3]);
    const char* output_file = "output.webm"; // WebM 支援透明背景 (VP9)

    size_t frame_size = width * height * 4 * sizeof(unsigned char);
    unsigned char *h_frame = (unsigned char*)malloc(frame_size);
    unsigned char *d_frame;

    cudaMalloc(&d_frame, frame_size);

    // 準備 FFmpeg 讀取指令 (解碼為 RGBA 原始資料)
    char cmd_in[512];
    sprintf(cmd_in, "ffmpeg -i %s -f image2pipe -vcodec rawvideo -pix_fmt rgba -", input_file);
    FILE *pipein = popen(cmd_in, "r");
    if (!pipein) {
        printf("無法開啟輸入影片管線。\n");
        return -1;
    }

    // 準備 FFmpeg 寫入指令 (編碼為 webm，支援透明度)
    char cmd_out[512];
    sprintf(cmd_out, "ffmpeg -y -f rawvideo -vcodec rawvideo -s %dx%d -pix_fmt rgba -r 30 -i - -c:v libvpx-vp9 -pix_fmt yuva420p -b:v 2M %s", width, height, output_file);
    FILE *pipeout = popen(cmd_out, "w");
    if (!pipeout) {
        printf("無法開啟輸出影片管線。\n");
        return -1;
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    int frame_count = 0;
    
    printf("開始處理影片，按 Ctrl+C 中斷...\n");

    // 不斷讀取 Frame 直到結束
    while (fread(h_frame, 1, frame_size, pipein) == frame_size) {
        // 丟上 GPU
        cudaMemcpy(d_frame, h_frame, frame_size, cudaMemcpyHostToDevice);

        // 執行 Kernel
        remove_bg_kernel<<<gridSize, blockSize>>>(d_frame, width, height);
        cudaDeviceSynchronize();

        // 拿回結果
        cudaMemcpy(h_frame, d_frame, frame_size, cudaMemcpyDeviceToHost);

        // 寫入輸出 FFmpeg
        fwrite(h_frame, 1, frame_size, pipeout);

        frame_count++;
        if (frame_count % 30 == 0) {
            printf("已處理 %d 幀...\n", frame_count);
        }
    }

    printf("處理完成，共 %d 幀。產生檔案: %s\n", frame_count, output_file);

    fflush(pipeout);
    pclose(pipein);
    pclose(pipeout);
    
    cudaFree(d_frame);
    free(h_frame);

    return 0;
}
