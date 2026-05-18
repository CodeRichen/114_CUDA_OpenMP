#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <cuda_runtime.h>

// --- 1. 計算 Sobel 梯度與方向 ---
__global__ void sobel_kernel(const unsigned char* input, float* mag, unsigned char* dir, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < w - 1 && y > 0 && y < h - 1) {
        float dx = (-1 * input[(y-1)*w + (x-1)]) + (1 * input[(y-1)*w + (x+1)]) +
                   (-2 * input[(y)*w   + (x-1)]) + (2 * input[(y)*w   + (x+1)]) +
                   (-1 * input[(y+1)*w + (x-1)]) + (1 * input[(y+1)*w + (x+1)]);
        
        float dy = (-1 * input[(y-1)*w + (x-1)]) + (-2 * input[(y-1)*w + x]) + (-1 * input[(y-1)*w + (x+1)]) +
                   ( 1 * input[(y+1)*w + (x-1)]) + ( 2 * input[(y+1)*w + x]) + ( 1 * input[(y+1)*w + (x+1)]);
        
        float magnitude = sqrtf(dx*dx + dy*dy);
        mag[y*w + x] = magnitude;

        // 計算梯度方向，並分為 0, 45, 90, 135 度 (將角度離散化) 
        float angle = atan2f(dy, dx) * 180.0f / 3.14159265f;
        if (angle < 0) angle += 180.0f;

        unsigned char d = 0;
        if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180.0)) d = 0;
        else if (angle >= 22.5 && angle < 67.5) d = 45;
        else if (angle >= 67.5 && angle < 112.5) d = 90;
        else if (angle >= 112.5 && angle < 157.5) d = 135;

        dir[y*w + x] = d;
    } else if (x < w && y < h) {
        mag[y*w + x] = 0;
        dir[y*w + x] = 0;
    }
}

// --- 2. 非極大值抑制 (NMS) 與雙閾值 (Double Threshold) ---
__global__ void nms_threshold_kernel(const float* mag, const unsigned char* dir, unsigned char* output, int w, int h, float lowThr, float highThr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < w - 1 && y > 0 && y < h - 1) {
        float m = mag[y*w + x];
        unsigned char d = dir[y*w + x];
        float m1 = 0, m2 = 0;

        // 照著梯度方向看相鄰像素
        if (d == 0) { m1 = mag[y*w + (x-1)]; m2 = mag[y*w + (x+1)]; }
        else if (d == 45) { m1 = mag[(y+1)*w + (x-1)]; m2 = mag[(y-1)*w + (x+1)]; }
        else if (d == 90) { m1 = mag[(y-1)*w + x]; m2 = mag[(y+1)*w + x]; }
        else if (d == 135) { m1 = mag[(y-1)*w + (x-1)]; m2 = mag[(y+1)*w + (x+1)]; }

        // NMS: 如果自己最大，則保留做閾值判斷；否則視為 0
        if (m >= m1 && m >= m2) {
            if (m >= highThr) output[y*w + x] = 255;      // 強邊緣
            else if (m >= lowThr) output[y*w + x] = 128;  // 弱邊緣
            else output[y*w + x] = 0;
        } else {
            output[y*w + x] = 0;
        }
    } else if (x < w && y < h) {
        output[y*w + x] = 0;
    }
}

// --- 3. 邊緣連接 (Hysteresis) ---
// 將跟「強邊緣 (255)」相連的「弱邊緣 (128)」升級為強邊緣
__global__ void hysteresis_kernel(unsigned char* edges, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < w - 1 && y > 0 && y < h - 1) {
        if (edges[y*w + x] == 128) {
            bool connected = false;
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    if (edges[(y+i)*w + (x+j)] == 255) {
                        connected = true;
                    }
                }
            }
            if (connected) {
                edges[y*w + x] = 255;
            }
        }
    }
}

// --- 4. 清除剩下的未連結弱邊緣 ---
__global__ void cleanup_kernel(unsigned char* edges, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        if (edges[y*w + x] == 128) {
            edges[y*w + x] = 0;
        }
    }
}

int main() {
    int width, height, channels;
    // 1. 讀取圖片 (強制轉為灰階 1 channel)
    unsigned char *h_in = stbi_load("input2.jpg", &width, &height, &channels, 1);
    if (!h_in) {
        printf("找不到 input2.jpg 檔案！\n");
        return -1;
    }

    size_t img_size = width * height * sizeof(unsigned char);
    size_t float_size = width * height * sizeof(float);
    
    unsigned char *h_out = (unsigned char*)malloc(img_size);
    unsigned char *d_in, *d_out, *d_dir;
    float *d_mag;

    // 2. 分配 GPU 記憶體並拷貝
    cudaMalloc(&d_in, img_size);
    cudaMalloc(&d_out, img_size);
    cudaMalloc(&d_dir, img_size);
    cudaMalloc(&d_mag, float_size);
    
    cudaMemcpy(d_in, h_in, img_size, cudaMemcpyHostToDevice);

    // 3. 配置 Thread 塊 (16x16)
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 4. 執行與計時
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    // CUDA 執行 Canny 各個階段：
    
    // (1) 計算梯度與方向
    sobel_kernel<<<gridSize, blockSize>>>(d_in, d_mag, d_dir, width, height);
    cudaDeviceSynchronize();

    // (2) 非極大值抑制 (NMS) 與雙閾值 (Low=50, High=100，可依需求調整)
    nms_threshold_kernel<<<gridSize, blockSize>>>(d_mag, d_dir, d_out, width, height, 50.0f, 100.0f);
    cudaDeviceSynchronize();

    // (3) Hysteresis 連接邊緣 (跑 5 次迭代以確保長邊緣能傳遞)
    for (int i = 0; i < 5; i++) {
        hysteresis_kernel<<<gridSize, blockSize>>>(d_out, width, height);
        cudaDeviceSynchronize();
    }

    // (4) 清除孤立弱邊緣
    cleanup_kernel<<<gridSize, blockSize>>>(d_out, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU 處理時間: %f ms\n", ms);

    // 5. 拷貝回結果並存檔
    cudaMemcpy(h_out, d_out, img_size, cudaMemcpyDeviceToHost);
    stbi_write_jpg("output.jpg", width, height, 1, h_out, 0);

    printf("結果已存至 output.jpg\n");

    stbi_image_free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_dir);
    cudaFree(d_mag);
    return 0;
}