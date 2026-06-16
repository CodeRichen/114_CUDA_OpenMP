#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <pthread.h>
#include <cuda_runtime.h>

#define NUM_THREADS 8   // pthread 執行緒數量（可依 CPU 核心數調整）

// ============================================================
//  CUDA — Sobel Kernel
//  輸出：magnitude 正規化至 [0,255] 的灰階影像
// ============================================================
__global__ void sobel_kernel_gpu(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < w - 1 && y > 0 && y < h - 1) {
        float dx = (-1*in[(y-1)*w+(x-1)]) + (1*in[(y-1)*w+(x+1)]) +
                   (-2*in[(y  )*w+(x-1)]) + (2*in[(y  )*w+(x+1)]) +
                   (-1*in[(y+1)*w+(x-1)]) + (1*in[(y+1)*w+(x+1)]);

        float dy = (-1*in[(y-1)*w+(x-1)]) + (-2*in[(y-1)*w+x]) + (-1*in[(y-1)*w+(x+1)]) +
                   ( 1*in[(y+1)*w+(x-1)]) + ( 2*in[(y+1)*w+x]) + ( 1*in[(y+1)*w+(x+1)]);

        float mag = sqrtf(dx*dx + dy*dy);
        // clamp 至 255
        out[y*w+x] = (unsigned char)(mag > 255.0f ? 255.0f : mag);
    } else if (x < w && y < h) {
        out[y*w+x] = 0;
    }
}

// ============================================================
//  CPU 單執行緒 — Sobel
// ============================================================
void cpu_sobel(const unsigned char* in, unsigned char* out, int w, int h) {
    memset(out, 0, w * h);
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            float dx = (-1*in[(y-1)*w+(x-1)]) + (1*in[(y-1)*w+(x+1)]) +
                       (-2*in[(y  )*w+(x-1)]) + (2*in[(y  )*w+(x+1)]) +
                       (-1*in[(y+1)*w+(x-1)]) + (1*in[(y+1)*w+(x+1)]);

            float dy = (-1*in[(y-1)*w+(x-1)]) + (-2*in[(y-1)*w+x]) + (-1*in[(y-1)*w+(x+1)]) +
                       ( 1*in[(y+1)*w+(x-1)]) + ( 2*in[(y+1)*w+x]) + ( 1*in[(y+1)*w+(x+1)]);

            float mag = sqrtf(dx*dx + dy*dy);
            out[y*w+x] = (unsigned char)(mag > 255.0f ? 255.0f : mag);
        }
    }
}

// ============================================================
//  pthread — Sobel（每條執行緒處理一段列範圍）
// ============================================================
struct ThreadArgs {
    const unsigned char* in;
    unsigned char*       out;
    int w, h;
    int y_start, y_end;   // 負責列範圍 [y_start, y_end)
};

void* pthread_sobel(void* arg) {
    ThreadArgs* a = (ThreadArgs*)arg;
    int w = a->w, h = a->h;
    for (int y = a->y_start; y < a->y_end; y++) {
        if (y == 0 || y == h - 1) {
            memset(a->out + y*w, 0, w);
            continue;
        }
        a->out[y*w]     = 0;
        a->out[y*w+w-1] = 0;
        for (int x = 1; x < w - 1; x++) {
            float dx = (-1*a->in[(y-1)*w+(x-1)]) + (1*a->in[(y-1)*w+(x+1)]) +
                       (-2*a->in[(y  )*w+(x-1)]) + (2*a->in[(y  )*w+(x+1)]) +
                       (-1*a->in[(y+1)*w+(x-1)]) + (1*a->in[(y+1)*w+(x+1)]);

            float dy = (-1*a->in[(y-1)*w+(x-1)]) + (-2*a->in[(y-1)*w+x]) + (-1*a->in[(y-1)*w+(x+1)]) +
                       ( 1*a->in[(y+1)*w+(x-1)]) + ( 2*a->in[(y+1)*w+x]) + ( 1*a->in[(y+1)*w+(x+1)]);

            float mag = sqrtf(dx*dx + dy*dy);
            a->out[y*w+x] = (unsigned char)(mag > 255.0f ? 255.0f : mag);
        }
    }
    return nullptr;
}

// ============================================================
//  MAIN
// ============================================================
int main() {
    int width, height, channels;
    unsigned char* h_in = stbi_load("input2.jpg", &width, &height, &channels, 1);
    if (!h_in) { printf("找不到 input2.jpg 檔案！\n"); return -1; }
    printf("圖片大小: %d x %d\n\n", width, height);

    size_t img_size = (size_t)width * height;

    // ──────────────────────────────────────────
    // ① GPU（CUDA）
    // ──────────────────────────────────────────
    {
        unsigned char *d_in, *d_out;
        cudaMalloc(&d_in,  img_size);
        cudaMalloc(&d_out, img_size);
        cudaMemcpy(d_in, h_in, img_size, cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);

        sobel_kernel_gpu<<<grid, block>>>(d_in, d_out, width, height);

        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms_gpu;
        cudaEventElapsedTime(&ms_gpu, t0, t1);
        printf("[CUDA GPU   ]  Sobel 處理時間: %.3f ms\n", ms_gpu);

        unsigned char* h_out = (unsigned char*)malloc(img_size);
        cudaMemcpy(h_out, d_out, img_size, cudaMemcpyDeviceToHost);
        stbi_write_jpg("output_gpu.jpg", width, height, 1, h_out, 90);
        free(h_out);

        cudaFree(d_in); cudaFree(d_out);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    // ──────────────────────────────────────────
    // ② CPU 單執行緒（for 迴圈）
    // ──────────────────────────────────────────
    {
        unsigned char* out = (unsigned char*)malloc(img_size);

        struct timespec ts, te;
        clock_gettime(CLOCK_MONOTONIC, &ts);

        cpu_sobel(h_in, out, width, height);

        clock_gettime(CLOCK_MONOTONIC, &te);
        double ms = (te.tv_sec - ts.tv_sec) * 1000.0
                  + (te.tv_nsec - ts.tv_nsec) / 1e6;
        printf("[CPU  單緒  ]  Sobel 處理時間: %.3f ms\n", ms);

        stbi_write_jpg("output_cpu.jpg", width, height, 1, out, 90);
        free(out);
    }

    // ──────────────────────────────────────────
    // ③ CPU 多執行緒（pthread）
    // ──────────────────────────────────────────
    {
        unsigned char* out = (unsigned char*)malloc(img_size);

        pthread_t  tids[NUM_THREADS];
        ThreadArgs args[NUM_THREADS];
        int rows_per = height / NUM_THREADS;
        for (int t = 0; t < NUM_THREADS; t++) {
            args[t].in      = h_in;
            args[t].out     = out;
            args[t].w       = width;
            args[t].h       = height;
            args[t].y_start = t * rows_per;
            args[t].y_end   = (t == NUM_THREADS - 1) ? height : args[t].y_start + rows_per;
        }

        struct timespec ts, te;
        clock_gettime(CLOCK_MONOTONIC, &ts);

        for (int t = 0; t < NUM_THREADS; t++)
            pthread_create(&tids[t], nullptr, pthread_sobel, &args[t]);
        for (int t = 0; t < NUM_THREADS; t++)
            pthread_join(tids[t], nullptr);

        clock_gettime(CLOCK_MONOTONIC, &te);
        double ms = (te.tv_sec - ts.tv_sec) * 1000.0
                  + (te.tv_nsec - ts.tv_nsec) / 1e6;
        printf("[pthread %2d緒]  Sobel 處理時間: %.3f ms\n", NUM_THREADS, ms);

        stbi_write_jpg("output_pthread.jpg", width, height, 1, out, 90);
        free(out);
    }

    printf("\n結果已分別存至:\n");
    printf("  output_gpu.jpg     (CUDA GPU)\n");
    printf("  output_cpu.jpg     (CPU 單執行緒)\n");
    printf("  output_pthread.jpg (pthread %d 執行緒)\n", NUM_THREADS);

    stbi_image_free(h_in);
    return 0;
}
