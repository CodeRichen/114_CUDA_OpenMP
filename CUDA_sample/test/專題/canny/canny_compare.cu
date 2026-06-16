#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <pthread.h>
#include <cuda_runtime.h>

// ============================================================
//  共用常數
// ============================================================
#define LOW_THR  50.0f
#define HIGH_THR 100.0f
#define HYST_ITER 5
#define NUM_THREADS 8   // pthread 執行緒數量（可依 CPU 核心數調整）


//  CUDA Kernels
// ============================================================

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

        float angle = atan2f(dy, dx) * 180.0f / 3.14159265f;
        if (angle < 0) angle += 180.0f;

        unsigned char d = 0;
        if      ((angle >= 0 && angle < 22.5f) || (angle >= 157.5f && angle <= 180.0f)) d = 0;
        else if (angle >= 22.5f  && angle < 67.5f)  d = 45;
        else if (angle >= 67.5f  && angle < 112.5f) d = 90;
        else if (angle >= 112.5f && angle < 157.5f) d = 135;

        dir[y*w + x] = d;
    } else if (x < w && y < h) {
        mag[y*w + x] = 0;
        dir[y*w + x] = 0;
    }
}

__global__ void nms_threshold_kernel(const float* mag, const unsigned char* dir,
                                     unsigned char* output, int w, int h,
                                     float lowThr, float highThr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < w - 1 && y > 0 && y < h - 1) {
        float m = mag[y*w + x];
        unsigned char d = dir[y*w + x];
        float m1 = 0, m2 = 0;

        if      (d == 0)   { m1 = mag[y*w + (x-1)];         m2 = mag[y*w + (x+1)]; }
        else if (d == 45)  { m1 = mag[(y+1)*w + (x-1)];     m2 = mag[(y-1)*w + (x+1)]; }
        else if (d == 90)  { m1 = mag[(y-1)*w + x];         m2 = mag[(y+1)*w + x]; }
        else if (d == 135) { m1 = mag[(y-1)*w + (x-1)];     m2 = mag[(y+1)*w + (x+1)]; }

        if (m >= m1 && m >= m2) {
            if      (m >= highThr) output[y*w + x] = 255;
            else if (m >= lowThr)  output[y*w + x] = 128;
            else                   output[y*w + x] = 0;
        } else {
            output[y*w + x] = 0;
        }
    } else if (x < w && y < h) {
        output[y*w + x] = 0;
    }
}

__global__ void hysteresis_kernel(unsigned char* edges, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < w - 1 && y > 0 && y < h - 1) {
        if (edges[y*w + x] == 128) {
            bool connected = false;
            for (int i = -1; i <= 1; i++)
                for (int j = -1; j <= 1; j++)
                    if (edges[(y+i)*w + (x+j)] == 255)
                        connected = true;
            if (connected) edges[y*w + x] = 255;
        }
    }
}

__global__ void cleanup_kernel(unsigned char* edges, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h && edges[y*w + x] == 128)
        edges[y*w + x] = 0;
}

// ============================================================
//   ██████╗██████╗ ██╗   ██╗    ███████╗███████╗██████╗ ██╗ █████╗ ██╗
//  ██╔════╝██╔══██╗██║   ██║    ██╔════╝██╔════╝██╔══██╗██║██╔══██╗██║
//  ██║     ██████╔╝██║   ██║    ███████╗█████╗  ██████╔╝██║███████║██║
//  ██║     ██╔═══╝ ██║   ██║    ╚════██║██╔══╝  ██╔══██╗██║██╔══██║██║
//  ╚██████╗██║     ╚██████╔╝    ███████║███████╗██║  ██║██║██║  ██║███████╗
//   ╚═════╝╚═╝      ╚═════╝     ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚══════╝
//  純 CPU（單執行緒 for 迴圈）實作
// ============================================================

static inline unsigned char quantize_angle(float angle) {
    if (angle < 0) angle += 180.0f;
    if      ((angle >= 0 && angle < 22.5f) || (angle >= 157.5f)) return 0;
    else if (angle >= 22.5f  && angle < 67.5f)  return 45;
    else if (angle >= 67.5f  && angle < 112.5f) return 90;
    else                                          return 135;
}

void cpu_sobel(const unsigned char* in, float* mag, unsigned char* dir, int w, int h) {
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            float dx = (-1*in[(y-1)*w+(x-1)]) + (1*in[(y-1)*w+(x+1)]) +
                       (-2*in[(y)*w+(x-1)])   + (2*in[(y)*w+(x+1)])   +
                       (-1*in[(y+1)*w+(x-1)]) + (1*in[(y+1)*w+(x+1)]);

            float dy = (-1*in[(y-1)*w+(x-1)]) + (-2*in[(y-1)*w+x]) + (-1*in[(y-1)*w+(x+1)]) +
                       ( 1*in[(y+1)*w+(x-1)]) + ( 2*in[(y+1)*w+x]) + ( 1*in[(y+1)*w+(x+1)]);

            mag[y*w+x] = sqrtf(dx*dx + dy*dy);
            dir[y*w+x] = quantize_angle(atan2f(dy, dx) * 180.0f / 3.14159265f);
        }
    }
}

void cpu_nms_threshold(const float* mag, const unsigned char* dir,
                       unsigned char* out, int w, int h) {
    memset(out, 0, w * h);
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            float m = mag[y*w+x];
            unsigned char d = dir[y*w+x];
            float m1 = 0, m2 = 0;

            if      (d == 0)   { m1 = mag[y*w+(x-1)];     m2 = mag[y*w+(x+1)]; }
            else if (d == 45)  { m1 = mag[(y+1)*w+(x-1)]; m2 = mag[(y-1)*w+(x+1)]; }
            else if (d == 90)  { m1 = mag[(y-1)*w+x];     m2 = mag[(y+1)*w+x]; }
            else if (d == 135) { m1 = mag[(y-1)*w+(x-1)]; m2 = mag[(y+1)*w+(x+1)]; }

            if (m >= m1 && m >= m2) {
                if      (m >= HIGH_THR) out[y*w+x] = 255;
                else if (m >= LOW_THR)  out[y*w+x] = 128;
                else                    out[y*w+x] = 0;
            }
        }
    }
}

void cpu_hysteresis(unsigned char* edges, int w, int h) {
    for (int iter = 0; iter < HYST_ITER; iter++) {
        for (int y = 1; y < h - 1; y++) {
            for (int x = 1; x < w - 1; x++) {
                if (edges[y*w+x] == 128) {
                    bool conn = false;
                    for (int i = -1; i <= 1 && !conn; i++)
                        for (int j = -1; j <= 1 && !conn; j++)
                            if (edges[(y+i)*w+(x+j)] == 255) conn = true;
                    if (conn) edges[y*w+x] = 255;
                }
            }
        }
    }
}

void cpu_cleanup(unsigned char* edges, int w, int h) {
    for (int i = 0; i < w * h; i++)
        if (edges[i] == 128) edges[i] = 0;
}

// ============================================================
//  ██████╗ ████████╗██╗  ██╗██████╗ ███████╗ █████╗ ██████╗
//  ██╔══██╗╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗██╔══██╗
//  ██████╔╝   ██║   ███████║██████╔╝█████╗  ███████║██║  ██║
//  ██╔═══╝    ██║   ██╔══██║██╔══██╗██╔══╝  ██╔══██║██║  ██║
//  ██║        ██║   ██║  ██║██║  ██║███████╗██║  ██║██████╔╝
//  ╚═╝        ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═════╝
//  pthread 多執行緒實作
// ============================================================

struct ThreadArgs {
    const unsigned char* in;
    const unsigned char* dir;
    const float*         mag;
    float*               mag_out;
    unsigned char*       dir_out;
    unsigned char*       edges;
    int w, h;
    int y_start, y_end;   // 負責的列範圍 [y_start, y_end)
};

void* pthread_sobel(void* arg) {
    ThreadArgs* a = (ThreadArgs*)arg;
    int w = a->w, h = a->h;
    for (int y = a->y_start; y < a->y_end; y++) {
        if (y == 0 || y == h - 1) {
            for (int x = 0; x < w; x++) { a->mag_out[y*w+x] = 0; a->dir_out[y*w+x] = 0; }
            continue;
        }
        for (int x = 0; x < w; x++) {
            if (x == 0 || x == w - 1) { a->mag_out[y*w+x] = 0; a->dir_out[y*w+x] = 0; continue; }
            float dx = (-1*a->in[(y-1)*w+(x-1)]) + (1*a->in[(y-1)*w+(x+1)]) +
                       (-2*a->in[(y)*w+(x-1)])   + (2*a->in[(y)*w+(x+1)])   +
                       (-1*a->in[(y+1)*w+(x-1)]) + (1*a->in[(y+1)*w+(x+1)]);
            float dy = (-1*a->in[(y-1)*w+(x-1)]) + (-2*a->in[(y-1)*w+x]) + (-1*a->in[(y-1)*w+(x+1)]) +
                       ( 1*a->in[(y+1)*w+(x-1)]) + ( 2*a->in[(y+1)*w+x]) + ( 1*a->in[(y+1)*w+(x+1)]);
            a->mag_out[y*w+x] = sqrtf(dx*dx + dy*dy);
            a->dir_out[y*w+x] = quantize_angle(atan2f(dy, dx) * 180.0f / 3.14159265f);
        }
    }
    return nullptr;
}

void* pthread_nms(void* arg) {
    ThreadArgs* a = (ThreadArgs*)arg;
    int w = a->w, h = a->h;
    for (int y = a->y_start; y < a->y_end; y++) {
        if (y == 0 || y == h - 1) {
            for (int x = 0; x < w; x++) a->edges[y*w+x] = 0;
            continue;
        }
        for (int x = 0; x < w; x++) {
            if (x == 0 || x == w - 1) { a->edges[y*w+x] = 0; continue; }
            float m = a->mag[y*w+x];
            unsigned char d = a->dir[y*w+x];
            float m1 = 0, m2 = 0;
            if      (d == 0)   { m1 = a->mag[y*w+(x-1)];     m2 = a->mag[y*w+(x+1)]; }
            else if (d == 45)  { m1 = a->mag[(y+1)*w+(x-1)]; m2 = a->mag[(y-1)*w+(x+1)]; }
            else if (d == 90)  { m1 = a->mag[(y-1)*w+x];     m2 = a->mag[(y+1)*w+x]; }
            else if (d == 135) { m1 = a->mag[(y-1)*w+(x-1)]; m2 = a->mag[(y+1)*w+(x+1)]; }

            if (m >= m1 && m >= m2) {
                if      (m >= HIGH_THR) a->edges[y*w+x] = 255;
                else if (m >= LOW_THR)  a->edges[y*w+x] = 128;
                else                    a->edges[y*w+x] = 0;
            } else {
                a->edges[y*w+x] = 0;
            }
        }
    }
    return nullptr;
}

void* pthread_hyst(void* arg) {
    ThreadArgs* a = (ThreadArgs*)arg;
    int w = a->w, h = a->h;
    for (int y = a->y_start; y < a->y_end; y++) {
        if (y == 0 || y == h - 1) continue;
        for (int x = 1; x < w - 1; x++) {
            if (a->edges[y*w+x] == 128) {
                bool conn = false;
                for (int i = -1; i <= 1 && !conn; i++)
                    for (int j = -1; j <= 1 && !conn; j++)
                        if (a->edges[(y+i)*w+(x+j)] == 255) conn = true;
                if (conn) a->edges[y*w+x] = 255;
            }
        }
    }
    return nullptr;
}

// 用來分配行範圍給各執行緒的輔助函式
void dispatch_threads(pthread_t* tids, ThreadArgs* args, int n, int h,
                      void* (*fn)(void*)) {
    int rows_per = h / n;
    for (int t = 0; t < n; t++) {
        args[t].y_start = t * rows_per;
        args[t].y_end   = (t == n - 1) ? h : args[t].y_start + rows_per;
        pthread_create(&tids[t], nullptr, fn, &args[t]);
    }
    for (int t = 0; t < n; t++)
        pthread_join(tids[t], nullptr);
}

// ============================================================
//  MAIN
// ============================================================
int main() {
    int width, height, channels;
    unsigned char* h_in = stbi_load("input2.jpg", &width, &height, &channels, 1);
    if (!h_in) {
        printf("找不到 input2.jpg 檔案！\n");
        return -1;
    }
    printf("圖片大小: %d x %d\n\n", width, height);

    size_t img_size   = (size_t)width * height * sizeof(unsigned char);
    size_t float_size = (size_t)width * height * sizeof(float);

    // ──────────────────────────────────────────
    // ① GPU（CUDA）
    // ──────────────────────────────────────────
    {
        unsigned char *d_in, *d_out, *d_dir;
        float *d_mag;
        cudaMalloc(&d_in,  img_size);
        cudaMalloc(&d_out, img_size);
        cudaMalloc(&d_dir, img_size);
        cudaMalloc(&d_mag, float_size);
        cudaMemcpy(d_in, h_in, img_size, cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((width + 15) / 16, (height + 15) / 16);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);

        sobel_kernel<<<grid, block>>>(d_in, d_mag, d_dir, width, height);
        cudaDeviceSynchronize();

        nms_threshold_kernel<<<grid, block>>>(d_mag, d_dir, d_out, width, height, LOW_THR, HIGH_THR);
        cudaDeviceSynchronize();

        for (int i = 0; i < HYST_ITER; i++) {
            hysteresis_kernel<<<grid, block>>>(d_out, width, height);
            cudaDeviceSynchronize();
        }

        cleanup_kernel<<<grid, block>>>(d_out, width, height);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float ms_gpu;
        cudaEventElapsedTime(&ms_gpu, t0, t1);
        printf("[CUDA GPU ]  處理時間: %.3f ms\n", ms_gpu);

        unsigned char* h_gpu_out = (unsigned char*)malloc(img_size);
        cudaMemcpy(h_gpu_out, d_out, img_size, cudaMemcpyDeviceToHost);
        stbi_write_jpg("output_gpu.jpg", width, height, 1, h_gpu_out, 90);
        free(h_gpu_out);

        cudaFree(d_in); cudaFree(d_out); cudaFree(d_dir); cudaFree(d_mag);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    // ──────────────────────────────────────────
    // ② CPU 單執行緒（for 迴圈）
    // ──────────────────────────────────────────
    {
        float*         mag = (float*)malloc(float_size);
        unsigned char* dir = (unsigned char*)malloc(img_size);
        unsigned char* out = (unsigned char*)calloc(width * height, 1);

        struct timespec ts, te;
        clock_gettime(CLOCK_MONOTONIC, &ts);

        cpu_sobel(h_in, mag, dir, width, height);
        cpu_nms_threshold(mag, dir, out, width, height);
        cpu_hysteresis(out, width, height);
        cpu_cleanup(out, width, height);

        clock_gettime(CLOCK_MONOTONIC, &te);
        double ms_cpu = (te.tv_sec - ts.tv_sec) * 1000.0
                      + (te.tv_nsec - ts.tv_nsec) / 1e6;
        printf("[CPU  單緒]  處理時間: %.3f ms\n", ms_cpu);

        stbi_write_jpg("output_cpu.jpg", width, height, 1, out, 90);
        free(mag); free(dir); free(out);
    }

    // ──────────────────────────────────────────
    // ③ CPU 多執行緒（pthread）
    // ──────────────────────────────────────────
    {
        float*         mag = (float*)malloc(float_size);
        unsigned char* dir = (unsigned char*)malloc(img_size);
        unsigned char* out = (unsigned char*)calloc(width * height, 1);

        pthread_t   tids[NUM_THREADS];
        ThreadArgs  args[NUM_THREADS];

        // 各執行緒共享同一份輸入/輸出指標，只有行範圍不同
        for (int t = 0; t < NUM_THREADS; t++) {
            args[t].in      = h_in;
            args[t].mag_out = mag;
            args[t].dir_out = dir;
            args[t].mag     = mag;
            args[t].dir     = dir;
            args[t].edges   = out;
            args[t].w       = width;
            args[t].h       = height;
        }

        struct timespec ts, te;
        clock_gettime(CLOCK_MONOTONIC, &ts);

        // (1) Sobel
        dispatch_threads(tids, args, NUM_THREADS, height, pthread_sobel);

        // (2) NMS + 閾值
        dispatch_threads(tids, args, NUM_THREADS, height, pthread_nms);

        // (3) Hysteresis（需多次同步迭代，每次都要等全部執行緒結束才能進行下一次）
        for (int iter = 0; iter < HYST_ITER; iter++)
            dispatch_threads(tids, args, NUM_THREADS, height, pthread_hyst);

        // (4) Cleanup
        for (int i = 0; i < width * height; i++)
            if (out[i] == 128) out[i] = 0;

        clock_gettime(CLOCK_MONOTONIC, &te);
        double ms_pt = (te.tv_sec - ts.tv_sec) * 1000.0
                     + (te.tv_nsec - ts.tv_nsec) / 1e6;
        printf("[pthread %d緒]  處理時間: %.3f ms\n", NUM_THREADS, ms_pt);

        stbi_write_jpg("output_pthread.jpg", width, height, 1, out, 90);
        free(mag); free(dir); free(out);
    }

    printf("\n結果已分別存至:\n");
    printf("  output_gpu.jpg     (CUDA GPU)\n");
    printf("  output_cpu.jpg     (CPU 單執行緒)\n");
    printf("  output_pthread.jpg (pthread %d 執行緒)\n", NUM_THREADS);

    stbi_image_free(h_in);
    return 0;
}
