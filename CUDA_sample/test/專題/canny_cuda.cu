#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void sobel_kernel(float* input, float* output, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < w - 1 && y > 0 && y < h - 1) {
        float dx = (-1 * input[(y-1)*w + (x-1)]) + (1 * input[(y-1)*w + (x+1)]) +
                   (-2 * input[(y)*w   + (x-1)]) + (2 * input[(y)*w   + (x+1)]) +
                   (-1 * input[(y+1)*w + (x-1)]) + (1 * input[(y+1)*w + (x+1)]);
        
        float dy = (-1 * input[(y-1)*w + (x-1)]) + (-2 * input[(y-1)*w + x]) + (-1 * input[(y-1)*w + (x+1)]) +
                   ( 1 * input[(y+1)*w + (x-1)]) + ( 2 * input[(y+1)*w + x]) + ( 1 * input[(y+1)*w + (x+1)]);
        
        output[y*w + x] = sqrtf(dx*dx + dy*dy);
    }
}

int main() {
    int W = 1024, H = 1024;
    size_t size = W * H * sizeof(float);

    float *h_in = new float[W*H];
    float *h_out = new float[W*H];
    for(int i=0; i<W*H; i++) h_in[i] = (i % 500 < 250) ? 255.0f : 0.0f; // 模擬條紋

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((W + blockSize.x - 1) / blockSize.x, (H + blockSize.y - 1) / blockSize.y);

    // 計時開始
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    sobel_kernel<<<gridSize, blockSize>>>(d_in, d_out, W, H);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    std::cout << "GPU Sobel Time: " << ms << " ms" << std::endl;

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return 0;
}