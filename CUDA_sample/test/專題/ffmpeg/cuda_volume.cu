#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

__global__ void volumeKernel(float* data, int size, float gain) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx] * gain;
        if (val > 1.0f) val = 1.0f;
        else if (val < -1.0f) val = -1.0f;
        data[idx] = val;
    }
}

int main() {
    const int N = 1 << 24;
    size_t bytes = N * sizeof(float);

    float *h_data = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_data[i] = 0.05f;

    float *d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    auto start = std::chrono::high_resolution_clock::now();
    
    volumeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N, 10.0f);
    cudaDeviceSynchronize(); // 確保核心執行完畢
    
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    std::chrono::duration<double, std::milli> diff = end - start;
    std::cout << "GPU 核心執行時間: " << diff.count() << " ms\n";
    std::cout << "處理後第一個採樣點: " << h_data[0] << "\n";

    cudaFree(d_data);
    free(h_data);
    return 0;
}