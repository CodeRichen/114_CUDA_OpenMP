#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <stdint.h>
#include <chrono>
#include <string>
#include <cstdlib>

static inline void cuda_check(cudaError_t err, const char* file, int line) {
    if (err == cudaSuccess) return;
    std::cerr << "CUDA error at " << file << ":" << line << ": " << cudaGetErrorString(err) << std::endl;
    std::exit(1);
}

#define CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)

// 定義與 Python 一致的常數
#define MEM_SIZE    33554432ULL // 4,194,304 * 8
#define ROUNDS      51200000
#define M1          6364136223846793005ULL
#define A1          1442695040888963407ULL
#define M2          2654435761ULL
#define MASK        0xFFFFFFFFFFFFFFFFULL
#define NUM_COORDS  100

#if (MEM_SIZE & (MEM_SIZE - 1))
#error "MEM_SIZE must be a power of two"
#endif

// CUDA Kernel
__global__ void k_hash_kernel(const uint64_t* __restrict__ d_seeds,
                             int* __restrict__ d_results,
                             uint64_t* __restrict__ d_big_mem,
                             int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // 定位到該 Thread 專屬的 256MB 記憶體區段
    uint64_t* my_mem = &d_big_mem[tid * MEM_SIZE];

    uint64_t seed = d_seeds[tid];

    // 1. 初始化 mem (對應 Python: mem[i] = (mem[i-1] * M1 + A1) & MASK)
    my_mem[0] = seed;
    for (uint64_t i = 1; i < MEM_SIZE; i++) {
        my_mem[i] = (my_mem[i - 1] * M1 + A1); // uint64_t 自動處理 MASK (溢位)
    }

    // 2. 狀態混淆 (對應 Python: state 變換)
    constexpr uint64_t MEM_MASK = (uint64_t)(MEM_SIZE - 1ULL);
    uint64_t state = seed;
    for (int r = 0; r < ROUNDS; r++) {
        uint64_t idx = state & MEM_MASK; // MEM_SIZE is power of two
        state = (state ^ my_mem[idx]);
        state = (state * (uint64_t)M2 + (uint64_t)r);
        my_mem[idx] = state;
    }

    // 3. 洗牌演算法 (Shuffle coords)
    int coords[NUM_COORDS];
    for (int i = 0; i < NUM_COORDS; i++) coords[i] = i;

    uint64_t s = state;
    for (int i = NUM_COORDS - 1; i > 0; i--) {
        s = (s * M1 + 1);
        int j = s % (i + 1);
        // Swap
        int temp = coords[i];
        coords[i] = coords[j];
        coords[j] = temp;
    }

    // 4. 回傳前 78 個座標 (轉為 1D 存放)
    for (int i = 0; i < 78; i++) {
        d_results[tid * 78 + i] = coords[i];
    }
}

int main(int argc, char** argv) {
    // 用法：./hash [seed] [n] [mode]
    // mode: same (預設) | inc
    // 注意：每個 n 會佔用 256MB 顯存 (MEM_SIZE * 8 bytes)。
    uint64_t h_seed_val = 12345;
    int n = 1;
    std::string mode = "same";
    if (argc >= 2) h_seed_val = (uint64_t)std::stoull(argv[1]);
    if (argc >= 3) n = std::stoi(argv[2]);
    if (argc >= 4) mode = std::string(argv[3]);
    if (n <= 0) {
        std::cerr << "n must be > 0" << std::endl;
        return 1;
    }

    if (!(mode == "same" || mode == "inc")) {
        std::cerr << "mode must be 'same' or 'inc'" << std::endl;
        return 1;
    }

    std::vector<uint64_t> h_seeds((size_t)n);
    if (mode == "inc") {
        for (int i = 0; i < n; i++) h_seeds[i] = h_seed_val + (uint64_t)i;
    } else {
        for (int i = 0; i < n; i++) h_seeds[i] = h_seed_val;
    }

    uint64_t* d_seeds = nullptr;
    int* d_results = nullptr;
    uint64_t* d_big_mem = nullptr;

    auto t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMalloc(&d_seeds, (size_t)n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_results, (size_t)n * 78 * sizeof(int)));
    // 分配巨大的 Global Memory
    CUDA_CHECK(cudaMalloc(&d_big_mem, (size_t)n * (size_t)MEM_SIZE * sizeof(uint64_t)));
    auto t1 = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(d_seeds, h_seeds.data(), (size_t)n * sizeof(uint64_t), cudaMemcpyHostToDevice));
    auto t2 = std::chrono::high_resolution_clock::now();

    // Kernel 計時（只量 kernel，不含 malloc/memcpy）
    cudaEvent_t evStart, evStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    CUDA_CHECK(cudaEventRecord(evStart));
    k_hash_kernel<<<blocks, threads>>>(d_seeds, d_results, d_big_mem, n);
    CUDA_CHECK(cudaEventRecord(evStop));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(evStop));

    float kernelMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, evStart, evStop));
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));

    auto tKernelDone = std::chrono::high_resolution_clock::now();

    std::vector<int> h_results((size_t)n * 78);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, (size_t)n * 78 * sizeof(int), cudaMemcpyDeviceToHost));
    auto t3 = std::chrono::high_resolution_clock::now();

    auto allocMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto h2dMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
    auto kernelHostMs = std::chrono::duration<double, std::milli>(tKernelDone - t2).count();
    auto d2hMs = std::chrono::duration<double, std::milli>(t3 - tKernelDone).count();

    std::cout << "Alloc time:  " << allocMs << " ms" << std::endl;
    std::cout << "H2D time:    " << h2dMs << " ms" << std::endl;
    std::cout << "Kernel time: " << kernelMs << " ms" << std::endl;
    std::cout << "Kernel(host sync) time: " << kernelHostMs << " ms" << std::endl;
    std::cout << "D2H time:    " << d2hMs << " ms" << std::endl;
    std::cout << "Kernel avg per hash: " << (kernelMs / (float)n) << " ms" << std::endl;
    double bigMemGiB = ((double)n * (double)MEM_SIZE * (double)sizeof(uint64_t)) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "n=" << n << ", mode=" << mode << ", seed=" << h_seed_val << ", big_mem~" << bigMemGiB << " GiB" << std::endl;

    // 顯示第 0 筆結果（和 CPU 比對用）
    std::cout << "Top 10 Coordinates (x, y):" << std::endl;
    for (int i = 0; i < 10; i++) {
        int val = h_results[i];
        std::cout << "(" << val / 10 << ", " << val % 10 << ") ";
    }
    std::cout << std::endl;

    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_big_mem));

    return 0;
}