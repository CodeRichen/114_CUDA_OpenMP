#include <iostream>
#include <vector>
#include <stdint.h>
#include <chrono>
#include <string>

// 你的參數
#define MEM_SIZE    33554432ULL 
#define ROUNDS      51200000
#define M1          6364136223846793005ULL
#define A1          1442695040888963407ULL
#define M2          2654435761ULL
#define NUM_COORDS  100

void cpu_k_hash(uint64_t seed) {
    // 在 CPU 上動態分配這 256MB，避免 Stack Overflow
    uint64_t* mem = new uint64_t[MEM_SIZE];

    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. 初始化
    mem[0] = seed;
    for (uint64_t i = 1; i < MEM_SIZE; i++) {
        mem[i] = (mem[i - 1] * M1 + A1); // 自動 64-bit 溢位
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // 2. 混合
    uint64_t state = seed;
    for (int r = 0; r < ROUNDS; r++) {
        uint64_t idx = state % MEM_SIZE;
        state = (state ^ mem[idx]);
        state = (state * M2 + r);
        mem[idx] = state;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    // 3. 洗牌
    int coords[NUM_COORDS];
    for (int i = 0; i < NUM_COORDS; i++) coords[i] = i;

    uint64_t s = state;
    for (int i = NUM_COORDS - 1; i > 0; i--) {
        s = (s * M1 + 1);
        int j = s % (i + 1);
        int temp = coords[i];
        coords[i] = coords[j];
        coords[j] = temp;
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    // 顯示前 5 個結果
    std::cout << "CPU Results (first 5): " << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "(" << coords[i] / 10 << ", " << coords[i] % 10 << ") ";
    }
    std::cout << std::endl;

    auto initMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto mixMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
    auto shuffleMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
    auto totalMs = std::chrono::duration<double, std::milli>(t3 - t0).count();
    std::cout << "CPU init:   " << initMs << " ms" << std::endl;
    std::cout << "CPU mix:    " << mixMs << " ms" << std::endl;
    std::cout << "CPU shuffle:" << shuffleMs << " ms" << std::endl;
    std::cout << "CPU total:  " << totalMs << " ms" << std::endl;

    delete[] mem;
}

int main(int argc, char** argv) {
    // 用法：./hash_cpu [seed]
    uint64_t test_seed = 12345;
    if (argc >= 2) test_seed = (uint64_t)std::stoull(argv[1]);
    cpu_k_hash(test_seed);
    return 0;
}