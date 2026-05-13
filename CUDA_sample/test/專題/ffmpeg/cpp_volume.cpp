#include <iostream>
#include <vector>
#include <chrono>

void processVolumeCPU(float* data, int size, float gain) {
    for (int i = 0; i < size; ++i) {
        data[i] *= gain;
        // 簡易限幅
        if (data[i] > 1.0f) data[i] = 1.0f;
        if (data[i] < -1.0f) data[i] = -1.0f;
    }
}

int main() {
    const int N = 1 << 24; // 約 1600 萬個採樣點
    std::vector<float> h_data(N, 0.05f); // 初始音量較小

    auto start = std::chrono::high_resolution_clock::now();
    processVolumeCPU(h_data.data(), N, 10.0f);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> diff = end - start;
    std::cout << "CPU 處理時間: " << diff.count() << " ms\n";
    std::cout << "處理後第一個採樣點: " << h_data[0] << " (預期 0.5)\n";

    return 0;
}