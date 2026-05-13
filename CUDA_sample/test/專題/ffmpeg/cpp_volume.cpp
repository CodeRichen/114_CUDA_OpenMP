#include <iostream>
#include <vector>

// 核心處理函數保持不變
void processVolumeCPU(float* data, int size, float gain) {
    for (int i = 0; i < size; ++i) {
        data[i] *= gain;
        if (data[i] > 1.0f) data[i] = 1.0f;
        if (data[i] < -1.0f) data[i] = -1.0f;
    }
}

int main() {
    const int chunk_size = 4096; // 每次處理一小塊音訊，節省記憶體
    std::vector<float> buffer(chunk_size);

    // 從標準輸入讀取二進位資料，直到結束
    while (std::cin.read(reinterpret_cast<char*>(buffer.data()), chunk_size * sizeof(float))) {
        processVolumeCPU(buffer.data(), chunk_size, 10.0f);
        std::cout.write(reinterpret_cast<char*>(buffer.data()), chunk_size * sizeof(float));
    }

    return 0;
}