#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

int main() {
    int width, height, channels;
    
    // 1. 讀取圖片 (強制轉為灰階 1 channel)
    unsigned char *input_data = stbi_load("input.png", &width, &height, &channels, 1);
    if (!input_data) {
        std::cerr << "錯誤：找不到 input.png！請確認圖片在同資料夾下。" << std::endl;
        return -1;
    }

    // 準備輸出緩衝區
    std::vector<unsigned char> output_data(width * height, 0);

    std::cout << "圖片讀取成功: " << width << "x" << height << " 像素" << std::endl;

    // 2. 開始計時
    auto start = std::chrono::high_resolution_clock::now();

    // 3. Sobel 運算 (跳過最外圈邊緣像素)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            // Sobel 算子卷積 (與 CUDA 版邏輯一致)
            float dx = (-1 * input_data[(y-1)*width + (x-1)]) + (1 * input_data[(y-1)*width + (x+1)]) +
                       (-2 * input_data[(y)*width   + (x-1)]) + (2 * input_data[(y)*width   + (x+1)]) +
                       (-1 * input_data[(y+1)*width + (x-1)]) + (1 * input_data[(y+1)*width + (x+1)]);
            
            float dy = (-1 * input_data[(y-1)*width + (x-1)]) + (-2 * input_data[(y-1)*width + x]) + (-1 * input_data[(y-1)*width + (x+1)]) +
                       ( 1 * input_data[(y+1)*width + (x-1)]) + ( 2 * input_data[(y+1)*width + x]) + ( 1 * input_data[(y+1)*width + (x+1)]);
            
            float grad = std::sqrt(dx*dx + dy*dy);
            
            // 限制在 0-255 之間
            output_data[y*width + x] = (grad > 255) ? 255 : (unsigned char)grad;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // 4. 存檔
    stbi_write_png("output_cpu.png", width, height, 1, output_data.data(), 0);

    std::cout << "CPU 處理完成！" << std::endl;
    std::cout << "消耗時間: " << elapsed.count() << " ms" << std::endl;
    std::cout << "結果已存至 output_cpu.png" << std::endl;

    stbi_image_free(input_data);
    return 0;
}