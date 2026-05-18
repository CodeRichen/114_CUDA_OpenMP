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
    unsigned char *input_data = stbi_load("input2.jpg", &width, &height, &channels, 1);
    if (!input_data) {
        std::cerr << "錯誤：找不到 input.jpg！請確認圖片在同資料夾下。" << std::endl;
        return -1;
    }

    // 準備輸出緩衝區
    std::vector<unsigned char> output_data(width * height, 0);
    std::vector<float> mag(width * height, 0.0f);
    std::vector<unsigned char> dir(width * height, 0);

    std::cout << "圖片讀取成功: " << width << "x" << height << " 像素" << std::endl;

    // 2. 開始計時
    auto start = std::chrono::high_resolution_clock::now();

    // 3. Canny 各個階段
    
    // (1) 計算 Sobel 梯度與方向
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float dx = (-1 * input_data[(y-1)*width + (x-1)]) + (1 * input_data[(y-1)*width + (x+1)]) +
                       (-2 * input_data[(y)*width   + (x-1)]) + (2 * input_data[(y)*width   + (x+1)]) +
                       (-1 * input_data[(y+1)*width + (x-1)]) + (1 * input_data[(y+1)*width + (x+1)]);
            
            float dy = (-1 * input_data[(y-1)*width + (x-1)]) + (-2 * input_data[(y-1)*width + x]) + (-1 * input_data[(y-1)*width + (x+1)]) +
                       ( 1 * input_data[(y+1)*width + (x-1)]) + ( 2 * input_data[(y+1)*width + x]) + ( 1 * input_data[(y+1)*width + (x+1)]);
            
            mag[y*width + x] = std::sqrt(dx*dx + dy*dy);

            float angle = std::atan2(dy, dx) * 180.0f / 3.14159265f;
            if (angle < 0) angle += 180.0f;

            unsigned char d = 0;
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180.0)) d = 0;
            else if (angle >= 22.5 && angle < 67.5) d = 45;
            else if (angle >= 67.5 && angle < 112.5) d = 90;
            else if (angle >= 112.5 && angle < 157.5) d = 135;

            dir[y*width + x] = d;
        }
    }

    // (2) 非極大值抑制 (NMS) 與雙閾值
    float lowThr = 50.0f;
    float highThr = 100.0f;
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float m = mag[y*width + x];
            unsigned char d = dir[y*width + x];
            float m1 = 0, m2 = 0;

            if (d == 0) { m1 = mag[y*width + (x-1)]; m2 = mag[y*width + (x+1)]; }
            else if (d == 45) { m1 = mag[(y+1)*width + (x-1)]; m2 = mag[(y-1)*width + (x+1)]; }
            else if (d == 90) { m1 = mag[(y-1)*width + x]; m2 = mag[(y+1)*width + x]; }
            else if (d == 135) { m1 = mag[(y-1)*width + (x-1)]; m2 = mag[(y+1)*width + (x+1)]; }

            if (m >= m1 && m >= m2) {
                if (m >= highThr) output_data[y*width + x] = 255;
                else if (m >= lowThr) output_data[y*width + x] = 128;
                else output_data[y*width + x] = 0;
            } else {
                output_data[y*width + x] = 0;
            }
        }
    }

    // (3) 邊緣連接 (Hysteresis)
    bool changed = true;
    while (changed) {
        changed = false;
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                if (output_data[y*width + x] == 128) {
                    bool connected = false;
                    for (int i = -1; i <= 1; i++) {
                        for (int j = -1; j <= 1; j++) {
                            if (output_data[(y+i)*width + (x+j)] == 255) {
                                connected = true;
                                break;
                            }
                        }
                    }
                    if (connected) {
                        output_data[y*width + x] = 255;
                        changed = true;
                    }
                }
            }
        }
    }

    // (4) 清除殘留弱邊緣
    for (int i = 0; i < width * height; i++) {
        if (output_data[i] == 128) output_data[i] = 0;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // 4. 存檔
    stbi_write_jpg("output_cpu.jpg", width, height, 1, output_data.data(), 0);

    std::cout << "CPU 處理完成！" << std::endl;
    std::cout << "消耗時間: " << elapsed.count() << " ms" << std::endl;
    std::cout << "結果已存至 output_cpu.jpg" << std::endl;

    stbi_image_free(input_data);
    return 0;
}