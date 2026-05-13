#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// 簡單的影像結構
struct Image {
    int width, height;
    std::vector<float> data;
    Image(int w, int h) : width(w), height(h), data(w * h, 0) {}
};

// CPU 版 Sobel 邊緣偵測
void sobel_cpu(const Image& input, Image& output) {
    int w = input.width;
    int h = input.height;
    
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            // Sobel 算子卷積
            float dx = (-1 * input.data[(y-1)*w + (x-1)]) + (1 * input.data[(y-1)*w + (x+1)]) +
                       (-2 * input.data[(y)*w   + (x-1)]) + (2 * input.data[(y)*w   + (x+1)]) +
                       (-1 * input.data[(y+1)*w + (x-1)]) + (1 * input.data[(y+1)*w + (x+1)]);
            
            float dy = (-1 * input.data[(y-1)*w + (x-1)]) + (-2 * input.data[(y-1)*w + x]) + (-1 * input.data[(y-1)*w + (x+1)]) +
                       ( 1 * input.data[(y+1)*w + (x-1)]) + ( 2 * input.data[(y+1)*w + x]) + ( 1 * input.data[(y+1)*w + (x+1)]);
            
            output.data[y*w + x] = std::sqrt(dx*dx + dy*dy);
        }
    }
}

int main() {
    int W = 1024, H = 1024;
    Image img_in(W, H), img_out(W, H);

    // 模擬一張中間有個方塊的圖片
    for(int y=256; y<768; y++) for(int x=256; x<768; x++) img_in.data[y*W+x] = 255.0f;

    auto start = std::chrono::high_resolution_clock::now();
    sobel_cpu(img_in, img_out);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "CPU Sobel Time: " << elapsed.count() << " ms" << std::endl;
    std::cout << "Edge detected at (512, 256): " << img_out.data[256*W + 512] << std::endl;

    return 0;
}