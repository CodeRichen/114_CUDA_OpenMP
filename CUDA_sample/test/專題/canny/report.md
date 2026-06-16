# CUDA Canny 邊緣偵測程式碼詳解

## 概覽

本程式使用 **CUDA（Compute Unified Device Architecture）** 在 GPU 上平行化實作 Canny 邊緣偵測演算法。Canny 是電腦視覺中最常用的邊緣偵測方法，能在雜訊抑制與邊緣精準度之間取得平衡。

整體流程分為四個階段：

```
輸入圖片 → [1] Sobel 梯度計算 → [2] NMS + 雙閾值 → [3] Hysteresis 連接 → [4] 清除弱邊緣 → 輸出
```

---

## 依賴函式庫

```cpp
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cuda_runtime.h>
```

- **stb_image / stb_image_write**：單頭檔圖片讀寫函式庫，支援 JPG、PNG 等格式。前方的 `#define` 是啟用實作碼的必要宣告。
- **cuda_runtime.h**：CUDA 執行時期 API，提供 `cudaMalloc`、`cudaMemcpy`、Kernel 啟動等功能。

---

## 階段一：Sobel 梯度計算

```cpp
__global__ void sobel_kernel(const unsigned char* input, float* mag, unsigned char* dir, int w, int h)
```

### 目的

計算每個像素的**梯度強度（magnitude）**與**梯度方向（direction）**，以找出亮度變化劇烈的區域（即邊緣）。

### CUDA 執行模型

`__global__` 修飾詞表示這是一個 **Kernel 函式**，由 CPU 呼叫、在 GPU 上執行。每個 CUDA Thread 負責處理一個像素：

```cpp
int x = blockIdx.x * blockDim.x + threadIdx.x;  // 像素的 X 座標
int y = blockIdx.y * blockDim.y + threadIdx.y;  // 像素的 Y 座標
```

### Sobel 運算子

Sobel 使用兩個 3×3 卷積核分別計算水平（dx）與垂直（dy）方向的梯度：

**水平方向（Gx）：**
```
-1  0  +1
-2  0  +2
-1  0  +1
```

**垂直方向（Gy）：**
```
-1  -2  -1
 0   0   0
+1  +2  +1
```

程式碼實作如下：

```cpp
float dx = (-1 * input[(y-1)*w + (x-1)]) + (1 * input[(y-1)*w + (x+1)]) +
           (-2 * input[(y)*w   + (x-1)]) + (2 * input[(y)*w   + (x+1)]) +
           (-1 * input[(y+1)*w + (x-1)]) + (1 * input[(y+1)*w + (x+1)]);

float dy = (-1 * input[(y-1)*w + (x-1)]) + (-2 * input[(y-1)*w + x]) + (-1 * input[(y-1)*w + (x+1)]) +
           ( 1 * input[(y+1)*w + (x-1)]) + ( 2 * input[(y+1)*w + x]) + ( 1 * input[(y+1)*w + (x+1)]);
```

> **注意**：邊界像素（x=0, y=0 等）直接設為 0，避免存取越界。

### 梯度強度與方向

```cpp
float magnitude = sqrtf(dx*dx + dy*dy);  // 歐幾里得距離
float angle = atan2f(dy, dx) * 180.0f / 3.14159265f;  // 轉換為角度（0°~180°）
```

**角度離散化**（4 個方向）：

| 角度範圍 | 離散方向 | 意義 |
|----------|----------|------|
| 0° ~ 22.5° 或 157.5° ~ 180° | 0 | 水平邊緣 |
| 22.5° ~ 67.5° | 45 | 右上-左下斜邊 |
| 67.5° ~ 112.5° | 90 | 垂直邊緣 |
| 112.5° ~ 157.5° | 135 | 左上-右下斜邊 |

---

## 階段二：非極大值抑制（NMS）與雙閾值

```cpp
__global__ void nms_threshold_kernel(const float* mag, const unsigned char* dir, unsigned char* output,
                                      int w, int h, float lowThr, float highThr)
```

### 非極大值抑制（Non-Maximum Suppression）

目的：**細化邊緣**，只保留梯度方向上的局部最大值，消除邊緣的「胖化」問題。

每個像素會沿著其梯度方向，比較左右兩個鄰近像素的強度：

```cpp
if (d == 0)   { m1 = mag[y*w + (x-1)];   m2 = mag[y*w + (x+1)]; }    // 水平比較
if (d == 45)  { m1 = mag[(y+1)*w+(x-1)]; m2 = mag[(y-1)*w+(x+1)]; }  // 右上-左下
if (d == 90)  { m1 = mag[(y-1)*w + x];   m2 = mag[(y+1)*w + x]; }    // 垂直比較
if (d == 135) { m1 = mag[(y-1)*w+(x-1)]; m2 = mag[(y+1)*w+(x+1)]; }  // 左上-右下
```

只有當前像素的梯度強度 `m >= m1 && m >= m2` 時才保留，否則歸零。

### 雙閾值分類

保留的像素依強度分為三類：

```cpp
if (m >= highThr) output[y*w + x] = 255;      // 強邊緣（一定是邊緣）
else if (m >= lowThr) output[y*w + x] = 128;  // 弱邊緣（可能是邊緣）
else output[y*w + x] = 0;                      // 非邊緣
```

| 值 | 類型 | 說明 |
|----|------|------|
| 255 | 強邊緣 | 梯度強，直接確認為邊緣 |
| 128 | 弱邊緣 | 梯度中等，需看連通性決定 |
| 0 | 非邊緣 | 直接丟棄 |

> 程式預設 `lowThr = 50`，`highThr = 100`，可視圖片性質調整。

---

## 階段三：Hysteresis 邊緣連接

```cpp
__global__ void hysteresis_kernel(unsigned char* edges, int w, int h)
```

### 目的

將與**強邊緣（255）相鄰的弱邊緣（128）**升級為強邊緣，從而連接斷裂的邊緣線段。

```cpp
if (edges[y*w + x] == 128) {
    bool connected = false;
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
            if (edges[(y+i)*w + (x+j)] == 255)
                connected = true;

    if (connected)
        edges[y*w + x] = 255;  // 升級為強邊緣
}
```

每個弱邊緣像素會檢查其 **3×3 鄰域**（8 個方向）是否存在強邊緣，若是則升級。

### 多次迭代

```cpp
for (int i = 0; i < 5; i++) {
    hysteresis_kernel<<<gridSize, blockSize>>>(d_out, width, height);
    cudaDeviceSynchronize();
}
```

由於升級資訊需要逐步傳遞（弱邊緣 A 連接弱邊緣 B 連接強邊緣 C），程式執行 **5 次迭代**確保長鏈的弱邊緣都能正確連接。

> **說明**：理想做法是使用迭代直到收斂，但固定 5 次在多數情況下已足夠。

---

## 階段四：清除未連結的弱邊緣

```cpp
__global__ void cleanup_kernel(unsigned char* edges, int w, int h)
```

```cpp
if (edges[y*w + x] == 128)
    edges[y*w + x] = 0;  // 孤立的弱邊緣，判定為雜訊，清除
```

Hysteresis 完成後，仍殘留的 128 值像素表示它們**無法追溯到任何強邊緣**，視為雜訊直接清除。

---

## 主程式流程（main）

### 1. 讀取圖片

```cpp
unsigned char *h_in = stbi_load("input2.jpg", &width, &height, &channels, 1);
```

最後一個參數 `1` 強制輸出為**灰階**，無論原圖為彩色或黑白，Canny 演算法只需灰階資料。

### 2. GPU 記憶體配置

```cpp
cudaMalloc(&d_in, img_size);   // 輸入灰階影像
cudaMalloc(&d_out, img_size);  // 輸出邊緣影像
cudaMalloc(&d_dir, img_size);  // 梯度方向
cudaMalloc(&d_mag, float_size); // 梯度強度（浮點數）

cudaMemcpy(d_in, h_in, img_size, cudaMemcpyHostToDevice);  // CPU → GPU
```

### 3. Thread 配置

```cpp
dim3 blockSize(16, 16);  // 每個 Block 包含 16×16 = 256 個 Thread
dim3 gridSize((width + 15) / 16, (height + 15) / 16);  // 覆蓋整張影像
```

2D 的 Block/Grid 設計讓每個 Thread 天然對應一個 2D 像素座標，程式結構清晰。

### 4. CUDA 事件計時

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
// ... 執行 Kernel ...
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("GPU 處理時間: %f ms\n", ms);
```

`cudaEvent` 是 CUDA 提供的高精度 GPU 計時器，比 CPU 計時更準確，能排除資料傳輸等 CPU 端延遲的干擾。

### 5. 結果回傳與存檔

```cpp
cudaMemcpy(h_out, d_out, img_size, cudaMemcpyDeviceToHost);  // GPU → CPU
stbi_write_jpg("output.jpg", width, height, 1, h_out, 0);
```

### 6. 記憶體釋放

```cpp
stbi_image_free(h_in);  // 釋放 CPU 圖片記憶體
free(h_out);
cudaFree(d_in);         // 釋放 GPU 記憶體
cudaFree(d_out);
cudaFree(d_dir);
cudaFree(d_mag);
```

---

## 資料流總覽

```
                    ┌─────────────────────────────────────────────────┐
                    │                    GPU                          │
  CPU (Host)        │                                                 │
                    │  d_in (灰階)                                    │
  h_in ──────────► │     │                                           │
  (stbi_load)       │     ▼                                           │
                    │  sobel_kernel ──► d_mag (梯度強度)              │
                    │               └─► d_dir (梯度方向)              │
                    │     │                                           │
                    │     ▼                                           │
                    │  nms_threshold_kernel ──► d_out (強/弱/無邊緣)  │
                    │     │                                           │
                    │     ▼                                           │
                    │  hysteresis_kernel ×5 ──► d_out (邊緣連接)     │
                    │     │                                           │
                    │     ▼                                           │
                    │  cleanup_kernel ──► d_out (最終邊緣)            │
                    │     │                                           │
  h_out ◄────────── │     ▼                                           │
  (stbi_write_jpg)  │                                                 │
                    └─────────────────────────────────────────────────┘
```

---

## 效能考量

| 議題 | 說明 |
|------|------|
| **平行度** | 每個像素獨立計算，GPU 可同時處理數百萬像素 |
| **記憶體存取** | Sobel 需存取 3×3 鄰域，可用 Shared Memory 優化減少 Global Memory 讀取 |
| **Hysteresis 迭代** | 固定 5 次，不保證完全收斂，可改為含終止條件的迴圈 |
| **邊界處理** | 邊界直接設 0，簡單但會略微損失邊界處的邊緣資訊 |
| **閾值選擇** | `lowThr=50, highThr=100` 為固定值，實際應用建議改為自適應（如 Otsu 方法） |

---

## 編譯方式

```bash
nvcc -o canny_cuda canny_cuda.cu -O2
./canny_cuda
```

確保 `input2.jpg` 與執行檔在同一目錄下，結果將輸出至 `output.jpg`。