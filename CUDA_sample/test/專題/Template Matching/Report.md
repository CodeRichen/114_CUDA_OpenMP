# 2026 嵌入式多核心系統與軟體 - Cuda專題報告

**姓名：** [請填寫你的姓名與學號]

---

## 一、 問題回答

### 1. 請問每組資料各需要多少時間才能做完整張圖片的Template Matching?
*(請先執行程式後，將終端機顯示的時間填入下表)*
* **測資 1 (T:3750x4320, S:3x3):** 約 `1.98` ms (根據 32x16 的表現)
* **測資 2 (T:7750x1320, S:5x5):** 約 `2.65` ms (根據 32x16 的表現)
* **測資 3 (T:8140x9925, S:3x3):** 約 `9.04` ms (根據 32x16 的表現)
* **測資 4 (T:50x50, S:5x5):** 約 `0.008` ms (根據 32x16 的表現)

### 2. 你發現最佳 Block Size 的配置是什麼？請試著分析原因。
* **最佳 Block Size：** `32x16` (或 `32x32`)
* **原因分析：** 
*(範例回答：根據測試結果，32x32 或是適配 GPU Warp Size (32) 的倍數效能較佳。因為 GPU 在調度執行緒時是以 Warp (32個 thread) 為單位，如果 Block 的寬度是 32 的倍數，可以最大化記憶體存取合併 (Memory Coalescing) 的效益，減少 Divergence，也能更充分利用 Streaming Multiprocessor (SM) 的運算資源。)*

### 3. 除了Block Size的參數之外，你還有做什麼特別的效能優化嗎?
* 目前實作中採取了**將兩種算法 (PCC 與 SSD) 合併在同一個 Kernel 中計算**。
* 由於不管是 PCC 還是 SSD，都需要讀取同一個視窗 `S` 與 `T` 中的圖素，合併在一起算可以大幅度減少對 Global Memory 的存取次數 (memory accesses)，也能減少 Kernel 啟動的 Overhead。

### 4. 這次作業的難度?或是其他建議
`[自由發揮分享心得]`

---

## 二、 作業遇到的困難
`[請填寫你在實作時遇到的困難，例如環境設定、記憶體配置、浮點數誤差處理等]`

---

## 三、 程式碼講解

為了兼顧準確性與執行效能，本作業的實作重點如下：

### 1. 檔案讀取最佳化 (`load_matrix`)
由於輸入資料夾內都是 0-9 之間的單一數字並且以逗號分隔，傳統使用 `fscanf` 解析字串速度較慢。程式碼中選用逐字元讀取 `fgetc` 的方式：
```c
while ((c = fgetc(fp)) != EOF && count < rows * cols) {
    if (c >= '0' && c <= '9') {
        mat[count++] = (unsigned char)(c - '0');
    }
}
```
遇到 ASCII 為 '0' 到 '9' 的字元時，自行將其轉為整數值存入記憶體中，此舉大大加快了讀取測資的速度。資料型態選擇使用 `unsigned char` (1 byte)，節省主機與 GPU 間的記憶體搬運開銷與頻寬。

### 2. CUDA Kernel 核心邏輯 (`matchKernel`)
使用 2D 的 Grid 與 Block 維度進行任務拆解，這是一次將每個起始座標的比對任務平行化的核心技術：
```c
int c = blockIdx.x * blockDim.x + threadIdx.x; // 定位滑動視窗的 Column (x)
int r = blockIdx.y * blockDim.y + threadIdx.y; // 定位滑動視窗的 Row (y)
```
* **工作分配**：透過上述的座標映射，GPU 上數以萬計的 Threads 會各自負責 Target (T) 影像上**一個獨立的左上角起點 (r, c)**。每個 Thread 的任務就是在該特定起點抓出與 S 大小相等的窗口，並計算 PCC 與 SSD 數值。
* **預防越界 (Boundary Check)**：
  ```c
  if (r < T_r - S_r + 1 && c < T_c - S_c + 1)
  ```
  確保算出來的 (r, c) 在有效邊界內。防止滑動視窗 (Sliding Window) 讀取到圖片邊界外的非法記憶體段（產生 Segmentation Fault 崩潰）。
* **PCC 與 SSD 同步計算詳細流程**：
    1. **求平均 ($\bar{X}$ 與 $\bar{Y}$)**：利用第一個雙層迴圈 (對應視窗大小 `S_r` 與 `S_c`)，取得 `S` 所有像素以及 `T` 當前窗口 `(r+i, c+j)` 所有像素之總和，分別除以像素數量 $n$ 得到 meanX 和 meanY。
    2. **求分子分母與誤差平方和**：
       - 利用 `dx = x - meanX` 和 `dy = y - meanY`。
       - 分子 `num += dx * dy` 來做其共變異數加總。
       - 分母 `denX += dx*dx` 與 `denY += dy*dy` 做自變異平方和加總。
       - 在同一個迴圈層中，我們直接利用原數值的差值 `diff = x - y` 來計算 SSD 公式 (`ssd += diff * diff`)！
    3. **寫回 Device Memory**：最後針對 PCC 進行開根號運算 `pcc = num / (sqrt(denX) * sqrt(denY))`，並透過 2D 轉 1D 偏移量的公式 `int out_idx = r * (T_c - S_c + 1) + c`，將計算完成的 PCC 與 SSD 寫回 Global Memory。將兩種原本應該獨立呼叫 Kernel 的演算法整合，不僅寫回時共用 Index 計算，還達成 Memory 的高重用率。

### 3. 多組 Block Size 自動測試與事件計時
為了方便分析報告第二題，主程式內配置了一個迴圈與 `cudaEvent_t` 計時器：
```c
int block_sizes[][2] = {{16, 16}, {32, 32}, {32, 16}};
for (int i = 0; i < num_configs; i++) {
    // 建立對應的 block 與 grid
    // cudaEventRecord(start) -> 執行 Kernel -> cudaEventRecord(stop)
}
```
透過這樣可以一次看出同一個 Kernel 在不同 Block 設定之下的耗時差異。

### 4. 浮點數誤差處理與結果尋找
從 Device 複製回 Host 後，因為 PCC 會因為浮點數運算 (floating point) 產生微小誤差，所以在尋找最大值與對應位置時，對最高相似度的容許差為 `fabs(h_pcc_out[idx] - max_pcc) < 1e-4`。SSD 為整數運算則無此問題，可以直接使用 `==`。

