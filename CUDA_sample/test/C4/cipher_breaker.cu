#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <string>

// 定義常數
#define ALPHABET_SIZE 36
#define TOTAL_KEYS 1679616 // 36^4 總共約 168 萬種金鑰空間

// 表 A 的字符陣列對照表 (a~z: 0~25, 0~9: 26~35)
__device__ char get_A_char(int v) {
    v = (v % 36 + 36) % 36;
    if (v < 26) return 'a' + v;
    return '0' + (v - 26);
}

__device__ int get_A_val(char c) {
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c >= '0' && c <= '9') return c - '0' + 26;
    return 0;
}

// 輾轉相除法求模逆元 (GPU 版本)
__device__ int modInverse(int a, int m) {
    a = (a % m + m) % m;
    for (int x = 1; x < m; x++) {
        if ((a * x) % m == 1) return x;
    }
    return -1;
}

// CUDA 核心：在 GPU 內同時展開 168 萬個執行緒平行破解
__global__ void bruteForceCipher(char* d_c2, char* d_result_key, char* d_result_pt, int* d_found) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL_KEYS) return;

    // 1. 從執行緒 ID 逆推 4 位原始 Master Key (k1)
    int temp = idx;
    char k1[4];
    k1[3] = get_A_char(temp % 36); temp /= 36;
    k1[2] = get_A_char(temp % 36); temp /= 36;
    k1[1] = get_A_char(temp % 36); temp /= 36;
    k1[0] = get_A_char(temp % 36);

    // 2. 模擬子金鑰衍生：決定 Hill 矩陣 M
    int m_sum = 0;
    for (int i = 0; i < 4; i++) m_sum += get_A_val(k1[i]);
    int M_val = m_sum % 6;

    // 定義 6 組 2x2 希爾矩陣
    int HILL[6][2][2] = {
        {{1, 2}, {3, 5}},
        {{5, 6}, {17, 3}},
        {{3, 5}, {1, 2}},
        {{7, 8}, {11, 11}},
        {{5, 4}, {3, 7}},
        {{9, 4}, {5, 7}}
    };

    int mat[2][2];
    mat[0][0] = HILL[M_val][0][0]; mat[0][1] = HILL[M_val][0][1];
    mat[1][0] = HILL[M_val][1][0]; mat[1][1] = HILL[M_val][1][1];

    // 計算希爾矩陣的行列式值與反矩陣
    int det = (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]) % 36;
    det = (det + 36) % 36;
    int det_inv = modInverse(det, 36);

    if (det_inv == -1) return; // 矩陣不可逆，排除此無效 Key

    int inv_mat[2][2];
    inv_mat[0][0] = (mat[1][1] * det_inv) % 36;
    inv_mat[0][1] = ((-mat[0][1]) * det_inv) % 36;
    inv_mat[1][0] = ((-mat[1][0]) * det_inv) % 36;
    inv_mat[1][1] = (mat[0][0] * det_inv) % 36;

    // 3. 執行 2 號逆換位 (假設 C2 長度為 12，將其逆轉對調回 C1 狀態)
    char c1_str[12];
    for (int i = 0; i < 12; i++) {
        c1_str[i] = d_c2[11 - i]; // 頭尾翻轉置換
    }

    // 4. 執行 1 號逆代換 (Hill 解密 + 維吉尼亞還原)
    char plaintext[13];
    plaintext[12] = '\0';
    bool valid = true;

    for (int i = 0; i < 12; i += 2) {
        int c_val1 = get_A_val(c1_str[i]);
        int c_val2 = get_A_val(c1_str[i + 1]);

        // Hill 矩陣解密
        int p_hill1 = (inv_mat[0][0] * c_val1 + inv_mat[0][1] * c_val2) % 36;
        int p_hill2 = (inv_mat[1][0] * c_val1 + inv_mat[1][1] * c_val2) % 36;
        p_hill1 = (p_hill1 + 36) % 36;
        p_hill2 = (p_hill2 + 36) % 36;

        // 維吉尼亞逆位移還原明文
        int k_val1 = get_A_val(k1[i % 4]);
        int k_val2 = get_A_val(k1[(i + 1) % 4]);

        int p_orig1 = (p_hill1 - k_val1 + 36) % 36;
        int p_orig2 = (p_hill2 - k_val2 + 36) % 36;

        plaintext[i] = get_A_char(p_orig1);
        plaintext[i + 1] = get_A_char(p_orig2);
    }

    // 5. 檢查前 4 碼是不是 "book"
    if (plaintext[0] == 'b' && plaintext[1] == 'o' && plaintext[2] == 'o' && plaintext[3] == 'k') {
        // 原子操作：記錄找到的答案
        if (atomicExch(d_found, 1) == 0) {
            for (int i = 0; i < 4; i++) d_result_key[i] = k1[i];
            for (int i = 0; i < 12; i++) d_result_pt[i] = plaintext[i];
        }
    }
}

int main() {
    // 輸入你提供長度為 12 的 C2 密文字串
    std::string c2_input = "ryryrxwyxtxr"; 
    
    char h_c2[12];
    memcpy(h_c2, c2_input.c_str(), 12);

    char h_result_key[4] = {0};
    char h_result_pt[12] = {0};
    int h_found = 0;

    // 配置 GPU 記憶體
    char *d_c2, *d_result_key, *d_result_pt;
    int *d_found;
    cudaMalloc((void**)&d_c2, 12 * sizeof(char));
    cudaMalloc((void**)&d_result_key, 4 * sizeof(char));
    cudaMalloc((void**)&d_result_pt, 12 * sizeof(char));
    cudaMalloc((void**)&d_found, sizeof(int));

    // 複製資料到 GPU
    cudaMemcpy(d_c2, h_c2, 12 * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found, &h_found, sizeof(int), cudaMemcpyDeviceToHost);

    // 設定 CUDA 執行緒形狀：並行爆破
    int threadsPerBlock = 256;
    int blocksPerGrid = (TOTAL_KEYS + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "🚀 GPU 平行密碼破譯核心已啟動，正在枚舉 " << TOTAL_KEYS << " 組金鑰空間..." << std::endl;
    
    // 呼叫 GPU 核心
    bruteForceCipher<<<blocksPerGrid, threadsPerBlock>>>(d_c2, d_result_key, d_result_pt, d_found);
    cudaDeviceSynchronize();

    // 撈回結果
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_found) {
        cudaMemcpy(h_result_key, d_result_key, 4 * sizeof(char), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_result_pt, d_result_pt, 12 * sizeof(char), cudaMemcpyDeviceToHost);

        std::cout << "\n[🔓 破譯成功！]" << std::endl;
        std::cout << "🔑 找到符合邏輯的原始 Master Key: ";
        for(int i=0; i<4; i++) std::cout << h_result_key[i];
        std::cout << "\n📖 解密出的明文前綴確實為: " << std::string(h_result_pt, 4) << std::endl;
        std::cout << "📄 完整長度解密明文: " << std::string(h_result_pt, 12) << std::endl;
    } else {
        std::cout << "\n❌ 在當前的 2 號逆置換與 1 號解密邏輯下，無法將 \"" << c2_input << "\" 還原出以 \"book\" 開頭的明文。" << std::endl;
        std::cout << "💡 提示：請確認該密文是否已包含了 3 號（奇偶代換）或 4 號（三角形換位）的混合。若是，需要進一步將其解密核心上傳至核心函數中。" << std::endl;
    }

    // 釋放記憶體
    cudaFree(d_c2); cudaFree(d_result_key); cudaFree(d_result_pt); cudaFree(d_found);
    return 0;
}