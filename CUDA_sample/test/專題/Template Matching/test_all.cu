#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <float.h>
#include <chrono>

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

unsigned char* load_matrix(const char* filename, int rows, int cols) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    
    unsigned char* mat = (unsigned char*)malloc(rows * cols * sizeof(unsigned char));
    int count = 0;
    int c;
    while ((c = fgetc(fp)) != EOF && count < rows * cols) { 
        if (c >= '0' && c <= '9') {
            mat[count++] = (unsigned char)(c - '0');
        }
    }
    fclose(fp);
    return mat;
}

void matchCPU(const unsigned char* T, int T_r, int T_c,
              const unsigned char* S, int S_r, int S_c,
              float* pcc_out, unsigned int* ssd_out) 
{
    for (int r = 0; r < T_r - S_r + 1; r++) {
        for (int c = 0; c < T_c - S_c + 1; c++) {
            float sumX = 0, sumY = 0;
            int n = S_r * S_c;
            for (int i = 0; i < S_r; i++) {
                for (int j = 0; j < S_c; j++) {
                    float valX = S[i * S_c + j];
                    float valY = T[(r + i) * T_c + (c + j)];
                    sumX += valX;
                    sumY += valY;
                }
            }
            float meanX = sumX / n;
            float meanY = sumY / n;

            float num = 0, denX = 0, denY = 0;
            unsigned int ssd = 0;

            for (int i = 0; i < S_r; i++) {
                for (int j = 0; j < S_c; j++) {
                    float x = S[i * S_c + j];
                    float y = T[(r + i) * T_c + (c + j)];
                    
                    float dx = x - meanX;
                    float dy = y - meanY;
                    
                    num += dx * dy;
                    denX += dx * dx;
                    denY += dy * dy;
                    
                    float diff = x - y;
                    ssd += (unsigned int)(diff * diff);
                }
            }

            float pcc = 0.0f;
            if (denX > 0 && denY > 0) {
                pcc = num / (sqrtf(denX) * sqrtf(denY));
            }

            int out_idx = r * (T_c - S_c + 1) + c;
            pcc_out[out_idx] = pcc;
            ssd_out[out_idx] = ssd;
        }
    }
}

__global__ void matchKernel(const unsigned char* T, int T_r, int T_c,
                            const unsigned char* S, int S_r, int S_c,
                            float* pcc_out, unsigned int* ssd_out) 
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < T_r - S_r + 1 && c < T_c - S_c + 1) {
        float sumX = 0, sumY = 0;
        int n = S_r * S_c;
        for (int i = 0; i < S_r; i++) {
            for (int j = 0; j < S_c; j++) {
                float valX = S[i * S_c + j];
                float valY = T[(r + i) * T_c + (c + j)];
                sumX += valX;
                sumY += valY;
            }
        }
        float meanX = sumX / n;
        float meanY = sumY / n;

        float num = 0, denX = 0, denY = 0;
        unsigned int ssd = 0;

        for (int i = 0; i < S_r; i++) {
            for (int j = 0; j < S_c; j++) {
                float x = S[i * S_c + j];
                float y = T[(r + i) * T_c + (c + j)];
                
                float dx = x - meanX;
                float dy = y - meanY;
                
                num += dx * dy;
                denX += dx * dx;
                denY += dy * dy;
                
                float diff = x - y;
                ssd += (unsigned int)(diff * diff);
            }
        }

        float pcc = 0.0f;
        if (denX > 0 && denY > 0) {
            pcc = num / (sqrtf(denX) * sqrtf(denY));
        }

        int out_idx = r * (T_c - S_c + 1) + c;
        pcc_out[out_idx] = pcc;
        ssd_out[out_idx] = ssd;
    }
}

#define MAX_S_SIZE 1024
__constant__ unsigned char c_S[MAX_S_SIZE];

extern __shared__ unsigned char s_T[];

__global__ void matchKernelOptimized(const unsigned char* T, int T_r, int T_c,
                                    int S_r, int S_c,
                                    float* pcc_out, unsigned int* ssd_out) 
{
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // 把目標區域 (Tile of T 包含 padding) 載入 Shared Memory 中
    int tile_w = blockDim.x;
    int tile_h = blockDim.y;
    int T_tile_w = tile_w + S_c - 1;
    int T_tile_h = tile_h + S_r - 1;
    int T_area = T_tile_w * T_tile_h;

    int block_start_c = blockIdx.x * tile_w;
    int block_start_r = blockIdx.y * tile_h;

    // [改進1] Coalesced memory access:
    // 將 1D 的 tid 拆解為 tr, tc，並對應到 global_c。
    // 因相鄰的 thread 擁有連續的 tid，計算出的 tc 也會是連續的，
    // 這確保了 Warp 中的 threads 讀取 T 時是連續的記憶體位置 (One memory transaction)
    for (int i = tid; i < T_area; i += blockDim.x * blockDim.y) {
        int tr = i / T_tile_w;
        int tc = i % T_tile_w;
        int global_r = block_start_r + tr;
        int global_c = block_start_c + tc;
        
        if (global_r < T_r && global_c < T_c) {
            s_T[tr * T_tile_w + tc] = T[global_r * T_c + global_c];
        } else {
            s_T[tr * T_tile_w + tc] = 0;
        }
    }
    __syncthreads();

    int c = block_start_c + threadIdx.x;
    int r = block_start_r + threadIdx.y;

    if (r < T_r - S_r + 1 && c < T_c - S_c + 1) {
        // [改進2] 減少重複計算: 將兩次迴圈化簡為單次迴圈，透過數學展開 O(1) 算出變異數與共變異數
        // sum(x-mean)^2 = sum(x^2) - n*mean^2 
        // 且透過使用 Shared Memory 避免重複到 Global Memory 抓取資料
        float sumX = 0, sumY = 0;
        float sumX2 = 0, sumY2 = 0, sumXY = 0;
        int n = S_r * S_c;

        for (int i = 0; i < S_r; i++) {
            for (int j = 0; j < S_c; j++) {
                float x = c_S[i * S_c + j];
                float y = s_T[(threadIdx.y + i) * T_tile_w + (threadIdx.x + j)];
                
                sumX += x;
                sumY += y;
                sumX2 += x * x;
                sumY2 += y * y;
                sumXY += x * y;
            }
        }
        
        float meanX = sumX / n;
        float meanY = sumY / n;

        float num = sumXY - n * meanX * meanY;
        float denX = sumX2 - n * meanX * meanX;
        float denY = sumY2 - n * meanY * meanY;

        // 計算 SSD: (x-y)^2 = x^2 + y^2 - 2xy
        unsigned int ssd = (unsigned int)(sumX2 + sumY2 - 2 * sumXY);

        float pcc = 0.0f;
        if (denX > 0 && denY > 0) {
            pcc = num / (sqrtf(denX) * sqrtf(denY));
        }

        int out_idx = r * (T_c - S_c + 1) + c;
        pcc_out[out_idx] = pcc;
        ssd_out[out_idx] = ssd;
    }
}

// 測資結構體
typedef struct {
    int id;
    const char* t_file;
    int t_rows, t_cols;
    const char* s_file;
    int s_rows, s_cols;
} TestCase;

// 印出擷取出對應匹配位置的 T 陣列內容
void print_matched_array(const unsigned char* h_T, int T_cols, int target_r, int target_c, int S_r, int S_c) {
    for (int r = 0; r < S_r; r++) {
        printf("        [ ");
        for (int c = 0; c < S_c; c++) {
            printf("%3d ", h_T[(target_r + r) * T_cols + (target_c + c)]);
        }
        printf("]\n");
    }
}

void run_test_case(TestCase tc) {
    printf("=================================================================================\n");
    printf("[測資 %d] T:(%dx%d) S:(%dx%d)\n", tc.id, tc.t_rows, tc.t_cols, tc.s_rows, tc.s_cols);
    
    unsigned char* h_T = load_matrix(tc.t_file, tc.t_rows, tc.t_cols);
    unsigned char* h_S = load_matrix(tc.s_file, tc.s_rows, tc.s_cols);

    int out_r = tc.t_rows - tc.s_rows + 1;
    int out_c = tc.t_cols - tc.s_cols + 1;
    int out_size = out_r * out_c;

    unsigned char *d_T, *d_S;
    float *d_pcc;
    unsigned int *d_ssd;

    CHECK_CUDA(cudaMalloc(&d_T, tc.t_rows * tc.t_cols * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_S, tc.s_rows * tc.s_cols * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_pcc, out_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ssd, out_size * sizeof(unsigned int)));

    CHECK_CUDA(cudaMemcpy(d_T, h_T, tc.t_rows * tc.t_cols * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_S, h_S, tc.s_rows * tc.s_cols * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpyToSymbol(c_S, h_S, tc.s_rows * tc.s_cols * sizeof(unsigned char)));

    int test_num = 9;
    int block_sizes[test_num][2] = {{16, 16}, {8, 128}, {32, 32}, {128, 8},{64, 16}, {16, 64},{4, 256}, {256, 4}, {16, 32}};
    
    float* h_pcc_out = (float*)malloc(out_size * sizeof(float));
    unsigned int* h_ssd_out = (unsigned int*)malloc(out_size * sizeof(unsigned int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int b = 0; b < test_num; b++) {
        dim3 block(block_sizes[b][0], block_sizes[b][1]);
        dim3 grid((out_c + block.x - 1) / block.x, (out_r + block.y - 1) / block.y);

        printf("---------------------------------------------------------------------------------\n");
        printf("▶ Block Size: %2dx%2d\n", block.x, block.y);

        float total_ms_global = 0, total_ms_optimized = 0;
        // 每種組合重複執行 3 次
        for (int r_run = 1; r_run <= 3; r_run++) {
            // 1. Global Memory Version
            cudaEventRecord(start);
            matchKernel<<<grid, block>>>(d_T, tc.t_rows, tc.t_cols, d_S, tc.s_rows, tc.s_cols, d_pcc, d_ssd);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float ms_global = 0;
            cudaEventElapsedTime(&ms_global, start, stop);
            total_ms_global += ms_global;

            // 2. Optimized Version (Shared T + Constant S)
            size_t shared_mem_size = (block.y + tc.s_rows - 1) * (block.x + tc.s_cols - 1) * sizeof(unsigned char);
            cudaEventRecord(start);
            matchKernelOptimized<<<grid, block, shared_mem_size>>>(d_T, tc.t_rows, tc.t_cols, tc.s_rows, tc.s_cols, d_pcc, d_ssd);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float ms_optimized = 0;
            cudaEventElapsedTime(&ms_optimized, start, stop);
            total_ms_optimized += ms_optimized;

            CHECK_CUDA(cudaMemcpy(h_pcc_out, d_pcc, out_size * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_ssd_out, d_ssd, out_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));

            printf("  [Run %d] Global: %8.4f ms | Opt(T_share, S_const): %8.4f ms\n", 
                   r_run, ms_global, ms_optimized);
        }
        printf("平均時間 - Global: %8.4f ms | Opt(T_share, S_const): %8.4f ms\n", 
               total_ms_global / 3.0f, total_ms_optimized / 3.0f);
    }

    // ========== CPU Sequential Version ==========
    printf("---------------------------------------------------------------------------------\n");
    printf("▶ CPU Sequential Version (No Parallelization)\n");
    float* h_pcc_cpu = (float*)malloc(out_size * sizeof(float));
    unsigned int* h_ssd_cpu = (unsigned int*)malloc(out_size * sizeof(unsigned int));
    
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matchCPU(h_T, tc.t_rows, tc.t_cols, h_S, tc.s_rows, tc.s_cols, h_pcc_cpu, h_ssd_cpu);
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> ms_cpu = stop_cpu - start_cpu;
    printf("  CPU Time: %8.4f ms\n", ms_cpu.count());
    
    float cpu_max1 = -2.0f;
    for (int i = 0; i < out_size; i++) {
        float v = h_pcc_cpu[i];
        if (!isnan(v) && v > cpu_max1) {
            cpu_max1 = v;
        }
    }
    if (cpu_max1 > -2.0f) {
        printf("  [CPU Top 1相似] PCC: %7.4f, 位置: ", cpu_max1);
        int cpu_first_pos = -1;
        for (int i = 0; i < out_size; i++) {
            if (fabs(h_pcc_cpu[i] - cpu_max1) < 1e-4) {
                if (cpu_first_pos == -1) cpu_first_pos = i;
                printf("(%d,%d) ", i / out_c, i % out_c);
            }
        }
        printf("\n");
        if (cpu_first_pos != -1) {
            print_matched_array(h_T, tc.t_cols, cpu_first_pos / out_c, cpu_first_pos % out_c, tc.s_rows, tc.s_cols);
        }
    }

    free(h_pcc_cpu);
    free(h_ssd_cpu);
    // ============================================

    // 在執行完所有 block size 測試後，針對最後一次獲得的結果統計 Top 1 相似
    float max1 = -2.0f;
    
    for (int i = 0; i < out_size; i++) {
        // PCC 處理
        float v = h_pcc_out[i];
        if (!isnan(v)) {
            // 找最大 (最相似)
            if (fabs(v - max1) < 1e-4) {
                // 已存在相同數值
            } else if (v > max1) {
                max1 = v;
            }
        }
    }

    printf("---------------------------------------------------------------------------------\n");
    printf("▶ 匹配結果 (僅顯示 PCC 第一名相似區塊)\n");
    
    // PCC 僅印出 Top 1
    if(max1 > -2.0f) {
        printf("  [Top 1相似] PCC: %7.4f, 位置: ", max1);
        int first_pos = -1;
        for (int i = 0; i < out_size; i++) {
            if (fabs(h_pcc_out[i] - max1) < 1e-4) {
                if (first_pos == -1) first_pos = i;
                printf("(%d,%d) ", i / out_c, i % out_c);
            }
        }
        printf("\n");
        // 輸出對應陣列內容
        if (first_pos != -1) {
            print_matched_array(h_T, tc.t_cols, first_pos / out_c, first_pos % out_c, tc.s_rows, tc.s_cols);
        }
    }

    free(h_T); free(h_S); free(h_pcc_out); free(h_ssd_out);
    cudaFree(d_T); cudaFree(d_S); cudaFree(d_pcc); cudaFree(d_ssd);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    FILE *out_file = freopen("output.txt", "w", stdout);
    if (!out_file) {
        fprintf(stderr, "Failed to open output.txt for writing\n");
        return 1;
    }

    TestCase tests[] = {
        {1, "test data/1/T1_3750_4320.txt", 3750, 4320, "test data/1/S1_3_3.txt", 3, 3},
        {2, "test data/2/T2_7750_1320.txt", 7750, 1320, "test data/2/S2_5_5.txt", 5, 5},
        {3, "test data/3/T3_8140_9925.txt", 8140, 9925, "test data/3/S3_3_3.txt", 3, 3},
        {4, "test data/4/T4_50_50.txt", 50, 50, "test data/4/S4_5_5.txt", 5, 5}
    };

    int num_tests = sizeof(tests) / sizeof(TestCase);
    for (int i = 0; i < num_tests; i++) {
        run_test_case(tests[i]);
    }

    return 0;
}
