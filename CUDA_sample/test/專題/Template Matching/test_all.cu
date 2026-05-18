#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <float.h>

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

extern __shared__ unsigned char s_S[];
__global__ void matchKernelShared(const unsigned char* T, int T_r, int T_c,
                                  const unsigned char* S, int S_r, int S_c,
                                  float* pcc_out, unsigned int* ssd_out) 
{
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int S_area = S_r * S_c;
    
    // 把 Template S 載入 Shared Memory 中
    for (int i = tid; i < S_area; i += blockDim.x * blockDim.y) {
        s_S[i] = S[i];
    }
    __syncthreads();

    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < T_r - S_r + 1 && c < T_c - S_c + 1) {
        float sumX = 0, sumY = 0;
        int n = S_area;
        for (int i = 0; i < S_r; i++) {
            for (int j = 0; j < S_c; j++) {
                float valX = s_S[i * S_c + j];
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
                float x = s_S[i * S_c + j];
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

// 測資結構體
typedef struct {
    int id;
    const char* t_file;
    int t_rows, t_cols;
    const char* s_file;
    int s_rows, s_cols;
} TestCase;

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

        float total_ms_global = 0, total_ms_shared = 0;
        // 每種組合重複執行 3 次
        for (int r = 1; r <= 3; r++) {
            // Global Memory Version
            cudaEventRecord(start);
            matchKernel<<<grid, block>>>(d_T, tc.t_rows, tc.t_cols, d_S, tc.s_rows, tc.s_cols, d_pcc, d_ssd);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float ms_global = 0;
            cudaEventElapsedTime(&ms_global, start, stop);
            total_ms_global += ms_global;

            // Shared Memory Version
            size_t shared_mem_size = tc.s_rows * tc.s_cols * sizeof(unsigned char);
            cudaEventRecord(start);
            matchKernelShared<<<grid, block, shared_mem_size>>>(d_T, tc.t_rows, tc.t_cols, d_S, tc.s_rows, tc.s_cols, d_pcc, d_ssd);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float ms_shared = 0;
            cudaEventElapsedTime(&ms_shared, start, stop);
            total_ms_shared += ms_shared;

            CHECK_CUDA(cudaMemcpy(h_pcc_out, d_pcc, out_size * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_ssd_out, d_ssd, out_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));

            printf("  [Run %d] Global Time: %8.4f ms  [Run %d(s)] Shared Time: %8.4f ms\n", r, ms_global, r, ms_shared);
        }
        printf("平均時間 - Global: %8.4f ms, Shared: %8.4f ms\n", total_ms_global / 3.0f, total_ms_shared / 3.0f);
    }

    // 在執行完所有 block size 測試後，針對最後一次獲得的結果統計 Top 3 相似與不相似
    float max1 = -2.0f, max2 = -2.0f, max3 = -2.0f;
    float min1 = 2.0f, min2 = 2.0f, min3 = 2.0f;
    
    // SSD 的極值預設值
    unsigned int min_ssd1 = -1, min_ssd2 = -1, min_ssd3 = -1; // 最小的SSD (相似)
    unsigned int max_ssd1 = 0, max_ssd2 = 0, max_ssd3 = 0; // 最大的SSD (不相似)

    for (int i = 0; i < out_size; i++) {
        // PCC 處理
        float v = h_pcc_out[i];
        if (!isnan(v)) {
            // 找最大的前三個 (最相似)
            if (fabs(v - max1) < 1e-4 || fabs(v - max2) < 1e-4 || fabs(v - max3) < 1e-4) {
                // 已存在相同數值
            } else if (v > max1) {
                max3 = max2; max2 = max1; max1 = v;
            } else if (v > max2) {
                max3 = max2; max2 = v;
            } else if (v > max3) {
                max3 = v;
            }

            // 找最小的前三個 (最不相似)
            if (fabs(v - min1) < 1e-4 || fabs(v - min2) < 1e-4 || fabs(v - min3) < 1e-4) {
                // 已存在相同數值
            } else if (v < min1) {
                min3 = min2; min2 = min1; min1 = v;
            } else if (v < min2) {
                min3 = min2; min2 = v;
            } else if (v < min3) {
                min3 = v;
            }
        }

        // SSD 處理
        unsigned int s = h_ssd_out[i];
        // 找最小前三個 (SSD最相似越小越好)
        if (s == min_ssd1 || s == min_ssd2 || s == min_ssd3) {
            // 已存在相同數值
        } else if (s < min_ssd1) {
            min_ssd3 = min_ssd2; min_ssd2 = min_ssd1; min_ssd1 = s;
        } else if (s < min_ssd2) {
            min_ssd3 = min_ssd2; min_ssd2 = s;
        } else if (s < min_ssd3) {
            min_ssd3 = s;
        }

        // 找最大前三個 (SSD最不相似越大越好)
        if (s == max_ssd1 || s == max_ssd2 || s == max_ssd3) {
            // 已存在相同數值
        } else if (s > max_ssd1) {
            max_ssd3 = max_ssd2; max_ssd2 = max_ssd1; max_ssd1 = s;
        } else if (s > max_ssd2) {
            max_ssd3 = max_ssd2; max_ssd2 = s;
        } else if (s > max_ssd3) {
            max_ssd3 = s;
        }
    }

    printf("---------------------------------------------------------------------------------\n");
    printf("▶ 匹配結果 (依照 PCC 相似度)\n");
    
    float top_max[] = {max1, max2, max3};
    for(int rank = 0; rank < 3; rank++) {
        if(top_max[rank] > -2.0f) {
            printf("  [Top %d相似] PCC: %7.4f, 位置: ", rank + 1, top_max[rank]);
            for (int i = 0; i < out_size; i++) {
                if (fabs(h_pcc_out[i] - top_max[rank]) < 1e-4) {
                    printf("(%d,%d) ", i / out_c, i % out_c);
                }
            }
            printf("\n");
        }
    }

    float top_min[] = {min1, min2, min3};
    for(int rank = 0; rank < 3; rank++) {
        if(top_min[rank] < 2.0f) {
            printf("  [Top %d不相似] PCC: %7.4f, 位置: ", rank + 1, top_min[rank]);
            for (int i = 0; i < out_size; i++) {
                if (fabs(h_pcc_out[i] - top_min[rank]) < 1e-4) {
                    printf("(%d,%d) ", i / out_c, i % out_c);
                }
            }
            printf("\n");
        }
    }

    printf("\n▶ 匹配結果 (依照 SSD 誤差)\n");

    unsigned int top_min_ssd[] = {min_ssd1, min_ssd2, min_ssd3};
    for(int rank = 0; rank < 3; rank++) {
        if(top_min_ssd[rank] != -1) {
            printf("  [Top %d相似] SSD: %10u, 位置: ", rank + 1, top_min_ssd[rank]);
            for (int i = 0; i < out_size; i++) {
                if (h_ssd_out[i] == top_min_ssd[rank]) {
                    printf("(%d,%d) ", i / out_c, i % out_c);
                }
            }
            printf("\n");
        }
    }

    unsigned int top_max_ssd[] = {max_ssd1, max_ssd2, max_ssd3};
    for(int rank = 0; rank < 3; rank++) {
        if(top_max_ssd[rank] != 0) {
            printf("  [Top %d不相似] SSD: %10u, 位置: ", rank + 1, top_max_ssd[rank]);
            for (int i = 0; i < out_size; i++) {
                if (h_ssd_out[i] == top_max_ssd[rank]) {
                    printf("(%d,%d) ", i / out_c, i % out_c);
                }
            }
            printf("\n");
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