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

    // 4種Block Size配置
    int block_sizes[4][2] = {{16, 16}, {32, 16}, {16, 32}, {32, 32}};
    
    float* h_pcc_out = (float*)malloc(out_size * sizeof(float));
    unsigned int* h_ssd_out = (unsigned int*)malloc(out_size * sizeof(unsigned int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int b = 0; b < 4; b++) {
        dim3 block(block_sizes[b][0], block_sizes[b][1]);
        dim3 grid((out_c + block.x - 1) / block.x, (out_r + block.y - 1) / block.y);

        printf("---------------------------------------------------------------------------------\n");
        printf("▶ Block Size: %2dx%2d\n", block.x, block.y);

        // 每種組合重複執行 3 次
        for (int r = 1; r <= 3; r++) {
            cudaEventRecord(start);
            matchKernel<<<grid, block>>>(d_T, tc.t_rows, tc.t_cols, d_S, tc.s_rows, tc.s_cols, d_pcc, d_ssd);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            
            CHECK_CUDA(cudaMemcpy(h_pcc_out, d_pcc, out_size * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_ssd_out, d_ssd, out_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));

            float max_pcc = -2.0f;
            unsigned int min_ssd = 0xFFFFFFFF;

            for (int i = 0; i < out_size; i++) {
                if (h_pcc_out[i] > max_pcc) max_pcc = h_pcc_out[i];
                if (h_ssd_out[i] < min_ssd) min_ssd = h_ssd_out[i];
            }
            
            printf("  [Run %d] Time: %8.4f ms | Max PCC: %7.4f, Min SSD: %5u | ", r, milliseconds, max_pcc, min_ssd);
            
            // 輸出結果座標與誤差判斷
            // 由於可能存在多個相同結果，僅印出座標第一筆作為概覽，或者全部印出
            int match_count = 0;
            printf("Pos: ");
            for (int i = 0; i < out_size; i++) {
                if (fabs(h_pcc_out[i] - max_pcc) < 1e-4) {
                    int row = i / out_c;
                    int col = i % out_c;
                    if (match_count < 2) {
                        printf("(%d,%d) ", row, col);
                    }
                    match_count++;
                }
            }
            if (match_count > 2) printf("...及其他共 %d 處", match_count);
            printf("\n");
        }
    }

    free(h_T); free(h_S); free(h_pcc_out); free(h_ssd_out);
    cudaFree(d_T); cudaFree(d_S); cudaFree(d_pcc); cudaFree(d_ssd);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
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