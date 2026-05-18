#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <float.h>

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", __FILE__, __line__, err, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 讀取 CSV 檔案。因為資料都是 single digits 0-9 且有逗號，可以使用 fgetc 或 fscanf。
// 為求效率，我們可以讀整個檔案並解析。
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
        // Compute mean of X (S) and Y (window in T)
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

int main(int argc, char** argv) {
    if (argc != 7) {
        printf("Usage: %s <T_file> <T_rows> <T_cols> <S_file> <S_rows> <S_cols>\n", argv[0]);
        return 1;
    }

    const char* t_file = argv[1];
    int t_rows = atoi(argv[2]);
    int t_cols = atoi(argv[3]);
    const char* s_file = argv[4];
    int s_rows = atoi(argv[5]);
    int s_cols = atoi(argv[6]);

    unsigned char* h_T = load_matrix(t_file, t_rows, t_cols);
    unsigned char* h_S = load_matrix(s_file, s_rows, s_cols);

    int out_r = t_rows - s_rows + 1;
    int out_c = t_cols - s_cols + 1;
    int out_size = out_r * out_c;

    unsigned char *d_T, *d_S;
    float *d_pcc;
    unsigned int *d_ssd;

    CHECK_CUDA(cudaMalloc(&d_T, t_rows * t_cols * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_S, s_rows * s_cols * sizeof(unsigned char)));
    CHECK_CUDA(cudaMalloc(&d_pcc, out_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ssd, out_size * sizeof(unsigned int)));

    CHECK_CUDA(cudaMemcpy(d_T, h_T, t_rows * t_cols * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_S, h_S, s_rows * s_cols * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Try multiple block sizes
    int block_sizes[][2] = {{16, 16}, {32, 32}, {16, 32}};
    int num_configs = sizeof(block_sizes) / sizeof(block_sizes[0]);

    float* h_pcc_out = (float*)malloc(out_size * sizeof(float));
    unsigned int* h_ssd_out = (unsigned int*)malloc(out_size * sizeof(unsigned int));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < num_configs; i++) {
        dim3 block(block_sizes[i][0], block_sizes[i][1]);
        dim3 grid((out_c + block.x - 1) / block.x, (out_r + block.y - 1) / block.y);

        cudaEventRecord(start);
        matchKernel<<<grid, block>>>(d_T, t_rows, t_cols, d_S, s_rows, s_cols, d_pcc, d_ssd);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Block Size: %dx%d, Execution Time: %f ms\n", block.x, block.y, milliseconds);
    }
    
    CHECK_CUDA(cudaMemcpy(h_pcc_out, d_pcc, out_size * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ssd_out, d_ssd, out_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    float max_pcc = -2.0f;
    unsigned int min_ssd = 0xFFFFFFFF;

    for (int i = 0; i < out_size; i++) {
        if (h_pcc_out[i] > max_pcc) max_pcc = h_pcc_out[i];
        if (h_ssd_out[i] < min_ssd) min_ssd = h_ssd_out[i];
    }
    
    // Allow slight float error
    printf("PCC Result (Row, Col):\n");
    for (int r = 0; r < out_r; r++) {
        for (int c = 0; c < out_c; c++) {
            int idx = r * out_c + c;
            if (fabs(h_pcc_out[idx] - max_pcc) < 1e-4) {
                printf("(%d,%d)\n", r, c);
            }
        }
    }

    printf("SSD Result (Row, Col):\n");
    for (int r = 0; r < out_r; r++) {
        for (int c = 0; c < out_c; c++) {
            int idx = r * out_c + c;
            if (h_ssd_out[idx] == min_ssd) {
                printf("(%d,%d)\n", r, c);
            }
        }
    }

    free(h_T);
    free(h_S);
    free(h_pcc_out);
    free(h_ssd_out);
    cudaFree(d_T);
    cudaFree(d_S);
    cudaFree(d_pcc);
    cudaFree(d_ssd);

    return 0;
}
