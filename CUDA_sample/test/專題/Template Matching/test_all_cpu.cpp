#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <chrono>
#include <omp.h>

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
              float* pcc_out, unsigned int* ssd_out, int num_threads) 
{
    #pragma omp parallel for num_threads(num_threads) collapse(2)
    for (int r = 0; r < T_r - S_r + 1; r++) {
        for (int c = 0; c < T_c - S_c + 1; c++) {
            float sumX = 0, sumY = 0;
            float sumX2 = 0, sumY2 = 0, sumXY = 0;
            int n = S_r * S_c;

            for (int i = 0; i < S_r; i++) {
                for (int j = 0; j < S_c; j++) {
                    float x = S[i * S_c + j];
                    float y = T[(r + i) * T_c + (c + j)];
                    
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
}

typedef struct {
    int id;
    const char* t_file;
    int t_rows, t_cols;
    const char* s_file;
    int s_rows, s_cols;
} TestCase;

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

    float* h_pcc_cpu = (float*)malloc(out_size * sizeof(float));
    unsigned int* h_ssd_cpu = (unsigned int*)malloc(out_size * sizeof(unsigned int));
    
    printf("---------------------------------------------------------------------------------\n");
    printf("▶ CPU Version (OpenMP Parallelization)\n");
    
    for (int t = 1; t <= 12; t++) {
        auto start_cpu = std::chrono::high_resolution_clock::now();
        matchCPU(h_T, tc.t_rows, tc.t_cols, h_S, tc.s_rows, tc.s_cols, h_pcc_cpu, h_ssd_cpu, t);
        auto stop_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> ms_cpu = stop_cpu - start_cpu;
        printf("  CPU Time (Threads: %2d): %8.4f ms\n", t, ms_cpu.count());
    }
    
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

    free(h_T); free(h_S);
    free(h_pcc_cpu); free(h_ssd_cpu);
}

int main() {
    FILE *out_file = freopen("output_cpu_only.txt", "w", stdout);
    if (!out_file) {
        fprintf(stderr, "Failed to open output_cpu_only.txt for writing\n");
        return 1;
    }

    TestCase tests[] = {
        {1, "test data/1/T1_3750_4320.txt", 3750, 4320, "test data/1/S1_3_3.txt", 3, 3},
        {2, "test data/2/T2_7750_1320.txt", 7750, 1320, "test data/2/S2_5_5.txt", 5, 5},
        {3, "test data/3/T3_8140_9925.txt", 8140, 9925, "test data/3/S3_3_3.txt", 3, 3},
        {4, "test data/4/T4_50_50.txt", 50, 50, "test data/4/S4_5_5.txt", 5, 5},
        {5, "test data/5/T5_5000_5000.txt", 5000, 5000, "test data/5/S5_5_5.txt", 5, 5}
    };

    int num_tests = sizeof(tests) / sizeof(TestCase);
    for (int i = 0; i < num_tests; i++) {
        run_test_case(tests[i]);
    }

    return 0;
}
