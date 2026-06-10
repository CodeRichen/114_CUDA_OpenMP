#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <chrono>
#include <omp.h>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

// ─────────────────────────────────────────────
// 資料結構
// ─────────────────────────────────────────────

typedef struct {
    int id;
    const char* t_file;
    int t_rows, t_cols;
    const char* s_file;
    int s_rows, s_cols;
} TestCase;

typedef struct {
    TestCase       tc;
    unsigned char* h_T;
    unsigned char* h_S;
} LoadedCase;

// ─────────────────────────────────────────────
// I/O
// ─────────────────────────────────────────────

unsigned char* load_matrix(const char* filename, int rows, int cols) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    unsigned char* mat = (unsigned char*)malloc(rows * cols * sizeof(unsigned char));
    int count = 0, c;
    while ((c = fgetc(fp)) != EOF && count < rows * cols)
        if (c >= '0' && c <= '9')
            mat[count++] = (unsigned char)(c - '0');
    fclose(fp);
    return mat;
}

// ─────────────────────────────────────────────
// 計算核心
// ─────────────────────────────────────────────

void matchCPU(const unsigned char* T, int T_r, int T_c,
              const unsigned char* S, int S_r, int S_c,
              float* pcc_out, unsigned int* ssd_out, int num_threads)
{
    const int out_r = T_r - S_r + 1;
    const int out_c = T_c - S_c + 1;
    const int n     = S_r * S_c;

    float sumX = 0, sumX2 = 0;
    for (int i = 0; i < n; i++) {
        float x = S[i];
        sumX  += x;
        sumX2 += x * x;
    }
    const float meanX = sumX / n;
    const float denX  = sumX2 - n * meanX * meanX;

    float* Sf = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) Sf[i] = (float)S[i];

    #pragma omp parallel for num_threads(num_threads) collapse(2)
    for (int r = 0; r < out_r; r++) {
        for (int c = 0; c < out_c; c++) {
            float sumY = 0, sumY2 = 0, sumXY = 0;

            for (int i = 0; i < S_r; i++) {
                for (int j = 0; j < S_c; j++) {
                    float x = Sf[i * S_c + j];
                    float y = (float)T[(r + i) * T_c + (c + j)];
                    sumY  += y;
                    sumY2 += y * y;
                    sumXY += x * y;
                }
            }

            const float meanY = sumY / n;
            const float denY  = sumY2 - n * meanY * meanY;
            const float num   = sumXY - n * meanX * meanY;

            const unsigned int ssd = (unsigned int)(sumX2 + sumY2 - 2.0f * sumXY);

            float pcc = 0.0f;
            if (denX > 0.0f && denY > 0.0f)
                pcc = num / (sqrtf(denX) * sqrtf(denY));

            const int idx = r * out_c + c;
            pcc_out[idx]  = pcc;
            ssd_out[idx]  = ssd;
        }
    }

    free(Sf);
}

// ─────────────────────────────────────────────
// 輸出工具
// ─────────────────────────────────────────────

void print_matched_array(const unsigned char* h_T, int T_cols,
                         int target_r, int target_c, int S_r, int S_c) {
    for (int r = 0; r < S_r; r++) {
        printf("        [ ");
        for (int c = 0; c < S_c; c++)
            printf("%3d ", h_T[(target_r + r) * T_cols + (target_c + c)]);
        printf("]\n");
    }
}

// ─────────────────────────────────────────────
// 消費者：執行計算（每個 thread 數跑 3 次取平均）
// ─────────────────────────────────────────────

#define REPEAT 3   // ← 每個 thread 數重複量測次數

void process_loaded_case(const LoadedCase& lc) {
    const TestCase& tc = lc.tc;
    const unsigned char* h_T = lc.h_T;
    const unsigned char* h_S = lc.h_S;

    printf("=================================================================================\n");
    printf("[測資 %d] T:(%dx%d) S:(%dx%d)\n", tc.id, tc.t_rows, tc.t_cols, tc.s_rows, tc.s_cols);

    const int out_r    = tc.t_rows - tc.s_rows + 1;
    const int out_c    = tc.t_cols - tc.s_cols + 1;
    const int out_size = out_r * out_c;

    float*        h_pcc_cpu = (float*)       malloc(out_size * sizeof(float));
    unsigned int* h_ssd_cpu = (unsigned int*)malloc(out_size * sizeof(unsigned int));

    printf("---------------------------------------------------------------------------------\n");
    printf("▶ CPU Version (OpenMP Parallelization)  [每個 Thread 數重複 %d 次取平均]\n", REPEAT);

    for (int t = 1; t <= 12; t++) {
        float total_ms = 0.0f;

        for (int r = 0; r < REPEAT; r++) {
            auto start = std::chrono::high_resolution_clock::now();
            matchCPU(h_T, tc.t_rows, tc.t_cols,
                     h_S, tc.s_rows, tc.s_cols,
                     h_pcc_cpu, h_ssd_cpu, t);
            auto stop = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float, std::milli> ms = stop - start;
            total_ms += ms.count();
        }

        float avg_ms = total_ms / REPEAT;
        printf("  CPU Time (Threads: %2d): %8.4f ms  (avg of %d runs)\n", t, avg_ms, REPEAT);
    }

    // 最後一次執行結果用於找最佳位置
    float cpu_max = -2.0f;
    for (int i = 0; i < out_size; i++) {
        float v = h_pcc_cpu[i];
        if (!isnan(v) && v > cpu_max) cpu_max = v;
    }
    if (cpu_max > -2.0f) {
        printf("  [CPU Top 1相似] PCC: %7.4f, 位置: ", cpu_max);
        int first_pos = -1;
        for (int i = 0; i < out_size; i++) {
            if (fabsf(h_pcc_cpu[i] - cpu_max) < 1e-4f) {
                if (first_pos == -1) first_pos = i;
                printf("(%d,%d) ", i / out_c, i % out_c);
            }
        }
        printf("\n");
        if (first_pos != -1)
            print_matched_array(h_T, tc.t_cols,
                                first_pos / out_c, first_pos % out_c,
                                tc.s_rows, tc.s_cols);
    }

    free(h_pcc_cpu);
    free(h_ssd_cpu);
}

// ─────────────────────────────────────────────
// 生產者-消費者共享緩衝區
// ─────────────────────────────────────────────

static std::queue<LoadedCase>  g_queue;
static std::mutex              g_mutex;
static std::condition_variable g_cv;
static bool                    g_producer_done = false;

void producer_thread_func(const TestCase* tests, int num_tests) {
    for (int i = 0; i < num_tests; i++) {
        const TestCase& tc = tests[i];

        unsigned char* h_T = load_matrix(tc.t_file, tc.t_rows, tc.t_cols);
        unsigned char* h_S = load_matrix(tc.s_file, tc.s_rows, tc.s_cols);

        LoadedCase lc;
        lc.tc  = tc;
        lc.h_T = h_T;
        lc.h_S = h_S;

        { std::unique_lock<std::mutex> lock(g_mutex); g_queue.push(lc); }
        g_cv.notify_one();
    }

    { std::unique_lock<std::mutex> lock(g_mutex); g_producer_done = true; }
    g_cv.notify_all();
}

// ─────────────────────────────────────────────
// main
// ─────────────────────────────────────────────

int main() {
    FILE* out_file = freopen("output_cpu_only.txt", "w", stdout);
    if (!out_file) {
        fprintf(stderr, "Failed to open output_cpu_only.txt for writing\n");
        return 1;
    }

    TestCase tests[] = {
        {1, "test data/1/T1_3750_4320.txt",  3750, 4320, "test data/1/S1_3_3.txt", 3, 3},
        {2, "test data/2/T2_7750_1320.txt",  7750, 1320, "test data/2/S2_5_5.txt", 5, 5},
        {3, "test data/3/T3_8140_9925.txt",  8140, 9925, "test data/3/S3_3_3.txt", 3, 3},
        {4, "test data/4/T4_50_50.txt",        50,   50, "test data/4/S4_5_5.txt", 5, 5},
        {5, "test data/5/T5_5000_5000.txt",  5000, 5000, "test data/5/S5_5_5.txt", 5, 5}
    };
    const int num_tests = (int)(sizeof(tests) / sizeof(TestCase));

    std::thread producer(producer_thread_func, tests, num_tests);

    int consumed = 0;
    while (consumed < num_tests) {
        LoadedCase lc;
        {
            std::unique_lock<std::mutex> lock(g_mutex);
            g_cv.wait(lock, [] { return !g_queue.empty() || g_producer_done; });
            if (g_queue.empty()) break;
            lc = g_queue.front();
            g_queue.pop();
        }
        process_loaded_case(lc);
        free(lc.h_T);
        free(lc.h_S);
        consumed++;
    }

    producer.join();
    return 0;
}