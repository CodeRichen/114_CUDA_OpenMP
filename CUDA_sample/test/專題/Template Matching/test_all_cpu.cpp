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

// 生產者讀完一筆測資後，打包成 LoadedCase 放進 queue
typedef struct {
    TestCase    tc;
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
// 計算核心（與原版相同）
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
// 消費者：拿到已載入資料後執行計算並輸出
// ─────────────────────────────────────────────

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
    printf("▶ CPU Version (OpenMP Parallelization)\n");

    for (int t = 1; t <= 12; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        matchCPU(h_T, tc.t_rows, tc.t_cols,
                 h_S, tc.s_rows, tc.s_cols,
                 h_pcc_cpu, h_ssd_cpu, t);
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> ms = stop - start;
        printf("  CPU Time (Threads: %2d): %8.4f ms\n", t, ms.count());
    }

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

static std::queue<LoadedCase>   g_queue;
static std::mutex               g_mutex;
static std::condition_variable  g_cv;
static bool                     g_producer_done = false;

// ─────────────────────────────────────────────
// 生產者執行緒：依序讀所有測資並推入 queue
// ─────────────────────────────────────────────

void producer_thread_func(const TestCase* tests, int num_tests) {
    for (int i = 0; i < num_tests; i++) {
        const TestCase& tc = tests[i];

        // ── I/O（與計算重疊的部分）──
        unsigned char* h_T = load_matrix(tc.t_file, tc.t_rows, tc.t_cols);
        unsigned char* h_S = load_matrix(tc.s_file, tc.s_rows, tc.s_cols);

        LoadedCase lc;
        lc.tc  = tc;
        lc.h_T = h_T;
        lc.h_S = h_S;

        {
            std::unique_lock<std::mutex> lock(g_mutex);
            g_queue.push(lc);
        }
        g_cv.notify_one();   // 通知消費者有新資料
    }

    // 通知消費者生產者已結束
    {
        std::unique_lock<std::mutex> lock(g_mutex);
        g_producer_done = true;
    }
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

    // 啟動生產者執行緒
    std::thread producer(producer_thread_func, tests, num_tests);

    // 消費者（主執行緒）：持續從 queue 取資料並計算
    int consumed = 0;
    while (consumed < num_tests) {
        LoadedCase lc;

        {
            std::unique_lock<std::mutex> lock(g_mutex);
            // 等待：queue 非空 或 生產者已結束
            g_cv.wait(lock, [] {
                return !g_queue.empty() || g_producer_done;
            });

            if (g_queue.empty()) break;   // 生產者結束且 queue 空 → 離開

            lc = g_queue.front();
            g_queue.pop();
        }

        // ── 計算（與下一筆的 I/O 重疊）──
        process_loaded_case(lc);

        free(lc.h_T);
        free(lc.h_S);
        consumed++;
    }

    producer.join();
    return 0;
}