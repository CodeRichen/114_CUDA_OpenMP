/*
 * brute_force.cpp
 * 暴力枚舉金鑰 (0-9 + a-z，長度 1~4)
 *
 * 編譯 (需要 OpenSSL):
 *   g++ -O3 -std=c++17 -pthread brute_force.cpp -lssl -lcrypto -o brute_force
 *
 * 執行:
 *   ./brute_force
 *
 * 注意：update_seed 完整模擬 Python 的 int(sha256_hex, 16)
 *       使用 256-bit 大整數（以 4 個 uint64_t 表示），保證與 Python 結果一致。
 */

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <openssl/sha.h>
#include <functional>

// ─────────────────────────────────────────────
// 常數
// ─────────────────────────────────────────────
static const std::string CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz";
static const int         CSIZE   = 36;
// 1000100111101111001011101011110110101001111101111010111010110101100110100100100011011001011011111010111000101001101111011011100100111
static const std::string TARGET  = "1000100111101111001011101011110110101001111101111010111010110101100110100100100011011001011011111010111000101001101111011011100100111";
static const std::string PLAINTEXT = "book";

// ─────────────────────────────────────────────
// 256-bit 大整數（用於 update_seed）
// 以 4 個 uint64_t big-endian 儲存
// ─────────────────────────────────────────────
struct uint256_t {
    uint64_t hi2, hi1, lo1, lo0; // hi2 最高位

    // 從 32-byte SHA-256 digest（big-endian）建構
    static uint256_t from_digest(const unsigned char* d) {
        uint256_t r;
        auto rd = [&](int offset) -> uint64_t {
            uint64_t v = 0;
            for (int i = 0; i < 8; ++i)
                v = (v << 8) | d[offset + i];
            return v;
        };
        r.hi2 = rd(0);
        r.hi1 = rd(8);
        r.lo1 = rd(16);
        r.lo0 = rd(24);
        return r;
    }

    // XOR
    uint256_t operator^(const uint256_t& o) const {
        return {hi2^o.hi2, hi1^o.hi1, lo1^o.lo1, lo0^o.lo0};
    }
};

// ─────────────────────────────────────────────
// 核心工具函數
// ─────────────────────────────────────────────
inline int char_to_idx(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    return c - 'a' + 10;
}
inline char idx_to_char(int i) { return CHARSET[i]; }

uint64_t key_to_seed_master(const std::string& key) {
    uint64_t seed = 0;
    for (char c : key) seed = seed * 36 + char_to_idx(c);
    return seed;
}

inline uint32_t lcg_next(uint32_t state) {
    return (uint32_t)(state * 1664525ULL + 1013904223ULL);
}

// Fisher-Yates shuffle，回傳 deck[0..35]
void fisher_yates_shuffle(uint64_t seed, int deck[36]) {
    for (int i = 0; i < CSIZE; ++i) deck[i] = i;
    uint32_t state = (uint32_t)(seed & 0xFFFFFFFF);
    for (int i = CSIZE - 1; i >= 0; --i) {
        state = lcg_next(state);
        int j  = state % (i + 1);
        std::swap(deck[i], deck[j]);
    }
}

inline uint32_t xor32(uint64_t a, uint64_t b) {
    return (uint32_t)((a ^ b) & 0xFFFFFFFF);
}

// compute_scores: scores[i] = ((i+1)*seed + 17) % 997，i = index in CHARSET
// 回傳長度 36 的陣列
void compute_scores(uint256_t seed256, uint64_t scores[36]) {
    // Python: ((i+1)*seed + 17) % 997
    // seed 是 256-bit，乘法需要 mod 997
    // 先把 seed256 mod 997
    // 用 Horner: val = ((hi2 * 2^192 + hi1 * 2^128 + lo1 * 2^64 + lo0)) % 997
    // 2^64 mod 997, 2^128 mod 997, 2^192 mod 997 預先計算
    // 2^64 mod 997
    static const uint64_t MOD = 997;
    // 2^64 mod 997
    uint64_t p64;
    {
        // 2^10 = 1024 = 997 + 27 => 27
        // 用快速冪
        uint64_t base = 2, exp = 64, result = 1;
        base %= MOD;
        while (exp > 0) {
            if (exp & 1) result = result * base % MOD;
            base = base * base % MOD;
            exp >>= 1;
        }
        p64 = result; // 2^64 mod 997
    }
    uint64_t p128 = p64 * p64 % MOD;
    uint64_t p192 = p128 * p64 % MOD;

    uint64_t seed_mod =
        (seed256.hi2 % MOD * p192 % MOD
       + seed256.hi1 % MOD * p128 % MOD
       + seed256.lo1 % MOD * p64  % MOD
       + seed256.lo0 % MOD) % MOD;

    for (int i = 0; i < CSIZE; ++i) {
        scores[i] = ((uint64_t)(i + 1) * seed_mod % MOD + 17) % MOD;
    }
}

// ─────────────────────────────────────────────
// Huffman 樹（最小堆，固定 36 個字元）
// ─────────────────────────────────────────────
struct HNode {
    uint64_t score;
    int      counter;
    int      ch;      // -1 = internal
    int      left, right; // index into pool, -1 = none
};

struct HuffPool {
    HNode nodes[72]; // 最多 2*36-1 = 71 個節點
    int   heap[72];
    int   heap_size;
    int   node_cnt;

    void push(int idx) {
        heap[heap_size++] = idx;
        // sift up
        int i = heap_size - 1;
        while (i > 0) {
            int p = (i - 1) / 2;
            HNode& ni = nodes[heap[i]];
            HNode& np = nodes[heap[p]];
            if (ni.score < np.score ||
               (ni.score == np.score && ni.counter < np.counter)) {
                std::swap(heap[i], heap[p]);
                i = p;
            } else break;
        }
    }

    int pop() {
        int top = heap[0];
        heap[0] = heap[--heap_size];
        // sift down
        int i = 0;
        while (true) {
            int l = 2*i+1, r = 2*i+2, smallest = i;
            auto less = [&](int a, int b) {
                return nodes[heap[a]].score < nodes[heap[b]].score ||
                      (nodes[heap[a]].score == nodes[heap[b]].score &&
                       nodes[heap[a]].counter < nodes[heap[b]].counter);
            };
            if (l < heap_size && less(l, smallest)) smallest = l;
            if (r < heap_size && less(r, smallest)) smallest = r;
            if (smallest == i) break;
            std::swap(heap[i], heap[smallest]);
            i = smallest;
        }
        return top;
    }

    void build(uint64_t scores[36]) {
        heap_size = 0; node_cnt = 0;
        for (int i = 0; i < CSIZE; ++i) {
            nodes[node_cnt] = {scores[i], node_cnt, i, -1, -1};
            push(node_cnt++);
        }
        while (heap_size > 1) {
            int a = pop(), b = pop();
            nodes[node_cnt] = {nodes[a].score + nodes[b].score,
                               node_cnt, -1, a, b};
            push(node_cnt++);
        }
    }
};

// 生成指定字元的 Huffman 編碼位元
// 返回編碼字串（0/1）
static void gen_code_for(const HuffPool& hp, int root, int target_ch,
                          std::string& out) {
    // 迭代 DFS
    struct Frame { int node; std::string prefix; };
    // 使用小型 stack
    struct SFrame { int node; int prefix_len; char bit; };
    static thread_local char path[128];
    int depth = 0;
    std::function<bool(int)> dfs = [&](int idx) -> bool {
        const HNode& n = hp.nodes[idx];
        if (n.ch != -1) {
            if (n.ch == target_ch) {
                path[depth] = '\0';
                out.assign(path, depth);
                if (out.empty()) out = "0";
                return true;
            }
            return false;
        }
        path[depth++] = '0';
        if (dfs(n.left)) { --depth; return true; }
        --depth;
        path[depth++] = '1';
        if (dfs(n.right)) { --depth; return true; }
        --depth;
        return false;
    };
    dfs(root);
}

// ─────────────────────────────────────────────
// update_seed：完整 256-bit
// Python: temp = seed ^ ord(char)
//         h = sha256(str(temp))
//         return int(h, 16)
// seed 是 uint256_t
// ─────────────────────────────────────────────
uint256_t update_seed(uint256_t seed256, char ch) {
    // temp = seed256 ^ ord(char)
    // ord(char) 只影響最低 64 bits
    uint256_t temp = seed256;
    temp.lo0 ^= (uint64_t)(unsigned char)ch;

    // str(temp)：Python 的 str(big_int) 是十進位
    // 把 uint256_t 轉成十進位字串
    // 用 __uint128_t 輔助做長除法
    // 先組成 256-bit，再反覆除 10

    // 以 4 個 uint64_t big-endian 組成大整數，轉十進位
    // 用 byte array + 長除法
    uint8_t digits[80]; // 最多 78 位十進位
    int     ndig = 0;

    // 把 uint256_t 存成 big-endian bytes（32 bytes）
    uint64_t parts[4] = {temp.hi2, temp.hi1, temp.lo1, temp.lo0};
    // 轉成 byte array for division
    uint8_t bignum[32];
    for (int i = 0; i < 4; ++i) {
        for (int j = 7; j >= 0; --j)
            bignum[i*8 + j] = (uint8_t)(parts[i] >> ((7-j)*8));
    }
    // 先用 big-endian 正確排列
    for (int i = 0; i < 4; ++i) {
        uint64_t v = parts[i];
        bignum[i*8+0] = (v >> 56) & 0xFF;
        bignum[i*8+1] = (v >> 48) & 0xFF;
        bignum[i*8+2] = (v >> 40) & 0xFF;
        bignum[i*8+3] = (v >> 32) & 0xFF;
        bignum[i*8+4] = (v >> 24) & 0xFF;
        bignum[i*8+5] = (v >> 16) & 0xFF;
        bignum[i*8+6] = (v >>  8) & 0xFF;
        bignum[i*8+7] = (v >>  0) & 0xFF;
    }

    // 長除法：bignum / 10，取餘數
    // 重複直到 bignum == 0
    auto is_zero = [](const uint8_t* b, int len) {
        for (int i = 0; i < len; ++i) if (b[i]) return false;
        return true;
    };
    auto divmod10 = [](uint8_t* b, int len) -> uint8_t {
        uint32_t rem = 0;
        for (int i = 0; i < len; ++i) {
            uint32_t cur = rem * 256 + b[i];
            b[i] = (uint8_t)(cur / 10);
            rem  = cur % 10;
        }
        return (uint8_t)rem;
    };

    if (is_zero(bignum, 32)) {
        digits[ndig++] = 0;
    } else {
        while (!is_zero(bignum, 32)) {
            digits[ndig++] = divmod10(bignum, 32);
        }
    }
    // digits 是 least-significant first，反轉得十進位字串
    std::string dec_str(ndig, '0');
    for (int i = 0; i < ndig; ++i)
        dec_str[ndig - 1 - i] = '0' + digits[i];

    // SHA-256(dec_str)
    unsigned char hash[32];
    SHA256((const unsigned char*)dec_str.data(), dec_str.size(), hash);

    return uint256_t::from_digest(hash);
}

// ─────────────────────────────────────────────
// STEP 1: poker_substitute
// ─────────────────────────────────────────────
std::string poker_substitute(const std::string& text, uint64_t seed_master,
                              int round_num) {
    int prev   = (int)(seed_master % 36);
    uint64_t sA = xor32(seed_master, (uint64_t)0x1111 * round_num);
    uint64_t sB = xor32(seed_master, (uint64_t)0x2222 * round_num);
    int deck_A[36], deck_B[36];
    fisher_yates_shuffle(sA, deck_A);
    fisher_yates_shuffle(sB, deck_B);

    std::string result(text.size(), ' ');
    for (int i = 0; i < (int)text.size(); ++i) {
        int idx_c  = char_to_idx(text[i]);
        int offset = (i + 1 + prev) % 36;
        int idx    = (idx_c + offset) % 36;
        int out_idx = (i % 2 == 0) ? deck_A[idx] : deck_B[idx];
        result[i]  = idx_to_char(out_idx);
        prev       = out_idx;
    }
    return result;
}

// ─────────────────────────────────────────────
// STEP 2: permute
// ─────────────────────────────────────────────
std::string permute(const std::string& text, uint64_t seed_master, int pid) {
    int n = (int)text.size();
    uint32_t seed1 = xor32(seed_master, (uint64_t)pid * 0x3333);
    uint32_t state = seed1;
    std::vector<int> randoms(n);
    for (int i = 0; i < n; ++i) {
        state      = lcg_next(state);
        randoms[i] = state % 10000;
    }
    std::vector<int> sorted_idx(n);
    for (int i = 0; i < n; ++i) sorted_idx[i] = i;
    std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                     [&](int a, int b){ return randoms[a] < randoms[b]; });
    std::string result(n, ' ');
    for (int i = 0; i < n; ++i) result[i] = text[sorted_idx[i]];
    return result;
}

// ─────────────────────────────────────────────
// STEP 3: score_tree_encode
// ─────────────────────────────────────────────
std::string score_tree_encode(const std::string& text, const std::string& key,
                               int round_num) {
    // suffix = f"R{round_num-1}"
    // seed = sum(ord(c) for c in key) + sum(ord(c) for c in suffix) + 1
    std::string suffix = "R" + std::to_string(round_num - 1);
    uint64_t seed_init = 1;
    for (char c : key)    seed_init += (unsigned char)c;
    for (char c : suffix) seed_init += (unsigned char)c;

    // 把 seed_init 包成 uint256_t（只用 lo0）
    uint256_t seed256 = {0, 0, 0, seed_init};

    static thread_local HuffPool hp;
    std::string bits;
    bits.reserve(text.size() * 8);

    for (char ch : text) {
        uint64_t scores[36];
        compute_scores(seed256, scores);
        hp.build(scores);
        int root = hp.heap[0]; // 唯一剩餘的根節點 index
        std::string code;
        gen_code_for(hp, root, char_to_idx(ch), code);
        bits += code;
        seed256 = update_seed(seed256, ch);
    }
    return bits;
}

// ─────────────────────────────────────────────
// 完整加密
// ─────────────────────────────────────────────
std::string encrypt(const std::string& plaintext, const std::string& key) {
    uint64_t sm = key_to_seed_master(key);
    // Round 1
    std::string t = poker_substitute(plaintext, sm, 1);
    t = permute(t, sm, 1);
    t = score_tree_encode(t, key, 1);
    // Round 2
    t = poker_substitute(t, sm, 2);
    t = permute(t, sm, 2);
    t = score_tree_encode(t, key, 2);
    return t;
}

// ─────────────────────────────────────────────
// 多執行緒枚舉
// ─────────────────────────────────────────────
static std::atomic<uint64_t> g_checked{0};
static std::atomic<bool>     g_found{false};
static std::mutex            g_mutex;
static std::vector<std::string> g_results;

// 枚舉指定長度，從 start_idx 到 end_idx（不含）
void worker(int length, uint64_t start_idx, uint64_t end_idx) {
    uint64_t total = 1;
    for (int i = 0; i < length; ++i) total *= CSIZE;

    // 把 start_idx 轉成初始 combo
    std::vector<int> combo(length);
    uint64_t idx = start_idx;
    for (int i = length - 1; i >= 0; --i) {
        combo[i] = idx % CSIZE;
        idx      /= CSIZE;
    }

    std::string key(length, ' ');
    uint64_t local_checked = 0;

    for (uint64_t k = start_idx; k < end_idx; ++k) {
        for (int i = 0; i < length; ++i) key[i] = CHARSET[combo[i]];

        std::string result = encrypt(PLAINTEXT, key);
        if (result == TARGET) {
            std::lock_guard<std::mutex> lk(g_mutex);
            g_results.push_back(key);
            g_found = true;
        }

        ++local_checked;
        if (local_checked % 200 == 0) {
            g_checked.fetch_add(200, std::memory_order_relaxed);
            local_checked = 0;
        }

        // 遞增 combo（最低位優先）
        int carry = 1;
        for (int i = length - 1; i >= 0 && carry; --i) {
            combo[i] += carry;
            if (combo[i] >= CSIZE) { combo[i] = 0; carry = 1; }
            else carry = 0;
        }
    }
    g_checked.fetch_add(local_checked, std::memory_order_relaxed);
}

int main() {
    unsigned int nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 4;

    uint64_t total = 0;
    for (int L = 1; L <= 4; ++L) {
        uint64_t cnt = 1;
        for (int i = 0; i < L; ++i) cnt *= CSIZE;
        total += cnt;
    }

    std::cout << "目標密文: " << TARGET    << "\n";
    std::cout << "明文    : " << PLAINTEXT << "\n";
    std::cout << "總候選數: " << total     << "  (長度 1~4，字元集 0-9a-z)\n";
    std::cout << "執行緒數: " << nthreads  << "\n";
    std::cout << std::string(65, '-') << "\n";

    auto t_start = std::chrono::steady_clock::now();

    // 依長度依序建立工作，按執行緒數切分
    std::vector<std::thread> threads;

    for (int length = 1; length <= 4; ++length) {
        uint64_t cnt = 1;
        for (int i = 0; i < length; ++i) cnt *= CSIZE;

        uint64_t chunk = (cnt + nthreads - 1) / nthreads;
        for (unsigned int t = 0; t < nthreads; ++t) {
            uint64_t s = t * chunk;
            uint64_t e = std::min(s + chunk, cnt);
            if (s >= cnt) break;
            threads.emplace_back(worker, length, s, e);
        }
    }

    // 進度列印執行緒
    std::thread progress_thread([&]() {
        while (!g_found) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            uint64_t checked = g_checked.load(std::memory_order_relaxed);
            auto now  = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t_start).count();
            double speed   = elapsed > 0 ? checked / elapsed : 0;
            double rem     = speed > 0 ? (total - checked) / speed : 1e9;
            double pct     = (double)checked / total * 100.0;
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2);
            oss << "\r" << std::setw(10) << checked
                << "  " << std::setw(6) << pct << "%"
                << "  " << std::setw(12) << (uint64_t)speed << " keys/s"
                << "  剩餘 " << std::setw(8) << (uint64_t)rem << "s";
            std::cout << oss.str() << std::flush;
            if (checked >= total) break;
        }
    });

    for (auto& th : threads) th.join();
    g_found = true; // 讓進度執行緒結束
    progress_thread.join();

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n" << std::string(65, '-') << "\n";
    std::cout << "枚舉完成！總耗時: " << elapsed << "s"
              << "  共檢查: " << total << " 個金鑰\n";
    if (!g_results.empty()) {
        std::cout << "找到 " << g_results.size() << " 個有效金鑰: ";
        for (auto& k : g_results) std::cout << "'" << k << "' ";
        std::cout << "\n";
    } else {
        std::cout << "未找到匹配金鑰。\n";
    }
    std::cout << std::string(65, '-') << "\n";
    return 0;
}