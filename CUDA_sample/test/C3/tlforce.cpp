/*
 * bruteforce_multithread.cpp
 * 暴力破解（多執行緒 + Step2 LCG 生成固定排列）
 * SHA256 純 C++ 內嵌實作，無需任何外部函式庫。
 *
 * 編譯（Linux/macOS）：
 *   g++ -O2 -std=c++17 -pthread bruteforce_multithread.cpp -o bruteforce
 *
 * 編譯（Windows MSVC）：
 *   cl /O2 /std:c++17 bruteforce_multithread.cpp
 *
 * 執行：
 *   ./bruteforce
 */

#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <cstring>
#include <cstdint>
#include <cstdio>

// ══════════════════════════════════════════════════════════════════════
//  內嵌 SHA-256（FIPS 180-4，pure C++，無外部依賴）
// ══════════════════════════════════════════════════════════════════════

namespace sha256_impl {

static const uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
    0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
    0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
    0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
    0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
    0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32-n)); }
static inline uint32_t ch (uint32_t e,uint32_t f,uint32_t g){ return (e&f)^(~e&g); }
static inline uint32_t maj(uint32_t a,uint32_t b,uint32_t c){ return (a&b)^(a&c)^(b&c); }
static inline uint32_t sig0(uint32_t a){ return rotr(a,2)^rotr(a,13)^rotr(a,22); }
static inline uint32_t sig1(uint32_t e){ return rotr(e,6)^rotr(e,11)^rotr(e,25); }
static inline uint32_t gam0(uint32_t x){ return rotr(x,7)^rotr(x,18)^(x>>3); }
static inline uint32_t gam1(uint32_t x){ return rotr(x,17)^rotr(x,19)^(x>>10); }

// 對 len 位元組的 msg 計算 SHA256，結果存入 digest[32]
static void sha256(const uint8_t *msg, size_t len, uint8_t digest[32]) {
    uint32_t h[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
    };

    // padding：訊息 + 0x80 + 零填充 + 64-bit big-endian 長度
    uint64_t bit_len = (uint64_t)len * 8;
    size_t padded = ((len + 9 + 63) / 64) * 64;
    uint8_t buf[128] = {};
    memcpy(buf, msg, len);
    buf[len] = 0x80;
    // 長度 big-endian 寫入最後 8 bytes
    for (int i = 0; i < 8; i++)
        buf[padded - 8 + i] = (uint8_t)(bit_len >> (56 - 8*i));

    // 每個 512-bit (64-byte) block
    for (size_t off = 0; off < padded; off += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = ((uint32_t)buf[off+4*i  ] << 24) |
                   ((uint32_t)buf[off+4*i+1] << 16) |
                   ((uint32_t)buf[off+4*i+2] <<  8) |
                    (uint32_t)buf[off+4*i+3];
        for (int i = 16; i < 64; i++)
            w[i] = gam1(w[i-2]) + w[i-7] + gam0(w[i-15]) + w[i-16];

        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],
                 e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = hh + sig1(e) + ch(e,f,g) + K[i] + w[i];
            uint32_t t2 = sig0(a) + maj(a,b,c);
            hh=g; g=f; f=e; e=d+t1;
            d=c;  c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d;
        h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
    }
    for (int i = 0; i < 8; i++) {
        digest[4*i  ] = (uint8_t)(h[i] >> 24);
        digest[4*i+1] = (uint8_t)(h[i] >> 16);
        digest[4*i+2] = (uint8_t)(h[i] >>  8);
        digest[4*i+3] = (uint8_t)(h[i]      );
    }
}

} // namespace sha256_impl

// ── 常數 ──────────────────────────────────────────────────────────────

static const std::string CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz";
static const int CLEN = 36;

// IDX[c] = CHARSET 中的索引（只對合法字元有效）
static int IDX[256];



// ── 初始化 ────────────────────────────────────────────────────────────

void init() {
    memset(IDX, -1, sizeof(IDX));
    for (int i = 0; i < CLEN; i++)
        IDX[(unsigned char)CHARSET[i]] = i;
}

// ── 工具函式 ──────────────────────────────────────────────────────────

static inline uint32_t xor32(uint32_t a, uint32_t b) { return a ^ b; }

static inline uint32_t lcg_next(uint32_t s) {
    return s * 1664525u + 1013904223u;
}

// Fisher-Yates shuffle，回傳 36 元素排列
static void do_shuffle(uint32_t seed, int deck[36]) {
    for (int i = 0; i < 36; i++) deck[i] = i;
    uint32_t s = seed;
    for (int i = 35; i >= 0; i--) {
        s = lcg_next(s);
        int j = s % (i + 1);
        std::swap(deck[i], deck[j]);
    }
}

// key (4 chars) → seed_master
static uint32_t key_to_seed_master(const char key[4]) {
    uint32_t seed = 0;
    for (int i = 0; i < 4; i++)
        seed = seed * 36 + IDX[(unsigned char)key[i]];
    return seed;
}

// SHA256 前 4 bytes → uint32 big-endian
static uint32_t sha_update(uint32_t seed, char c) {
    uint32_t val = seed ^ (uint32_t)(unsigned char)c;
    uint8_t in[4] = {
        (uint8_t)(val >> 24),
        (uint8_t)(val >> 16),
        (uint8_t)(val >>  8),
        (uint8_t)(val      )
    };
    uint8_t digest[32];
    sha256_impl::sha256(in, 4, digest);
    return ((uint32_t)digest[0] << 24) |
           ((uint32_t)digest[1] << 16) |
           ((uint32_t)digest[2] <<  8) |
            (uint32_t)digest[3];
}

// ── Huffman 編碼（兩 queue 線性建樹）────────────────────────────────

// 節點：leaf => char_idx >= 0, left/right = -1
//        internal => char_idx = -1
struct HNode {
    int   weight;
    int   char_idx;   // leaf: CHARSET index; internal: -1
    int   left, right; // child node indices; -1 = none
};

// 回傳 target_char 的 Huffman 編碼（最多 64 bit，存在 code/len）
static void encode_char_fast(uint32_t seed, char target_char,
                              uint64_t &code, int &code_len)
{
    code = 0; code_len = 0;

    // 計算 36 個 weight 並排序（stable sort by weight）
    std::array<std::pair<int,int>,36> items; // (weight, char_idx)
    for (int i = 0; i < CLEN; i++)
        items[i] = { (int)(((uint64_t)(i+1) * seed) % 997), i };
    std::stable_sort(items.begin(), items.end(),
                     [](const auto &a, const auto &b){ return a.first < b.first; });

    // 建立節點池
    // 最多 2*36-1 = 71 個節點
    static thread_local HNode pool[128];
    int node_cnt = 0;

    // Q1: leaf nodes (indices into pool)
    // Q2: internal nodes (indices into pool)
    int Q1[36], Q1h = 0, Q1t = 0;
    int Q2[72], Q2h = 0, Q2t = 0;

    for (int i = 0; i < CLEN; i++) {
        int ni = node_cnt++;
        pool[ni] = { items[i].first, items[i].second, -1, -1 };
        Q1[Q1t++] = ni;
    }

    auto pop_min = [&]() -> int {
        bool q1ok = (Q1h < Q1t), q2ok = (Q2h < Q2t);
        if (q1ok && q2ok)
            return (pool[Q1[Q1h]].weight <= pool[Q2[Q2h]].weight)
                   ? Q1[Q1h++] : Q2[Q2h++];
        if (q1ok) return Q1[Q1h++];
        return Q2[Q2h++];
    };

    while ((Q1t - Q1h) + (Q2t - Q2h) > 1) {
        int a = pop_min(), b = pop_min();
        int ni = node_cnt++;
        pool[ni] = { pool[a].weight + pool[b].weight, -1, a, b };
        Q2[Q2t++] = ni;
    }

    int root = (Q1h < Q1t) ? Q1[Q1h] : Q2[Q2h];
    int target_idx = IDX[(unsigned char)target_char];

    // iterative DFS
    struct Frame { int node; uint64_t pfx; int len; };
    static thread_local Frame stk[128];
    int top = 0;
    stk[top++] = { root, 0, 0 };

    while (top > 0) {
        auto [n, pfx, len] = stk[--top];
        if (pool[n].char_idx >= 0) {  // leaf
            if (pool[n].char_idx == target_idx) {
                code     = (len == 0) ? 0 : pfx;
                code_len = (len == 0) ? 1 : len;  // single-node tree -> '0'
                return;
            }
        } else {
            // push right (1) then left (0) so left is popped first
            stk[top++] = { pool[n].right, (pfx << 1) | 1, len + 1 };
            stk[top++] = { pool[n].left,   pfx << 1,      len + 1 };
        }
    }
}

// ── 單次 key 嘗試 ─────────────────────────────────────────────────────

static bool try_key(const char key[4],
                    const std::string &plaintext,
                    const std::string &target,
                    int n, int tgt_len)
{
    uint32_t sm   = key_to_seed_master(key);
    int      prev = sm % 36;

    int dA[36], dB[36];
    do_shuffle(xor32(sm, 0x1111u), dA);
    do_shuffle(xor32(sm, 0x2222u), dB);

    // Step 1: 撲克牌代換
    char after_poker[4];
    for (int i = 0; i < n; i++) {
        int ic  = IDX[(unsigned char)plaintext[i]];
        int off = (i + 1 + prev) % 36;
        int idx = (ic + off) % 36;
        int oi  = (i % 2 == 0) ? dA[idx] : dB[idx];
        after_poker[i] = CHARSET[oi];
        prev = oi;
    }

    // Step 2: LCG 生成固定排列
    uint32_t s1 = xor32(sm, 0x3333u);
    uint32_t s2 = xor32(sm, 0x6666u);
    uint32_t st = (s1 + s2) % 54251;
    int rands[4];
    for (int i = 0; i < n; i++) {
        st = (12813u * st + 29861u) % 54251;
        rands[i] = (int)st;
    }
    // indexed = argsort(rands)
    int indexed[4] = {0,1,2,3};
    std::stable_sort(indexed, indexed + n,
                     [&](int a, int b){ return rands[a] < rands[b]; });
    // perm[old_pos] = new_pos
    int perm[4];
    for (int new_pos = 0; new_pos < n; new_pos++)
        perm[indexed[new_pos]] = new_pos;

    char after_perm[4];
    for (int old_pos = 0; old_pos < n; old_pos++)
        after_perm[perm[old_pos]] = after_poker[old_pos];

    // Step 3: Score Tree 編碼並比對
    uint32_t seed = 82 + 49;
    for (int i = 0; i < 4; i++) seed += (unsigned char)key[i];

    int pos = 0;
    for (int i = 0; i < n; i++) {
        uint64_t code; int code_len;
        encode_char_fast(seed, after_perm[i], code, code_len);

        if (pos + code_len > tgt_len) return false;

        for (int b = code_len - 1; b >= 0; b--) {
            char expected = ((code >> b) & 1) ? '1' : '0';
            if (target[pos + (code_len - 1 - b)] != expected)
                return false;
        }
        pos += code_len;
        seed = sha_update(seed, after_perm[i]);
    }
    return pos == tgt_len;
}

// ── 共用原子計數器 ────────────────────────────────────────────────────

static std::atomic<uint64_t> g_checked{0};
static std::atomic<bool>     g_stop{false};

struct Result { std::string key; };
static std::mutex          g_result_mutex;
static std::vector<Result> g_results;

// ── Worker ────────────────────────────────────────────────────────────

void worker(const std::string &plaintext,
            const std::string &target_cipher,
            int n, int tgt_len,
            uint64_t start_idx, uint64_t end_idx)
{
    char key[4];

    for (uint64_t idx = start_idx; idx < end_idx; idx++) {
        if (g_stop.load(std::memory_order_relaxed)) break;

        uint64_t tmp = idx;
        for (int i = 3; i >= 0; i--) {
            key[i] = CHARSET[tmp % 36];
            tmp /= 36;
        }

        if (try_key(key, plaintext, target_cipher, n, tgt_len)) {
            {
                std::lock_guard<std::mutex> lk(g_result_mutex);
                g_results.push_back({ std::string(key, 4) });
            }
            g_stop.store(true, std::memory_order_relaxed);
            g_checked.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        g_checked.fetch_add(1, std::memory_order_relaxed);
    }
}

// ── 進度列印 ──────────────────────────────────────────────────────────

void progress_printer(uint64_t total,
                      std::chrono::steady_clock::time_point start)
{
    while (!g_stop.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        uint64_t checked = g_checked.load(std::memory_order_relaxed);
        if (checked == 0) continue;

        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        double pct = (double)checked / total * 100.0;
        double eta = (elapsed / checked) * (total - checked);

        int bar_fill = (int)(40.0 * pct / 100.0);
        std::string bar(bar_fill, '#');
        bar += std::string(40 - bar_fill, '-');

        printf("\r  [%s] %5.1f%%  %9llu/%llu  elapsed=%.0fs  ETA=%.0fs",
               bar.c_str(), pct,
               (unsigned long long)checked,
               (unsigned long long)total,
               elapsed, eta);
        fflush(stdout);
    }
}

// ── 主流程 ────────────────────────────────────────────────────────────

int main() {
    init();

    const std::string plaintext     = "book";
    const std::string target_cipher = "011001001011011011000";
    const int         key_length    = 4;
    const int         num_threads   = (int)std::thread::hardware_concurrency();

    int n       = (int)plaintext.size();
    int tgt_len = (int)target_cipher.size();

    // 驗證輸入
    for (char c : plaintext)
        if (IDX[(unsigned char)c] < 0) {
            fprintf(stderr, "錯誤：'%c' 不合法\n", c); return 1;
        }
    for (char c : target_cipher)
        if (c != '0' && c != '1') {
            fprintf(stderr, "錯誤：密文只能含 0 和 1\n"); return 1;
        }

    uint64_t total = 1;
    for (int i = 0; i < key_length; i++) total *= 36;

    printf("==============================================================\n");
    printf("  暴力破解（C++ 多執行緒 + LCG 固定排列）\n");
    printf("  明文    : %s\n", plaintext.c_str());
    printf("  目標密文: %s\n", target_cipher.c_str());
    printf("  Key 長度: %d 位  Key 空間: %llu 種\n",
           key_length, (unsigned long long)total);
    printf("  執行緒數: %d\n", num_threads);
    printf("==============================================================\n");

    auto start = std::chrono::steady_clock::now();

    // 切割工作區間
    std::vector<std::thread> threads;
    uint64_t chunk = (total + num_threads - 1) / num_threads;
    for (int i = 0; i < num_threads; i++) {
        uint64_t s = (uint64_t)i * chunk;
        uint64_t e = std::min(s + chunk, total);
        if (s >= total) break;
        threads.emplace_back(worker,
                             std::cref(plaintext),
                             std::cref(target_cipher),
                             n, tgt_len, s, e);
    }

    // 進度列印 thread
    std::thread prog(progress_printer, total, start);

    for (auto &t : threads) t.join();
    g_stop.store(true);
    prog.join();

    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    uint64_t checked = g_checked.load();

    printf("\n\n==============================================================\n");
    printf("  完成！共檢查 %llu 個 key，耗時 %.2f 秒\n",
           (unsigned long long)checked, elapsed);

    if (g_results.empty()) {
        printf("  未找到符合的 key\n");
    } else {
        for (const auto &r : g_results)
            printf("  找到 key = '%s'\n", r.key.c_str());
    }
    printf("==============================================================\n");
    return 0;
}