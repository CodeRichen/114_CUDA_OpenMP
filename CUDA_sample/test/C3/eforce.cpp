/*
 * enum_plaintext.cpp
 * 固定 key = "4z1j"，枚舉所有長度=4 的明文（0-9a-z，共 36^4 = 1,679,616）
 * 加密後與 c1 比對，找出明文
 *
 * 編譯:
 *   g++ -O3 -std=c++17 -pthread enum_plaintext.cpp -lssl -lcrypto -o enum_plaintext
 *
 * macOS:
 *   g++ -O3 -std=c++17 -pthread enum_plaintext.cpp \
 *       -I$(brew --prefix openssl)/include \
 *       -L$(brew --prefix openssl)/lib \
 *       -lssl -lcrypto -o enum_plaintext
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
#include <functional>
#include <openssl/sha.h>

// ─────────────────────────────────────────────
// 常數
// ─────────────────────────────────────────────
static const char   CHARSET[]  = "0123456789abcdefghijklmnopqrstuvwxyz";
static const int    CSIZE      = 36;
static const char   KEY[]      = "4z1j";
static const char   TARGET[]   =
    "0011101101101100000111000111010110011011110001111000100101011000"
    "01001110111010001101000011110000101100111111011011110101111";
static const int    PT_LEN     = 4;
static const uint64_t TOTAL    = 36ULL * 36 * 36 * 36; // 1,679,616

// ─────────────────────────────────────────────
// 工具
// ─────────────────────────────────────────────
inline int char_to_idx(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    return c - 'a' + 10;
}
inline char idx_to_char(int i) { return CHARSET[i]; }

uint64_t key_to_seed_master(const char* key, int klen) {
    uint64_t seed = 0;
    for (int i = 0; i < klen; ++i) seed = seed * 36 + char_to_idx(key[i]);
    return seed;
}

inline uint32_t lcg_next(uint32_t state) {
    return (uint32_t)(state * 1664525ULL + 1013904223ULL);
}

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

// ─────────────────────────────────────────────
// 256-bit 整數（update_seed 專用）
// ─────────────────────────────────────────────
struct u256 {
    uint64_t w[4]; // w[0]=最高位 … w[3]=最低位

    static u256 from_u64(uint64_t v) {
        u256 r{}; r.w[3] = v; return r;
    }
    static u256 from_digest(const unsigned char* d) {
        u256 r;
        for (int i = 0; i < 4; ++i) {
            r.w[i] = 0;
            for (int j = 0; j < 8; ++j)
                r.w[i] = (r.w[i] << 8) | d[i*8+j];
        }
        return r;
    }
    u256 xor_lo(uint64_t v) const {
        u256 r = *this; r.w[3] ^= v; return r;
    }
    bool is_zero() const { return !w[0] && !w[1] && !w[2] && !w[3]; }
};

// 256-bit → 十進位字串（長除法）
static std::string u256_to_dec(u256 n) {
    if (n.is_zero()) return "0";
    uint8_t buf[32];
    for (int i = 0; i < 4; ++i)
        for (int j = 7; j >= 0; --j)
            buf[i*8+(7-j)] = (n.w[i] >> (j*8)) & 0xFF;
    // 重排為 big-endian bytes
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j)
            buf[i*8+j] = (n.w[i] >> (56 - j*8)) & 0xFF;
    }
    char digits[80]; int nd = 0;
    auto is_zero_buf = [](const uint8_t* b) {
        for (int i = 0; i < 32; ++i) if (b[i]) return false;
        return true;
    };
    while (!is_zero_buf(buf)) {
        uint32_t rem = 0;
        for (int i = 0; i < 32; ++i) {
            uint32_t cur = rem * 256 + buf[i];
            buf[i] = (uint8_t)(cur / 10);
            rem    = cur % 10;
        }
        digits[nd++] = '0' + (char)rem;
    }
    std::string s(nd, ' ');
    for (int i = 0; i < nd; ++i) s[nd-1-i] = digits[i];
    return s;
}

u256 update_seed(u256 seed, char ch) {
    u256 temp = seed.xor_lo((uint64_t)(unsigned char)ch);
    std::string dec = u256_to_dec(temp);
    unsigned char hash[32];
    SHA256((const unsigned char*)dec.data(), dec.size(), hash);
    return u256::from_digest(hash);
}

// ─────────────────────────────────────────────
// compute_scores：seed mod 997，再算 ((i+1)*s+17)%997
// ─────────────────────────────────────────────
static void compute_scores(u256 seed, uint64_t scores[36]) {
    const uint64_t MOD = 997;
    // 2^64 mod 997（快速冪）
    static const uint64_t p64  = [](){
        uint64_t b=2,e=64,r=1; b%=997;
        while(e){if(e&1)r=r*b%997;b=b*b%997;e>>=1;}return r;}();
    static const uint64_t p128 = p64*p64%MOD;
    static const uint64_t p192 = p128*p64%MOD;

    uint64_t sm = (seed.w[0]%MOD*p192%MOD
                 + seed.w[1]%MOD*p128%MOD
                 + seed.w[2]%MOD*p64 %MOD
                 + seed.w[3]%MOD) % MOD;
    for (int i = 0; i < CSIZE; ++i)
        scores[i] = ((uint64_t)(i+1) * sm % MOD + 17) % MOD;
}

// ─────────────────────────────────────────────
// Huffman（最小堆，固定 36 字元）
// ─────────────────────────────────────────────
struct HNode { uint64_t score; int cnt, ch, left, right; };

struct HPool {
    HNode nd[72]; int heap[72], hsz, ncnt;

    void push(int i) {
        heap[hsz++] = i;
        int x = hsz-1;
        while (x > 0) {
            int p = (x-1)/2;
            auto& a=nd[heap[x]]; auto& b=nd[heap[p]];
            if (a.score < b.score || (a.score==b.score && a.cnt<b.cnt))
                { std::swap(heap[x],heap[p]); x=p; } else break;
        }
    }
    int pop() {
        int top=heap[0]; heap[0]=heap[--hsz];
        int x=0;
        for(;;){
            int l=2*x+1,r=2*x+2,s=x;
            auto less=[&](int a,int b){
                return nd[heap[a]].score<nd[heap[b]].score||
                      (nd[heap[a]].score==nd[heap[b]].score&&nd[heap[a]].cnt<nd[heap[b]].cnt);};
            if(l<hsz&&less(l,s))s=l;
            if(r<hsz&&less(r,s))s=r;
            if(s==x)break; std::swap(heap[x],heap[s]); x=s;
        }
        return top;
    }
    void build(uint64_t scores[36]) {
        hsz=ncnt=0;
        for(int i=0;i<CSIZE;++i){nd[ncnt]={scores[i],ncnt,i,-1,-1};push(ncnt++);}
        while(hsz>1){
            int a=pop(),b=pop();
            nd[ncnt]={nd[a].score+nd[b].score,ncnt,-1,a,b};
            push(ncnt++);
        }
    }
    // 找 target_ch 的編碼，存入 out_bits[]/out_len
    void get_code(int target_ch, char out_bits[36], int& out_len) const {
        char path[36]; int depth=0;
        std::function<bool(int)> dfs=[&](int idx)->bool{
            const HNode& n=nd[idx];
            if(n.ch!=-1){
                if(n.ch!=target_ch)return false;
                memcpy(out_bits,path,depth);
                out_len=depth?depth:1;
                if(!depth){out_bits[0]='0';}
                return true;
            }
            path[depth++]='0'; if(dfs(n.left)){--depth;return true;} --depth;
            path[depth++]='1'; if(dfs(n.right)){--depth;return true;} --depth;
            return false;
        };
        dfs(heap[0]);
    }
};

// ─────────────────────────────────────────────
// 加密三步驟
// ─────────────────────────────────────────────
// Step1: poker_substitute（結果存 out，長度同 in）
static void poker_sub(const char* in, int n, uint64_t sm, int rnd, char* out) {
    int prev   = (int)(sm % 36);
    int deck_A[36], deck_B[36];
    fisher_yates_shuffle(xor32(sm, 0x1111*(uint64_t)rnd), deck_A);
    fisher_yates_shuffle(xor32(sm, 0x2222*(uint64_t)rnd), deck_B);
    for (int i = 0; i < n; ++i) {
        int idx_c   = char_to_idx(in[i]);
        int offset  = (i + 1 + prev) % 36;
        int idx     = (idx_c + offset) % 36;
        int out_idx = (i%2==0) ? deck_A[idx] : deck_B[idx];
        out[i]      = idx_to_char(out_idx);
        prev        = out_idx;
    }
}

// Step2: permute
static void do_permute(const char* in, int n, uint64_t sm, int pid, char* out) {
    uint32_t state = xor32(sm, (uint64_t)pid * 0x3333);
    int randoms[PT_LEN];  // PT_LEN=4 固定
    for (int i = 0; i < n; ++i) {
        state      = lcg_next(state);
        randoms[i] = state % 10000;
    }
    int idx[PT_LEN];
    for (int i = 0; i < n; ++i) idx[i] = i;
    std::stable_sort(idx, idx+n, [&](int a,int b){return randoms[a]<randoms[b];});
    for (int i = 0; i < n; ++i) out[i] = in[idx[i]];
}

// Step3: score_tree_encode → bit string（存入 std::string）
static std::string score_encode(const char* in, int n,
                                 const char* key, int klen, int rnd) {
    // suffix = "R{rnd-1}"
    std::string suffix = "R" + std::to_string(rnd-1);
    uint64_t seed_init = 1;
    for (int i = 0; i < klen; ++i) seed_init += (unsigned char)key[i];
    for (char c : suffix)          seed_init += (unsigned char)c;

    u256 seed = u256::from_u64(seed_init);
    static thread_local HPool hp;
    std::string bits; bits.reserve(n * 10);
    char code[36]; int clen;
    for (int i = 0; i < n; ++i) {
        uint64_t scores[36];
        compute_scores(seed, scores);
        hp.build(scores);
        hp.get_code(char_to_idx(in[i]), code, clen);
        bits.append(code, clen);
        seed = update_seed(seed, in[i]);
    }
    return bits;
}

// 完整加密
static std::string encrypt(const char* pt, int pt_len,
                             const char* key, int klen, uint64_t sm) {
    // Round 1
    char t1[PT_LEN], t2[PT_LEN];
    poker_sub(pt, pt_len, sm, 1, t1);
    do_permute(t1, pt_len, sm, 1, t2);
    std::string bits1 = score_encode(t2, pt_len, key, klen, 1);

    // Round 2  （bits1 當作字元串，長度為 bits1.size()）
    // poker_sub / permute 作用在字元串上，字元集仍是 0-9a-z
    // 但 bits1 只含 '0'/'1'，恰好都在 CHARSET 內
    int b1len = (int)bits1.size();
    std::vector<char> t3(b1len), t4(b1len);
    poker_sub(bits1.data(), b1len, sm, 2, t3.data());
    do_permute(t3.data(), b1len, sm, 2, t4.data());
    std::string bits2 = score_encode(t4.data(), b1len, key, klen, 2);
    return bits2;
}

// ─────────────────────────────────────────────
// 多執行緒枚舉
// ─────────────────────────────────────────────
static std::atomic<uint64_t> g_checked{0};
static std::atomic<bool>     g_found{false};
static std::mutex            g_mtx;
static std::vector<std::string> g_results;

static const std::string TARGET_STR(TARGET);
static const int         KEY_LEN = 4; // strlen("4z1j")

void worker(uint64_t start, uint64_t end) {
    // 預先算好 seed_master（固定 key）
    uint64_t sm = key_to_seed_master(KEY, KEY_LEN);

    // 把 start 展開成初始 combo[4]
    int combo[PT_LEN];
    {
        uint64_t idx = start;
        for (int i = PT_LEN-1; i >= 0; --i) {
            combo[i] = idx % CSIZE; idx /= CSIZE;
        }
    }

    char pt[PT_LEN+1]; pt[PT_LEN] = '\0';
    uint64_t local = 0;

    for (uint64_t k = start; k < end; ++k) {
        for (int i = 0; i < PT_LEN; ++i) pt[i] = CHARSET[combo[i]];

        std::string result = encrypt(pt, PT_LEN, KEY, KEY_LEN, sm);
        if (result == TARGET_STR) {
            std::lock_guard<std::mutex> lk(g_mtx);
            g_results.push_back(std::string(pt, PT_LEN));
            g_found = true;
            // 不 break，找完所有
        }

        ++local;
        if (local % 500 == 0) {
            g_checked.fetch_add(500, std::memory_order_relaxed);
            local = 0;
        }

        // 遞增 combo
        for (int i = PT_LEN-1; i >= 0; --i) {
            if (++combo[i] < CSIZE) break;
            combo[i] = 0;
        }
    }
    g_checked.fetch_add(local, std::memory_order_relaxed);
}

int main() {
    unsigned int nt = std::thread::hardware_concurrency();
    if (nt == 0) nt = 4;

    std::cout << "目標密文: " << TARGET     << "\n";
    std::cout << "金鑰    : " << KEY        << "\n";
    std::cout << "明文長度: " << PT_LEN     << "\n";
    std::cout << "總候選數: " << TOTAL      << "\n";
    std::cout << "執行緒數: " << nt         << "\n";
    std::cout << std::string(65,'-') << "\n";

    auto t0 = std::chrono::steady_clock::now();

    // 切分工作
    std::vector<std::thread> threads;
    uint64_t chunk = (TOTAL + nt - 1) / nt;
    for (unsigned i = 0; i < nt; ++i) {
        uint64_t s = i * chunk;
        uint64_t e = std::min(s + chunk, TOTAL);
        if (s >= TOTAL) break;
        threads.emplace_back(worker, s, e);
    }

    // 進度執行緒
    std::thread prog([&](){
        while (!g_found) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            uint64_t c = g_checked.load(std::memory_order_relaxed);
            auto now   = std::chrono::steady_clock::now();
            double el  = std::chrono::duration<double>(now - t0).count();
            double spd = el > 0 ? c / el : 0;
            double rem = spd > 0 ? (TOTAL - c) / spd : 1e9;
            std::cout << "\r" << std::setw(9) << c
                      << " / " << TOTAL
                      << "  " << std::fixed << std::setprecision(2)
                      << (double)c/TOTAL*100 << "%"
                      << "  " << (uint64_t)spd << " pt/s"
                      << "  剩餘 " << (uint64_t)rem << "s   "
                      << std::flush;
            if (c >= TOTAL) break;
        }
    });

    for (auto& t : threads) t.join();
    g_found = true;
    prog.join();

    double el = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    std::cout << "\n" << std::string(65,'-') << "\n";
    std::cout << "完成！耗時 " << el << "s，共檢查 " << TOTAL << " 個明文\n";
    if (!g_results.empty()) {
        std::cout << "找到 " << g_results.size() << " 個明文:\n";
        for (auto& s : g_results) std::cout << "  '" << s << "'\n";
    } else {
        std::cout << "未找到匹配明文。\n";
    }
    return 0;
}