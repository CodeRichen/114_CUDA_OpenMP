#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <array>
#include <thread>
#include <atomic>
#include <mutex>
#include <map>
#include <openssl/sha.h>

using namespace std;

const string CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz";
int IDX[256];

void init_idx() {
    for (int i = 0; i < 36; ++i)
        IDX[(unsigned char)CHARSET[i]] = i;
}

struct Seed256 {
    array<uint32_t, 8> data;

    void update(char c) {
        data[7] ^= static_cast<unsigned char>(c);
        string dec_str = to_decimal_string();
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char*>(dec_str.c_str()), dec_str.length(), hash);
        for (int i = 0; i < 8; ++i) {
            data[i] = (static_cast<uint32_t>(hash[i * 4]) << 24) |
                      (static_cast<uint32_t>(hash[i * 4 + 1]) << 16) |
                      (static_cast<uint32_t>(hash[i * 4 + 2]) << 8) |
                      (static_cast<uint32_t>(hash[i * 4 + 3]));
        }
    }

    string to_decimal_string() const {
        vector<uint32_t> dec;
        dec.push_back(0);
        for (int i = 0; i < 8; ++i) {
            uint32_t val = data[i];
            for (int bit = 31; bit >= 0; --bit) {
                int carry = (val >> bit) & 1;
                for (size_t j = 0; j < dec.size(); ++j) {
                    uint64_t cur = static_cast<uint64_t>(dec[j]) * 2 + carry;
                    dec[j] = cur % 10;
                    carry = cur / 10;
                }
                while (carry > 0) {
                    dec.push_back(carry % 10);
                    carry /= 10;
                }
            }
        }
        string s = "";
        for (auto it = dec.rbegin(); it != dec.rend(); ++it) s += to_string(*it);
        return s;
    }

    uint32_t get_mod_997() const {
        uint64_t rem = 0;
        for (int i = 0; i < 8; ++i) rem = ((rem << 32) + data[i]) % 997;
        return static_cast<uint32_t>(rem);
    }
};

struct Node {
    int left = -1, right = -1;
    string chars = "";
};

string encode_char_fast(const Seed256& seed_obj, char target_char) {
    uint32_t seed_mod = seed_obj.get_mod_997();
    vector<pair<uint32_t, string>> items(36);
    for (int i = 0; i < 36; ++i)
        items[i] = {((i + 1) * seed_mod + 17) % 997, string(1, CHARSET[i])};
    sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second < b.second;
    });

    vector<pair<uint32_t, int>> Q1, Q2;
    vector<Node> tree; tree.reserve(72);
    for (int i = 0; i < 36; ++i) {
        Node n; n.chars = items[i].second;
        tree.push_back(n);
        Q1.push_back({items[i].first, i});
    }

    int q1_head = 0, q2_head = 0;
    auto pop_min = [&]() {
        if (q1_head < (int)Q1.size() && q2_head < (int)Q2.size())
            return Q1[q1_head].first <= Q2[q2_head].first ? Q1[q1_head++] : Q2[q2_head++];
        if (q1_head < (int)Q1.size()) return Q1[q1_head++];
        return Q2[q2_head++];
    };

    while ((Q1.size() - q1_head) + (Q2.size() - q2_head) > 1) {
        auto [wa, na] = pop_min();
        auto [wb, nb] = pop_min();
        Node new_node;
        new_node.left = na; new_node.right = nb;
        new_node.chars = tree[na].chars + tree[nb].chars;
        int new_idx = tree.size();
        tree.push_back(new_node);
        Q2.push_back({wa + wb, new_idx});
    }

    int root = (q1_head < (int)Q1.size()) ? Q1[q1_head].second : Q2[q2_head].second;
    string code = "";
    int curr = root;
    while (tree[curr].left != -1) {
        int l = tree[curr].left, r = tree[curr].right;
        if (tree[l].chars.find(target_char) != string::npos) { code += '0'; curr = l; }
        else { code += '1'; curr = r; }
    }
    return code.empty() ? "0" : code;
}

vector<vector<int>> ALL_PERMS;
void init_perms() {
    vector<int> p = {0, 1, 2, 3};
    do { ALL_PERMS.push_back(p); } while (next_permutation(p.begin(), p.end()));
}

inline uint32_t lcg_next(uint32_t s) { return s * 1664525 + 1013904223; }

vector<int> shuffle_deck(uint32_t s) {
    vector<int> deck(36);
    iota(deck.begin(), deck.end(), 0);
    for (int i = 35; i >= 0; --i) {
        s = lcg_next(s);
        int j = s % (i + 1);
        swap(deck[i], deck[j]);
    }
    return deck;
}

// ── 統計結構（執行緒本地，最後合併）────────────────────────────────────
struct Stats {
    uint64_t count = 0;
    uint64_t total_len = 0;
    int min_len = 1000;
    int max_len = 0;
    map<int, uint64_t> len_freq;   // 長度 → 出現次數（分佈直方圖）

    void record(int len) {
        count++;
        total_len += len;
        if (len < min_len) min_len = len;
        if (len > max_len) max_len = len;
        len_freq[len]++;
    }

    void merge(const Stats& other) {
        count     += other.count;
        total_len += other.total_len;
        if (other.min_len < min_len) min_len = other.min_len;
        if (other.max_len > max_len) max_len = other.max_len;
        for (const auto& [l, f] : other.len_freq)
            len_freq[l] += f;
    }
};

// ── 對單一 key 跑完所有排列，回傳各排列的密文長度 ─────────────────────
void measure_key(const string& plaintext, const string& key, int n, Stats& local_stats) {
    uint64_t sm = 0;
    for (char c : key) sm = sm * 36 + IDX[(unsigned char)c];
    int prev = sm % 36;

    vector<int> dA = shuffle_deck((sm ^ 0x1111) & 0xFFFFFFFF);
    vector<int> dB = shuffle_deck((sm ^ 0x2222) & 0xFFFFFFFF);

    string after_poker = "";
    for (int i = 0; i < n; ++i) {
        int ic = IDX[(unsigned char)plaintext[i]];
        int off = (i + 1 + prev) % 36;
        int idx = (ic + off) % 36;
        int oi = (i % 2 == 0) ? dA[idx] : dB[idx];
        after_poker += CHARSET[oi];
        prev = oi;
    }

    uint64_t seed_base_val = 82 + 49;
    for (char c : key) seed_base_val += static_cast<unsigned char>(c);

    for (const auto& perm : ALL_PERMS) {
        string after_perm = "";
        for (int i = 0; i < n; ++i) after_perm += after_poker[perm[i]];

        Seed256 seed_obj{};
        seed_obj.data[7] = seed_base_val;

        // 完整跑完 4 個字元，累加長度
        int total_len = 0;
        for (char char_item : after_perm) {
            total_len += (int)encode_char_fast(seed_obj, char_item).length();
            seed_obj.update(char_item);
        }

        local_stats.record(total_len);
    }
}

// ── 針對指定長度 key 的統計搜尋 ──────────────────────────────────────
Stats bruteforce_single_len(const string& plaintext, int key_length) {
    uint64_t total = 1;
    for (int i = 0; i < key_length; ++i) total *= 36;
    int n = (int)plaintext.length();

    cout << "\n" << string(62, '-') << "\n";
    cout << "  Key 長度 = " << key_length << "  (共 " << total << " 種 key × 24 排列 = "
         << total * 24 << " 次加密)\n";
    cout << string(62, '-') << "\n";

    atomic<uint64_t> checked(0);
    atomic<bool> done(false);
    auto start_time = chrono::high_resolution_clock::now();

    thread progress_thread([&]() {
        while (!done) {
            this_thread::sleep_for(chrono::milliseconds(300));
            uint64_t cur = checked.load();
            if (cur == 0) continue;
            auto now = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(now - start_time).count();
            double pct = (double)cur / total * 100.0;
            double eta = elapsed / cur * (total - cur);
            int bar_width = 38, pos = (int)(bar_width * pct / 100.0);
            cout << "\r  [";
            for (int i = 0; i < bar_width; ++i) cout << (i < pos ? "█" : "░");
            cout << "] " << fixed << setprecision(1) << setw(5) << pct << "%"
                 << "  " << cur << "/" << total
                 << "  ETA=" << (int)eta << "s" << flush;
        }
    });

    unsigned int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 2;
    vector<thread> workers;
    vector<Stats> thread_stats(num_threads);
    uint64_t chunk_size = total / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
        uint64_t start_idx = t * chunk_size;
        uint64_t end_idx = (t == num_threads - 1) ? total : (t + 1) * chunk_size;

        workers.push_back(thread([&, t, start_idx, end_idx, key_length]() {
            for (uint64_t idx = start_idx; idx < end_idx; ++idx) {
                string current_key(key_length, '0');
                uint64_t temp = idx;
                for (int i = key_length - 1; i >= 0; --i) {
                    current_key[i] = CHARSET[temp % 36];
                    temp /= 36;
                }
                measure_key(plaintext, current_key, n, thread_stats[t]);
                checked++;
            }
        }));
    }

    for (auto& w : workers) w.join();
    done = true;
    if (progress_thread.joinable()) progress_thread.join();

    // 合併所有執行緒結果
    Stats merged;
    for (auto& s : thread_stats) merged.merge(s);

    double elapsed = chrono::duration<double>(chrono::high_resolution_clock::now() - start_time).count();
    cout << "\n  長度 " << key_length << " 完成，耗時 " << fixed << setprecision(2) << elapsed << "s\n";
    return merged;
}

// ── 印出統計報告 ──────────────────────────────────────────────────────
void print_stats(const Stats& s, const string& label, int target_len) {
    if (s.count == 0) { cout << "  （無資料）\n"; return; }

    double avg = (double)s.total_len / s.count;

    cout << "\n  【" << label << "】\n";
    cout << "  樣本數  : " << s.count << " 次加密\n";
    cout << "  最短長度: " << s.min_len << " bits\n";
    cout << "  最長長度: " << s.max_len << " bits\n";
    cout << "  平均長度: " << fixed << setprecision(4) << avg << " bits\n";
    cout << "  目標密文: " << target_len << " bits\n";


    // 長度分佈直方圖
    cout << "\n  長度分佈直方圖：\n";
    uint64_t max_freq = 0;
    for (const auto& [l, f] : s.len_freq) max_freq = max(max_freq, f);

    for (const auto& [l, f] : s.len_freq) {
        int bar = (int)((double)f / max_freq * 40);
        cout << "  " << setw(4) << l << " bits | ";
        for (int i = 0; i < bar; ++i) cout << "█";
        cout << " " << f;
        if (l == target_len) cout << "  ← 目標長度";
        cout << "\n";
    }
}

void bruteforce(const string& plaintext, const string& target_cipher, int max_key_len = 4) {
    int target_len = (int)target_cipher.length();

    uint64_t grand_total = 0;
    for (int l = 1; l <= max_key_len; ++l) {
        uint64_t t = 1;
        for (int i = 0; i < l; ++i) t *= 36;
        grand_total += t * 24;
    }

    cout << string(62, '=') << "\n";
    cout << "  Huffman 加密長度統計分析\n";
    cout << "  明文    : " << plaintext << "  (" << plaintext.length() << " 字元)\n";
    cout << "  目標密文: " << target_cipher << "  (" << target_len << " bits)\n";
    cout << "  Key 範圍: 長度 1 ~ " << max_key_len << "\n";
    cout << "  總加密次數: " << grand_total << " 次\n";
    cout << string(62, '=') << "\n";

    auto global_start = chrono::high_resolution_clock::now();
    Stats global_stats;

    for (int key_len = 1; key_len <= max_key_len; ++key_len) {
        Stats s = bruteforce_single_len(plaintext, key_len);
        print_stats(s, "Key 長度 = " + to_string(key_len), target_len);
        global_stats.merge(s);
    }

    double total_elapsed = chrono::duration<double>(
        chrono::high_resolution_clock::now() - global_start).count();

    cout << "\n" << string(62, '=') << "\n";
    cout << "  全域統計（Key 長度 1 ~ " << max_key_len << " 合計）\n";
    cout << string(62, '=') << "\n";
    print_stats(global_stats, "全域", target_len);
    cout << "\n  總耗時：" << fixed << setprecision(2) << total_elapsed << " 秒\n";
    cout << string(62, '=') << "\n";
}

int main() {
    init_idx();
    init_perms();

    string plaintext = "book";
    string target_cipher = "01110110110110000011";

    bruteforce(plaintext, target_cipher, 4);
    return 0;
}