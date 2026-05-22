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
#include <openssl/sha.h>

using namespace std;

const string CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz";
int IDX[256];

void init_idx() {
    for (int i = 0; i < 36; ++i) {
        IDX[(unsigned char)CHARSET[i]] = i;
    }
}

// ── 256-bit 大數與 SHA256 狀態結構（完全保留原版） ──────────────────
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
        for (int i = 0; i < 8; ++i) {
            rem = ((rem << 32) + data[i]) % 997;
        }
        return static_cast<uint32_t>(rem);
    }
};

// ── 高效線性 Huffman 編碼（完全保留原版） ──────────────────────────
struct Node {
    int left = -1;
    int right = -1;
    string chars = "";
};

string encode_char_fast(const Seed256& seed_obj, char target_char) {
    uint32_t seed_mod = seed_obj.get_mod_997();

    vector<pair<uint32_t, string>> items(36);
    for (int i = 0; i < 36; ++i) {
        items[i] = {((i + 1) * seed_mod + 17) % 997, string(1, CHARSET[i])};
    }

    sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });

    vector<pair<uint32_t, int>> Q1, Q2;
    vector<Node> tree;
    tree.reserve(72);

    for (int i = 0; i < 36; ++i) {
        Node n; n.chars = items[i].second;
        tree.push_back(n);
        Q1.push_back({items[i].first, i});
    }

    int q1_head = 0, q2_head = 0;
    auto pop_min = [&]() {
        if (q1_head < (int)Q1.size() && q2_head < (int)Q2.size()) {
            if (Q1[q1_head].first <= Q2[q2_head].first) return Q1[q1_head++];
            else return Q2[q2_head++];
        }
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
        int l = tree[curr].left;
        int r = tree[curr].right;
        if (tree[l].chars.find(target_char) != string::npos) { code += '0'; curr = l; }
        else { code += '1'; curr = r; }
    }
    return code.empty() ? "0" : code;
}

// ── 排列生成與洗牌（完全保留原版） ──────────────────────────────────
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

// ── 依據原本的所有規則，計算單一 Key 在 24 種排列下的最大編碼長度 ────
int get_key_max_length(const string& plaintext, const string& key, int n, vector<int>& best_perm) {
    uint64_t sm = 0;
    for (char c : key) sm = sm * 36 + IDX[(unsigned char)c];
    int prev = sm % 36;

    // 原版撲克洗牌規則
    vector<int> dA = shuffle_deck((sm ^ 0x1111) & 0xFFFFFFFF);
    vector<int> dB = shuffle_deck((sm ^ 0x2222) & 0xFFFFFFFF);

    // 原版密文前置字元置換規則
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

    int max_len_across_perms = 0;

    // 遍歷 24 種排列，找出哪一種排列能讓 4 個字元加總的霍夫曼編碼最長
    for (const auto& perm : ALL_PERMS) {
        string after_perm = "";
        for (int i = 0; i < n; ++i) after_perm += after_poker[perm[i]];

        Seed256 seed_obj{};
        seed_obj.data[7] = seed_base_val;

        int current_perm_len = 0;
        for (char char_item : after_perm) {
            string code = encode_char_fast(seed_obj, char_item);
            current_perm_len += code.length();
            seed_obj.update(char_item); // 編碼完立即更新 Seed 狀態，完全符合原版
        }

        if (current_perm_len > max_len_across_perms) {
            max_len_across_perms = current_perm_len;
            best_perm = perm;
        }
    }
    return max_len_across_perms;
}

// ── 針對指定長度 key 的爆破（僅追蹤全域最長的 Key） ──────────────────
void bruteforce_single_len(
    const string& plaintext,
    int key_length,
    string& global_best_key,
    int& global_max_len,
    vector<int>& global_best_perm,
    mutex& global_mtx)
{
    uint64_t total = 1;
    for (int i = 0; i < key_length; ++i) total *= 36;

    cout << "\n" << string(62, '-') << "\n";
    cout << "  搜尋 Key 長度 = " << key_length << "  (共 " << total << " 種)\n";
    cout << string(62, '-') << "\n";

    atomic<uint64_t> checked(0);
    atomic<bool> done(false);
    auto start_time = chrono::high_resolution_clock::now();

    // 進度條
    thread progress_thread([&]() {
        while (!done) {
            this_thread::sleep_for(chrono::milliseconds(500));
            uint64_t cur = checked.load();
            if (cur == 0) continue;

            auto now = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(now - start_time).count();
            double pct = (double)cur / total * 100.0;
            double eta = elapsed / cur * (total - cur);

            int bar_width = 30;
            int pos = (int)(bar_width * pct / 100.0);

            cout << "\r  [";
            for (int i = 0; i < bar_width; ++i) cout << (i < pos ? "█" : "░");
            cout << "] " << fixed << setprecision(1) << setw(5) << pct << "%  "
                 << cur << "/" << total << "  ETA=" << (int)eta << "s" << flush;
        }
    });

    unsigned int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 2;
    vector<thread> workers;
    uint64_t chunk_size = total / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
        uint64_t start_idx = t * chunk_size;
        uint64_t end_idx = (t == num_threads - 1) ? total : (t + 1) * chunk_size;

        workers.push_back(thread([&, start_idx, end_idx, key_length]() {
            string local_best_key = "";
            int local_max_len = 0;
            vector<int> local_best_perm;

            for (uint64_t idx = start_idx; idx < end_idx; ++idx) {
                string current_key(key_length, '0');
                uint64_t temp = idx;
                for (int i = key_length - 1; i >= 0; --i) {
                    current_key[i] = CHARSET[temp % 36];
                    temp /= 36;
                }

                vector<int> current_best_perm;
                int len = get_key_max_length(plaintext, current_key, (int)plaintext.length(), current_best_perm);

                if (len > local_max_len) {
                    local_max_len = len;
                    local_best_key = current_key;
                    local_best_perm = current_best_perm;
                }
                checked++;
            }

            // 將此執行緒找到的最長 Key 與全域最大值做比較（執行緒安全）
            lock_guard<mutex> lock(global_mtx);
            if (local_max_len > global_max_len) {
                global_max_len = local_max_len;
                global_best_key = local_best_key;
                global_best_perm = local_best_perm;
                cout << "\n  🔥 [目前新高] Key = '" << global_best_key << "' -> 長度達到: " << global_max_len << "\n";
            }
        }));
    }

    for (auto& w : workers) w.join();
    done = true;
    if (progress_thread.joinable()) progress_thread.join();
}

// ── 主入口 ────────────────────────────────────────────────────────
void bruteforce(const string& plaintext, int max_key_len = 4) {
    cout << string(62, '=') << "\n";
    cout << "  完全保留原規則：尋找能產生最長霍夫曼編碼總長度的 Key\n";
    cout << "  明文 : " << plaintext << "\n";
    cout << string(62, '=') << "\n";

    auto global_start = chrono::high_resolution_clock::now();
    
    // 用於追蹤全域最長紀錄的變數
    string global_best_key = "";
    int global_max_len = 0;
    vector<int> global_best_perm;
    mutex global_mtx;

    for (int key_len = 1; key_len <= max_key_len; ++key_len) {
        bruteforce_single_len(plaintext, key_len, global_best_key, global_max_len, global_best_perm, global_mtx);
    }

    double total_elapsed = chrono::duration<double>(chrono::high_resolution_clock::now() - global_start).count();

    // 最終只印出最長的那一組 Key
    cout << "\n" << string(62, '=') << "\n";
    cout << "  全部搜尋完成！總耗時 " << fixed << setprecision(2) << total_elapsed << " 秒\n";
    cout << "  🏆 最終最長的 Key 結果：\n";
    cout << "  🔑 Key      : '" << global_best_key << "'\n";
    cout << "  📏 總編碼長度: " << global_max_len << " bits\n";
    cout << "  📌 對應排列  : [";
    for (size_t i = 0; i < global_best_perm.size(); ++i) {
        cout << global_best_perm[i] << (i == global_best_perm.size() - 1 ? "" : ", ");
    }
    cout << "]\n";
    cout << string(62, '=') << "\n";
}

int main() {
    init_idx();
    init_perms();

    string plaintext = "book";
    bruteforce(plaintext, 4); // 爆破長度 1~4 的 Key

    return 0;
}