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

// ── 高效線性 Huffman 編碼 ──────────────────────────────────────────
struct Node {
    int left = -1;
    int right = -1;
    string chars = "";
};

// 核心優化：完全保留原版建樹規則，但回傳編碼長度
int get_char_huffman_len(const Seed256& seed_obj, char target_char) {
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

    int len = 0;
    int curr = root;
    while (tree[curr].left != -1) {
        int l = tree[curr].left;
        int r = tree[curr].right;
        if (tree[l].chars.find(target_char) != string::npos) { curr = l; }
        else { curr = r; }
        len++;
    }
    return len == 0 ? 1 : len;
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

// ── 核心驗證：加入逐字剪枝 ──────────────────────────────────────────
bool verify_combination(const string& plaintext, const string& key, int& final_len, vector<int>& win_perm) {
    uint64_t sm = 0;
    for (char c : key) sm = sm * 36 + IDX[(unsigned char)c];
    int prev = sm % 36;

    vector<int> dA = shuffle_deck((sm ^ 0x1111) & 0xFFFFFFFF);
    vector<int> dB = shuffle_deck((sm ^ 0x2222) & 0xFFFFFFFF);

    string after_poker = "";
    for (int i = 0; i < 4; ++i) {
        int ic = IDX[(unsigned char)plaintext[i]];
        int off = (i + 1 + prev) % 36;
        int idx = (ic + off) % 36;
        int oi = (i % 2 == 0) ? dA[idx] : dB[idx];
        after_poker += CHARSET[oi];
        prev = oi;
    }

    uint64_t seed_base_val = 82 + 49;
    for (char c : key) seed_base_val += static_cast<unsigned char>(c);

    // 測試 24 種排列
    for (const auto& perm : ALL_PERMS) {
        string after_perm = "";
        for (int i = 0; i < 4; ++i) after_perm += after_poker[perm[i]];

        Seed256 seed_obj{};
        seed_obj.data[7] = seed_base_val;

        int current_total = 0;
        bool is_valid_path = true;

        for (int i = 0; i < 4; ++i) {
            int len = get_char_huffman_len(seed_obj, after_perm[i]);
            
            // 🔥 關鍵剪枝：因為總分要 140，4個字元，代表「每一個字元」都必須是 35 bits 滿分
            // 只要任何一個字的長度小於 35，這條路徑直接宣告失敗，立刻跳出！
            if (len < 35) {
                is_valid_path = false;
                break;
            }

            current_total += len;
            seed_obj.update(after_perm[i]);
        }

        if (is_valid_path && current_total == 140) {
            final_len = current_total;
            win_perm = perm;
            return true;
        }
    }
    return false;
}

// ── 多執行緒任務分配（以明文空間為外迴圈分工） ───────────────────────
void thread_worker(uint64_t start_plain_idx, uint64_t end_plain_idx, uint64_t total_key_space, 
                   atomic<uint64_t>& global_checked, mutex& out_mtx) {
    
    // 內迴圈遍歷所有 Key 空間 (36^4 = 1679616)
    for (uint64_t p_idx = start_plain_idx; p_idx < end_plain_idx; ++p_idx) {
        // 解析出當前明文字串
        string current_plaintext(4, '0');
        uint64_t temp_p = p_idx;
        for (int i = 3; i >= 0; --i) {
            current_plaintext[i] = CHARSET[temp_p % 36];
            temp_p /= 36;
        }

        for (uint64_t k_idx = 0; k_idx < total_key_space; ++k_idx) {
            // 解析出當前 Key 字串
            string current_key(4, '0');
            uint64_t temp_k = k_idx;
            for (int i = 3; i >= 0; --i) {
                current_key[i] = CHARSET[temp_k % 36];
                temp_k /= 36;
            }

            int final_len = 0;
            vector<int> win_perm;
            
            // 執行帶有快速剪枝的驗證
            if (verify_combination(current_plaintext, current_key, final_len, win_perm)) {
                lock_guard<mutex> lock(out_mtx);
                cout << "\n\n 🎯 [找到完美命中解！]" << endl;
                cout << " 📄 明文 (Plaintext): '" << current_plaintext << "'" << endl;
                cout << " 🔑 密鑰 (Key)      : '" << current_key << "'" << endl;
                cout << " 📏 總長度 (Length) : " << final_len << " bits" << endl;
                cout << " 📌 命中排列 (Perm) : [";
                for(size_t i=0; i<win_perm.size(); ++i) cout << win_perm[i] << (i==win_perm.size()-1?"":", ");
                cout << "]\n\n";
            }
        }
        global_checked += total_key_space; // 更新計數器（以完整處理完一個明文的所有 Key 為單位）
    }
}

int main() {
    init_idx();
    init_perms();

    // 空間計算：長度 4 的 36 進位空間為 36^4 = 1,679,616
    uint64_t space_size = 1679616; 
    uint64_t total_combinations = space_size * space_size; // 明文空間 × Key 空間

    cout << string(65, '=') << "\n";
    cout << " 🚀 多執行緒高速破密工具 (明文4位 × Key4位)" << "\n";
    cout << " ⚡ 核心優化：霍夫曼單步長度不足 35 bits 即刻淘汰 (極速剪枝)" << "\n";
    cout << " 📊 總尋求空間: " << total_combinations << " 次組合驗證\n";
    cout << string(65, '=') << "\n";

    auto start_clock = chrono::high_resolution_clock::now();

    unsigned int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    cout << " 💻 檢測到系統 CPU 核心數，啟用 " << num_threads << " 個平行線程進行加速...\n\n";

    atomic<uint64_t> global_checked(0);
    mutex out_mtx;
    vector<thread> workers;

    // 將明文空間 (space_size) 均分給各個執行緒
    uint64_t chunk_size = space_size / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
        uint64_t start_p = t * chunk_size;
        uint64_t end_p = (t == num_threads - 1) ? space_size : (t + 1) * chunk_size;
        
        workers.push_back(thread(thread_worker, start_p, end_p, space_size, ref(global_checked), ref(out_mtx)));
    }

    // 主執行緒負責印出高頻進度條
    while (global_checked.load() < total_combinations) {
        this_thread::sleep_for(chrono::milliseconds(1000));
        uint64_t cur = global_checked.load();
        if (cur == 0) continue;

        auto now = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(now - start_clock).count();
        double pct = (double)cur / total_combinations * 100.0;
        double eta = (elapsed / cur) * (total_combinations - cur);

        cout << "\r ⏳ 破解進度: " << fixed << setprecision(3) << pct << "% | 已檢查: " 
             << cur << "/" << total_combinations << " | 耗時: " << (int)elapsed << "s | 預估剩餘: " << (int)eta << "s" << flush;
        
        if (cur >= total_combinations) break;
    }

    for (auto& w : workers) w.join();

    double total_elapsed = chrono::duration<double>(chrono::high_resolution_clock::now() - start_clock).count();
    cout << "\n" << string(65, '=') << "\n";
    cout << " 🎉 全域爆破掃描結束！總共耗時 " << fixed << setprecision(2) << total_elapsed << " 秒\n";
    cout << string(65, '=') << "\n";

    return 0;
}