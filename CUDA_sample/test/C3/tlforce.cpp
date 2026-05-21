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

// ── 256-bit 大數與 SHA256 狀態結構 ────────────────────────────────
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
        if (q1_head < Q1.size() && q2_head < Q2.size()) {
            if (Q1[q1_head].first <= Q2[q2_head].first) return Q1[q1_head++];
            else return Q2[q2_head++];
        }
        if (q1_head < Q1.size()) return Q1[q1_head++];
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

    int root = (q1_head < Q1.size()) ? Q1[q1_head].second : Q2[q2_head].second;

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

// ── 排列生成 ────────────────────────────────────────────────────────
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

// ── 嘗試單一 Key ────────────────────────────────────────────────────
bool try_key(const string& plaintext, const string& target, const string& key, int n, int tgt_len, vector<vector<int>>& hit_perms) {
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

    bool any_hit = false;
    for (const auto& perm : ALL_PERMS) {
        string after_perm = "";
        for (int i = 0; i < n; ++i) after_perm += after_poker[perm[i]];

        Seed256 seed_obj{};
        seed_obj.data[7] = seed_base_val;

        int pos = 0;
        bool ok = true;
        for (char char_item : after_perm) {
            string code = encode_char_fast(seed_obj, char_item);
            int len = code.length();
            if (pos + len > tgt_len || target.compare(pos, len, code) != 0) { ok = false; break; }
            pos += len;
            seed_obj.update(char_item);
        }
        if (ok && pos == tgt_len) {
            hit_perms.push_back(perm);
            any_hit = true;
        }
    }
    return any_hit;
}

// ── 核心：帶有進度條與多執行緒的爆破主體 ──────────────────────────────
void bruteforce(const string& plaintext, const string& target_cipher, int key_length = 4) {
    uint64_t total = 1;
    for (int i = 0; i < key_length; ++i) total *= 36;
    uint64_t grand_total = total * 24;

    cout << string(62, '=') << "\n";
    cout << "  暴力破解（找出所有符合的 Key 組合 — C++ 進度條優化版）\n";
    cout << "  明文    : " << plaintext << "\n";
    cout << "  目標密文: " << target_cipher << "\n";
    cout << "  Key 空間: " << total << " 種  | 總搜尋狀態: " << grand_total << "\n";
    cout << string(62, '=') << "\n";

    // 執行緒安全計數器與結果搜集器
    atomic<uint64_t> checked(0);
    atomic<bool> done(false);
    vector<pair<string, vector<int>>> found;
    mutex found_mtx;

    auto start_time = chrono::high_resolution_clock::now();

    // 1. 建立獨立的 UI 背景執行緒，定時繪製進度條 (與 Python 顯示格式完全一致)
    thread progress_thread([&]() {
        while (!done) {
            this_thread::sleep_for(chrono::milliseconds(250)); // 每 250ms 更新一次畫面
            uint64_t current_checked = checked.load();
            if (current_checked == 0) continue;

            auto now = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(now - start_time).count();
            double pct = (double)current_checked / total * 100.0;
            double eta = (elapsed / current_checked) * (total - current_checked);

            int bar_width = 40;
            int pos = bar_width * (pct / 100.0);
            
            cout << "\r  [";
            for (int i = 0; i < bar_width; ++i) {
                if (i < pos) cout << "█";
                else cout << "░";
            }
            cout << "] " << fixed << setprecision(1) << setw(5) << pct << "%  "
                 << current_checked << "/" << total << "  "
                 << "elapsed=" << (int)elapsed << "s  ETA=" << (int)eta << "s" << flush;
        }
    });

    // 2. 使用 OpenMP 或手動多執行緒切分任務（此處採用標準多核心平行運算）
    // 為了編譯相容性，我們手寫均分區間給多個執行緒
    unsigned int num_threads = thread::hardware_concurrency();
    vector<thread> workers;
    uint64_t chunk_size = total / num_threads;

    for (unsigned int t = 0; t < num_threads; ++t) {
        uint64_t start_idx = t * chunk_size;
        uint64_t end_idx = (t == num_threads - 1) ? total : (t + 1) * chunk_size;

        workers.push_back(thread([&, start_idx, end_idx]() {
            for (uint64_t idx = start_idx; idx < end_idx; ++idx) {
                // 還原組合字串
                string current_key(key_length, '0');
                uint64_t temp = idx;
                for (int i = key_length - 1; i >= 0; --i) {
                    current_key[i] = CHARSET[temp % 36];
                    temp /= 36;
                }

                vector<vector<int>> hit_perms;
                if (try_key(plaintext, target_cipher, current_key, plaintext.length(), target_cipher.length(), hit_perms)) {
                    lock_guard<mutex> lock(found_mtx);
                    auto now = chrono::high_resolution_clock::now();
                    double elapsed = chrono::duration<double>(now - start_time).count();
                    for (const auto& p : hit_perms) {
                        found.push_back({current_key, p});
                        cout << "\n  ★ [命中] key = '" << current_key << "'  對應排列 = [";
                        for (size_t i = 0; i < p.size(); ++i) cout << p[i] << (i == p.size() - 1 ? "" : ", ");
                        cout << "]  (耗時 " << fixed << setprecision(2) << elapsed << "s)\n";
                    }
                }
                checked++;
            }
        }));
    }

    // 等待所有計算工作結束
    for (auto& worker : workers) worker.join();
    
    // 關閉進度條執行緒
    done = true;
    if (progress_thread.joinable()) progress_thread.join();

    auto end_time = chrono::high_resolution_clock::now();
    double total_elapsed = chrono::duration<double>(end_time - start_time).count();

    // 3. 列印最終成果結算
    cout << "\n\n" << string(62, '=') << "\n";
    cout << "  搜尋完成！共檢查 " << checked << " 個 key，總共耗時 " << total_elapsed << " 秒\n";
    cout << "  總共尋獲符合條件的答案： " << found.size() << " 組\n";
    cout << string(62, '-') << "\n";
    if (!found.empty()) {
        int res_idx = 1;
        for (const auto& item : found) {
            cout << "  [" << setfill('0') << setw(2) << res_idx++ << "] 🔑 key = '" << item.first << "'  📌 排列 = [";
            for (size_t i = 0; i < item.second.size(); ++i) cout << item.second[i] << (i == item.second.size() - 1 ? "" : ", ");
            cout << "]\n";
        }
    } else {
        cout << "  ❌ 非常遺憾，在此密文下未找到任何符合的 key\n";
    }
    cout << string(62, '=') << "\n";
}

int main() {
    init_idx();
    init_perms();

    string plaintext = "book";
    string target_cipher = "001110110110110000011100011101011001101111000111100010010101100001001110111010001101000011110000101100111111011011110101111";

    bruteforce(plaintext, target_cipher, 4);
    return 0;
}