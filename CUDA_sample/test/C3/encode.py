"""
加密除錯版：逐步印出所有中間結果，含完整 Huffman 表
plaintext = 'book' / key = 'wlaq'

修正摘要：
  STEP 2 換位：
    - seed1 = seed_master XOR (pid*0x3333)
    - LCG 規則: (x*1664525+1013904223) mod 2^32，取 mod 10000 作為排序鍵
    - sorted_idx 依亂數由小到大排列，result[new_pos]=text[sorted_idx[new_pos]]
  STEP 1 撲克牌：
    - 第 R 輪：seed_A = seed_master XOR (0x1111*R)
               seed_B = seed_master XOR (0x2222*R)
"""

import heapq
import hashlib

CHARSET = "0123456789abcdefghijklmnopqrstuvwxyz"
SEP  = "=" * 65
SEP2 = "-" * 65

# ─────────────────────────────────────────────
# 工具函數
# ─────────────────────────────────────────────

def char_to_idx(c): return CHARSET.index(c)
def idx_to_char(i): return CHARSET[i]

def key_to_seed_master(key):
    seed = 0
    for c in key:
        seed = seed * 36 + char_to_idx(c)
    return seed

def lcg_next(state):
    return (state * 1664525 + 1013904223) & 0xFFFFFFFF

def fisher_yates_shuffle(seed, n=36):
    deck = list(range(n))
    state = seed & 0xFFFFFFFF
    for i in range(n - 1, -1, -1):
        state = lcg_next(state)
        j = state % (i + 1)
        deck[i], deck[j] = deck[j], deck[i]
    return deck

def xor32(a, b): return (a ^ b) & 0xFFFFFFFF

def compute_scores(seed):
    return {c: ((i + 1) * seed + 17) % 997 for i, c in enumerate(CHARSET)}

def build_huffman_root(scores):
    counter = 0
    heap = []
    for c, s in scores.items():
        heapq.heappush(heap, (s, counter, c, None, None))
        counter += 1
    while len(heap) > 1:
        s1, c1, ch1, l1, r1 = heapq.heappop(heap)
        s2, c2, ch2, l2, r2 = heapq.heappop(heap)
        heapq.heappush(heap, (s1+s2, counter, None,
                               (s1,c1,ch1,l1,r1), (s2,c2,ch2,l2,r2)))
        counter += 1
    return heap[0]

def gen_codes(node, prefix, codes):
    s, cnt, ch, left, right = node
    if ch is not None:
        codes[ch] = prefix if prefix else "0"
        return
    gen_codes(left,  prefix + "0", codes)
    gen_codes(right, prefix + "1", codes)

def update_seed(seed, char):
    temp = seed ^ ord(char)
    h = hashlib.sha256(str(temp).encode()).hexdigest()
    return int(h, 16)


# ─────────────────────────────────────────────
# STEP 1：撲克牌代換
#   第 R 輪：seed_A = seed_master XOR (0x1111 × R)
#            seed_B = seed_master XOR (0x2222 × R)
# ─────────────────────────────────────────────

def poker_substitute_verbose(text, key, round_num=1):
    print(SEP)
    print(f"  STEP 1：撲克牌代換（Round {round_num}）")
    print(SEP)

    seed_master = key_to_seed_master(key)
    prev = seed_master % 36
    print(f"\n  key = {list(key)}")
    print(f"  各字元索引: { {c: char_to_idx(c) for c in key} }")
    print(f"  seed_master = {seed_master}  (36進位轉換)")
    print(f"  prev 初始值 = seed_master mod 36 = {prev}  ({idx_to_char(prev)})")

    seed_A = xor32(seed_master, 0x1111 * round_num)
    seed_B = xor32(seed_master, 0x2222 * round_num)
    print(f"\n  seed_A = seed_master XOR (0x1111×{round_num}) = {seed_master} XOR {0x1111*round_num} = {seed_A}")
    print(f"  seed_B = seed_master XOR (0x2222×{round_num}) = {seed_master} XOR {0x2222*round_num} = {seed_B}")

    deck_A = fisher_yates_shuffle(seed_A)
    deck_B = fisher_yates_shuffle(seed_B)

    print(f"\n  {'idx':<5}", end="")
    for i in range(36): print(f"{i:<4}", end="")
    print()
    print(f"  {'char':<5}", end="")
    for i in range(36): print(f"{CHARSET[i]:<4}", end="")
    print()
    print(f"  {'A':<5}", end="")
    for v in deck_A: print(f"{CHARSET[v]:<4}", end="")
    print()
    print(f"  {'B':<5}", end="")
    for v in deck_B: print(f"{CHARSET[v]:<4}", end="")
    print()

    print(f"\n  {'pos':<5}{'char':<7}{'idx_c':<8}{'offset':<10}{'idx':<7}{'deck':<7}{'out':<7}{'new prev'}")
    print(f"  {SEP2}")
    result = []
    for i, c in enumerate(text):
        idx_c   = char_to_idx(c)
        offset  = (i + 1 + prev) % 36
        idx     = (idx_c + offset) % 36
        deck    = 'A' if i % 2 == 0 else 'B'
        out_idx = deck_A[idx] if deck == 'A' else deck_B[idx]
        out_c   = idx_to_char(out_idx)
        result.append(out_c)
        print(f"  {i:<5}{c:<7}{idx_c:<8}{offset:<10}{idx:<7}{deck:<7}{out_c:<7}prev={out_idx}({out_c})")
        prev = out_idx

    out = "".join(result)
    print(f"\n  結果: {text} → {out}")
    return out


# ─────────────────────────────────────────────
# STEP 2：換位 Perm
#   - seed1 = seed_master XOR (pid × 0x3333)
#   - LCG 規則: (x×1664525+1013904223) mod 2^32，取 mod 10000
#   - sorted_idx = 索引依亂數由小到大排序
#   - result[new_pos] = text[sorted_idx[new_pos]]
# ─────────────────────────────────────────────

def permute_verbose(text, key, pid=1):
    print(f"\n{SEP}")
    print(f"  STEP 2：換位 Perm（pid={pid}）")
    print(SEP)

    seed_master = key_to_seed_master(key)
    n = len(text)

    seed1 = xor32(seed_master, pid * 0x3333)
    seed2 = xor32(seed_master, (pid + 1) * 0x3333)

    print(f"\n  seed_master = {seed_master}")
    print(f"  seed1 = seed_master XOR ({pid}×0x3333) = {seed_master} XOR {pid*0x3333} = {seed1}")
    print(f"  seed2 = seed_master XOR ({pid+1}×0x3333) = {seed_master} XOR {(pid+1)*0x3333} = {seed2}")
    print(f"\n  LCG 初始值 = seed1 = {seed1}")
    print(f"  LCG 規則: x = (x × 1664525 + 1013904223) mod 2^32，取 x mod 10000 作為排序鍵")

    state = seed1 & 0xFFFFFFFF
    randoms = []
    print(f"\n  LCG 序列 (取 mod 10000):")
    for k in range(n):
        state = lcg_next(state)
        r = state % 10000
        randoms.append(r)
        print(f"    X{k+1} = {state}  →  mod 10000 = {r}")

    # sorted_idx[new_pos] = old_pos
    sorted_idx = sorted(range(n), key=lambda i: randoms[i])

    print(f"\n  亂數 (mod 10000): {randoms}")
    print(f"  依亂數由小到大排序索引: {sorted_idx}")
    print(f"  換位規則: result[new_pos] = text[sorted_idx[new_pos]]")
    print(f"\n  {'new_pos':<10}{'old_pos':<12}{'字元'}")
    print(f"  {SEP2}")
    result = []
    for new_pos, old_pos in enumerate(sorted_idx):
        result.append(text[old_pos])
        print(f"  {new_pos:<10}{old_pos:<12}{text[old_pos]}")

    out = "".join(result)
    print(f"\n  結果: {text} → {out}")
    return out, sorted_idx


# ─────────────────────────────────────────────
# STEP 3：Score Tree（Huffman 編碼）
# ─────────────────────────────────────────────

def print_huffman_table(codes, char_input, table_num, seed):
    print(f"\n  ┌─ Huffman Table #{table_num}  (seed={seed}, 輸入字元='{char_input}') {'─'*20}")
    entries = sorted(codes.items(), key=lambda x: (len(x[1]), x[0]))
    cols = 4
    for row in [entries[i:i+cols] for i in range(0, len(entries), cols)]:
        line = "  │  "
        for ch, code in row:
            mark = " ◄" if ch == char_input else "  "
            line += f"{ch}:{code:<12}{mark}  "
        print(line)
    print(f"  │  → '{char_input}' 編碼為: {codes[char_input]}")
    print(f"  └{'─'*60}")

def score_tree_encode_verbose(text, key, round_num=1):
    print(f"\n{SEP}")
    print(f"  STEP 3：Score Tree 代換（Huffman）- Round {round_num}")
    print(SEP)

    suffix = f"R{round_num-1}"
    seed = sum(ord(c) for c in key) + sum(ord(c) for c in suffix) + 1
    print(f"\n  初始 seed = sum(ord(key)) + sum(ord('{suffix}')) + 1")
    print(f"           = {[ord(c) for c in key]} + {[ord(c) for c in suffix]} + 1")
    print(f"           = {seed}")

    bits = []
    all_codes = []
    for char_idx, char in enumerate(text):
        scores = compute_scores(seed)
        root   = build_huffman_root(scores)
        codes  = {}
        gen_codes(root, "", codes)
        all_codes.append((char, codes, seed))

        encoded = codes[char]
        bits.append(encoded)
        new_seed = update_seed(seed, char)

        print(f"\n  [字元 #{char_idx+1}] '{char}'")
        print(f"    score 公式: score_i = ((i+1) × {seed} + 17) mod 997")
        score_items = [(c, scores[c]) for c in CHARSET]
        print(f"    {'char':>5}{'score':>7}   " * 6)
        for row_i in range(6):
            line = "    "
            for col in range(6):
                idx = row_i * 6 + col
                if idx < len(score_items):
                    c, s = score_items[idx]
                    line += f"{c:>5}{s:>7}   "
            print(line)

        print_huffman_table(codes, char, char_idx+1, seed)
        print(f"    seed 更新: SHA256({seed} XOR ord('{char}')={ord(char)}) → {new_seed}")
        seed = new_seed

    final_bits = "".join(bits)
    print(f"\n{SEP}")
    print(f"  各字元編碼:")
    for i, (char, codes, s) in enumerate(all_codes):
        print(f"    字元 #{i+1} '{char}' → {codes[char]}")
    print(f"\n  串接結果: {'  '.join(codes[char] for char, codes, s in all_codes)}")
    print(f"  最終輸出: {final_bits}")
    return final_bits


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

if __name__ == "__main__":
    key       = input("請輸入 key: ").strip() or 'wlaq'
    plaintext = input("請輸入明文: ").strip() or 'book'

    print(SEP)
    print(f"  加密除錯模式")
    print(f"  明文: {plaintext}    key: {key}")
    print(SEP)

    # ── Round 1 ──
    after_poker_r1          = poker_substitute_verbose(plaintext, key, round_num=1)
    after_perm_r1,  perm1   = permute_verbose(after_poker_r1, key, pid=1)
    after_tree_r1           = score_tree_encode_verbose(after_perm_r1, key, round_num=1)

    # ── Round 2 ──
    after_poker_r2          = poker_substitute_verbose(after_tree_r1, key, round_num=2)
    after_perm_r2,  perm2   = permute_verbose(after_poker_r2, key, pid=2)
    after_tree_r2           = score_tree_encode_verbose(after_perm_r2, key, round_num=2)

    print(f"\n{SEP}")
    print(f"  ★ 加密流程總結")
    print(SEP)
    print(f"  [原始明文]          {plaintext}")
    print(f"  [R1-a 撲克牌後]     {after_poker_r1}")
    print(f"  [R1-b 換位後]       {after_perm_r1}  (sorted_idx={perm1})")
    print(f"  [R1-c Score Tree後] {after_tree_r1}")
    print(f"  [R2-a 撲克牌後]     {after_poker_r2}")
    print(f"  [R2-b 換位後]       {after_perm_r2}  (sorted_idx={perm2})")
    print(f"  [R2-c Score Tree後] {after_tree_r2}")
    print(SEP)
    expected = "1000100111101111001011101011110110101001111101111010111010110101100110100100100011011001011011111010111000101001101111011011100100111"
    match = "✓ 完全匹配！" if after_tree_r2 == expected else "✗ 不符"
    print(f"\n  期望密文: {expected}")
    print(f"  實際密文: {after_tree_r2}")
    print(f"  比對結果: {match}")
    print(SEP)