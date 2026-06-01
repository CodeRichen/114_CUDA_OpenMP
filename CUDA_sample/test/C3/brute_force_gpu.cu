/*
 * brute_force_gpu.cu
 *
 * GPU-accelerated brute-force key search (length 1-4, charset 0-9a-z)
 * for the custom two-round encryption scheme described in the Python script.
 *
 * Build:
 *   nvcc -O3 -arch=sm_75 -o brute_force_gpu brute_force_gpu.cu
 *   (adjust -arch to match your GPU: sm_60 for P-series, sm_86 for RTX 30xx, sm_89 for RTX 40xx)
 *
 * Run:
 *   ./brute_force_gpu
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

// ─────────────────────────────────────────────────────────────────────────────
// Constants (same as Python)
// ─────────────────────────────────────────────────────────────────────────────

#define CHARSET_LEN 36
// "0123456789abcdefghijklmnopqrstuvwxyz"
__constant__ char d_charset[CHARSET_LEN + 1] = "0123456789abcdefghijklmnopqrstuvwxyz";

// Target ciphertext (bit string)
// Python TARGET = "001110110110110000011100011101011001101111000111100010010101100001001110111010001101000011110000101100111111011011110101111"
#define TARGET_LEN 119
__constant__ char d_target[TARGET_LEN + 1] ="001110110110110000011100011101011001101111000111100010010101100001001110111010001101000011110000101100111111011011110101111";

// Plaintext
#define PLAIN_LEN 4
__constant__ char d_plain[PLAIN_LEN + 1] = "book";

// ─────────────────────────────────────────────────────────────────────────────
// Huffman: we need a fixed-size heap on the GPU (no dynamic alloc in kernels).
// 36 leaf nodes → at most 71 internal nodes → 107 total.
// We store the tree as a flat array of nodes.
// ─────────────────────────────────────────────────────────────────────────────

#define HUFF_MAX_NODES 128   // 36*2 + margin
#define CODE_MAX_LEN   24    // max Huffman code length for 36 symbols

struct HuffNode {
    int score;
    int order;       // tie-breaker (insertion counter)
    int ch;          // -1 for internal nodes; else index into d_charset
    int left, right; // indices into node array (-1 = leaf)
};

// ─────────────────────────────────────────────────────────────────────────────
// Device helper functions
// ─────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ int d_char_to_idx(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    return 10 + (c - 'a');
}

__device__ __forceinline__ char d_idx_to_char(int i)
{
    return d_charset[i];
}

__device__ __forceinline__ uint32_t d_lcg_next(uint32_t state)
{
    return (state * 1664525u + 1013904223u);
}

__device__ __forceinline__ uint32_t d_xor32(uint32_t a, uint32_t b)
{
    return a ^ b;
}

// key → seed_master
__device__ uint32_t d_key_to_seed(const char *key, int klen)
{
    uint32_t seed = 0;
    for (int i = 0; i < klen; i++)
        seed = seed * 36u + (uint32_t)d_char_to_idx(key[i]);
    return seed;
}

// Fisher–Yates shuffle, returns deck[36]
__device__ void d_fisher_yates(uint32_t seed, int deck[CHARSET_LEN])
{
    for (int i = 0; i < CHARSET_LEN; i++) deck[i] = i;
    uint32_t state = seed & 0xFFFFFFFFu;
    for (int i = CHARSET_LEN - 1; i >= 0; i--) {
        state = d_lcg_next(state);
        int j = (int)(state % (uint32_t)(i + 1));
        int tmp = deck[i]; deck[i] = deck[j]; deck[j] = tmp;
    }
}

// ── STEP 1: poker_substitute ──────────────────────────────────────────────
__device__ void d_poker_substitute(
    const char *in, int n, char *out,
    uint32_t seed_master, int round_num)
{
    uint32_t seed_A = d_xor32(seed_master, 0x1111u * (uint32_t)round_num);
    uint32_t seed_B = d_xor32(seed_master, 0x2222u * (uint32_t)round_num);
    int deck_A[CHARSET_LEN], deck_B[CHARSET_LEN];
    d_fisher_yates(seed_A, deck_A);
    d_fisher_yates(seed_B, deck_B);

    int prev = (int)(seed_master % 36u);
    for (int i = 0; i < n; i++) {
        int idx_c  = d_char_to_idx(in[i]);
        int offset = (i + 1 + prev) % CHARSET_LEN;
        int idx    = (idx_c + offset) % CHARSET_LEN;
        int out_idx = (i % 2 == 0) ? deck_A[idx] : deck_B[idx];
        out[i] = d_idx_to_char(out_idx);
        prev = out_idx;
    }
}

// ── STEP 2: permute ───────────────────────────────────────────────────────
// For n=4 (plaintext length) we only need a tiny permutation.
__device__ void d_permute(
    const char *in, int n, char *out,
    uint32_t seed_master, int pid)
{
    uint32_t seed1 = d_xor32(seed_master, (uint32_t)pid * 0x3333u);
    uint32_t state = seed1 & 0xFFFFFFFFu;
    int randoms[PLAIN_LEN];   // only works for n <= PLAIN_LEN during first permute
    for (int i = 0; i < n; i++) {
        state = d_lcg_next(state);
        randoms[i] = (int)(state % 10000u);
    }
    // sorted_idx: indices of 'in' sorted by randoms[i]
    // Simple O(n^2) insertion sort (n is tiny: 4 or a few bits)
    int sorted_idx[PLAIN_LEN];
    for (int i = 0; i < n; i++) sorted_idx[i] = i;
    for (int i = 1; i < n; i++) {
        int key_i = sorted_idx[i];
        int val_i = randoms[key_i];
        int j = i - 1;
        while (j >= 0 && randoms[sorted_idx[j]] > val_i) {
            sorted_idx[j+1] = sorted_idx[j];
            j--;
        }
        sorted_idx[j+1] = key_i;
    }
    for (int i = 0; i < n; i++)
        out[i] = in[sorted_idx[i]];
}

// ── SHA-256 (device) ──────────────────────────────────────────────────────
// We need sha256 of a decimal string representation of a uint64 (can be large).
// Python: h = hashlib.sha256(str(temp).encode()).hexdigest(); return int(h,16)
// Since we only need a 64-bit (or even 32-bit) portion for the seed update,
// we implement a minimal device SHA-256.

__device__ __constant__ uint32_t K256[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,
    0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
    0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,
    0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
    0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
    0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,
    0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,
    0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
    0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

#define ROTR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(e,f,g)  (((e)&(f))^((~(e))&(g)))
#define MAJ(a,b,c) (((a)&(b))^((a)&(c))^((b)&(c)))
#define SIG0(a) (ROTR32(a,2)^ROTR32(a,13)^ROTR32(a,22))
#define SIG1(e) (ROTR32(e,6)^ROTR32(e,11)^ROTR32(e,25))
#define sig0(x) (ROTR32(x,7)^ROTR32(x,18)^((x)>>3))
#define sig1(x) (ROTR32(x,17)^ROTR32(x,19)^((x)>>10))

// Compute sha256 of a byte buffer of 'len' bytes.
// Returns full 256-bit hash in hash[8] (big-endian uint32).
__device__ void d_sha256(const uint8_t *msg, int len, uint32_t hash[8])
{
    // We handle at most 2 blocks (for up to 55+64=119 byte messages).
    // A decimal representation of a uint64 is at most 20 digits, so 1 block is enough.
    uint8_t block[128];
    int total_blocks;

    // Copy message into block and pad
    memset(block, 0, 128);
    for (int i = 0; i < len; i++) block[i] = msg[i];
    block[len] = 0x80;
    uint64_t bitlen = (uint64_t)len * 8;

    if (len < 56) {
        total_blocks = 1;
        // write bit length big-endian into bytes 56..63
        for (int i = 0; i < 8; i++)
            block[63 - i] = (uint8_t)(bitlen >> (8*i));
    } else {
        // Two blocks
        total_blocks = 2;
        for (int i = 0; i < 8; i++)
            block[127 - i] = (uint8_t)(bitlen >> (8*i));
    }

    hash[0] = 0x6a09e667u; hash[1] = 0xbb67ae85u;
    hash[2] = 0x3c6ef372u; hash[3] = 0xa54ff53au;
    hash[4] = 0x510e527fu; hash[5] = 0x9b05688cu;
    hash[6] = 0x1f83d9abu; hash[7] = 0x5be0cd19u;

    for (int blk = 0; blk < total_blocks; blk++) {
        uint32_t W[64];
        const uint8_t *b = block + blk * 64;
        for (int i = 0; i < 16; i++)
            W[i] = ((uint32_t)b[i*4]<<24)|((uint32_t)b[i*4+1]<<16)|
                   ((uint32_t)b[i*4+2]<<8)|(uint32_t)b[i*4+3];
        for (int i = 16; i < 64; i++)
            W[i] = sig1(W[i-2]) + W[i-7] + sig0(W[i-15]) + W[i-16];

        uint32_t a=hash[0],b2=hash[1],c=hash[2],d=hash[3];
        uint32_t e=hash[4],f=hash[5],g=hash[6],h=hash[7];
        for (int i = 0; i < 64; i++) {
            uint32_t T1 = h + SIG1(e) + CH(e,f,g) + K256[i] + W[i];
            uint32_t T2 = SIG0(a) + MAJ(a,b2,c);
            h=g; g=f; f=e; e=d+T1;
            d=c; c=b2; b2=a; a=T1+T2;
        }
        hash[0]+=a; hash[1]+=b2; hash[2]+=c; hash[3]+=d;
        hash[4]+=e; hash[5]+=f;  hash[6]+=g; hash[7]+=h;
    }
}

// update_seed: Python does temp = seed ^ ord(char); sha256(str(temp)); return int(hex,16)
// We only use the lower 64 bits of the resulting big integer as seed (enough entropy).
// Actually Python returns the full 256-bit int. We keep it as two uint64.
// But for our Huffman score we only use seed as uint64 so lower 64 bits suffice.

struct Seed128 { uint64_t hi, lo; };

__device__ Seed128 d_update_seed(Seed128 seed, char ch)
{
    // temp = seed ^ ord(char)  (Python big-int XOR, same low bits)
    // str(temp) → decimal string
    uint64_t lo = seed.lo ^ (uint64_t)(unsigned char)ch;
    uint64_t hi = seed.hi; // XOR with char only affects lo bits

    // Convert (hi, lo) big integer to decimal string
    // max value: 2^128 ≈ 3.4e38, at most 39 digits
    // We do decimal conversion via repeated division (big num by 10)
    // Represent as 4x uint32 for easier division
    uint32_t v[4];
    v[0] = (uint32_t)(hi >> 32);
    v[1] = (uint32_t)(hi & 0xFFFFFFFFu);
    v[2] = (uint32_t)(lo >> 32);
    v[3] = (uint32_t)(lo & 0xFFFFFFFFu);

    char decstr[40];
    int dlen = 0;
    // Check if zero
    bool is_zero = (v[0]==0 && v[1]==0 && v[2]==0 && v[3]==0);
    if (is_zero) {
        decstr[0] = '0'; dlen = 1;
    } else {
        char tmp[40]; int tlen = 0;
        while (v[0]||v[1]||v[2]||v[3]) {
            // divide by 10, get remainder
            uint64_t rem = 0;
            for (int i = 0; i < 4; i++) {
                uint64_t cur = rem * 0x100000000ULL + v[i];
                v[i] = (uint32_t)(cur / 10);
                rem  = cur % 10;
            }
            tmp[tlen++] = (char)('0' + rem);
        }
        // reverse
        for (int i = tlen-1; i >= 0; i--)
            decstr[dlen++] = tmp[i];
    }
    decstr[dlen] = 0;

    // SHA-256 of decimal string
    uint32_t hash[8];
    d_sha256((const uint8_t*)decstr, dlen, hash);

    // Convert 256-bit hash to Seed128 (take high 128 bits = hash[0..3])
    Seed128 result;
    result.hi = ((uint64_t)hash[0] << 32) | hash[1];
    result.lo = ((uint64_t)hash[2] << 32) | hash[3];
    return result;
}

// compute_scores: score[charset[i]] = ((i+1)*seed + 17) % 997
// seed here is the low 64 bits of the Python seed (which can be a huge int).
// In Python: seed starts as an int, update_seed returns int(hex,16) which is 256-bit.
// We carry the full 128-bit seed to keep the modulo accurate.

__device__ __forceinline__ int d_compute_score(int char_idx, Seed128 seed)
{
    // ((char_idx+1) * seed + 17) % 997
    // seed is huge; we only need (seed % 997) then multiply.
    // (seed % 997): compute from hi and lo
    uint64_t mod_hi = seed.hi % 997;
    uint64_t mod_lo = ((mod_hi << 32) | (seed.lo >> 32)) % 997;
    mod_lo = ((mod_lo << 32) | (seed.lo & 0xFFFFFFFFu)) % 997;
    uint64_t seed_mod = mod_lo;
    return (int)(((uint64_t)(char_idx + 1) * seed_mod + 17) % 997);
}

// ── STEP 3: score_tree_encode (Huffman) ──────────────────────────────────
// We build a Huffman tree per character position.
// The tree has 36 leaves. We use a flat array + simple heap on the stack.

// Min-heap for Huffman building: element = (score, order, node_idx)
struct HeapElem { int score, order, node_idx; };

__device__ void heap_push(HeapElem *heap, int &sz, HeapElem e)
{
    heap[sz] = e;
    int i = sz++;
    while (i > 0) {
        int p = (i-1)/2;
        bool less = (heap[i].score < heap[p].score) ||
                    (heap[i].score == heap[p].score && heap[i].order < heap[p].order);
        if (!less) break;
        HeapElem tmp = heap[i]; heap[i] = heap[p]; heap[p] = tmp;
        i = p;
    }
}

__device__ HeapElem heap_pop(HeapElem *heap, int &sz)
{
    HeapElem top = heap[0];
    heap[0] = heap[--sz];
    int i = 0;
    while (true) {
        int l=2*i+1, r=2*i+2, smallest=i;
        if (l<sz) {
            bool less = (heap[l].score < heap[smallest].score) ||
                        (heap[l].score == heap[smallest].score && heap[l].order < heap[smallest].order);
            if (less) smallest=l;
        }
        if (r<sz) {
            bool less = (heap[r].score < heap[smallest].score) ||
                        (heap[r].score == heap[smallest].score && heap[r].order < heap[smallest].order);
            if (less) smallest=r;
        }
        if (smallest==i) break;
        HeapElem tmp=heap[i]; heap[i]=heap[smallest]; heap[smallest]=tmp;
        i=smallest;
    }
    return top;
}

// Build Huffman tree for 36 characters given seed, store in node array.
// Returns root index.
__device__ int d_build_huffman(HuffNode nodes[HUFF_MAX_NODES], Seed128 seed)
{
    HeapElem heap[HUFF_MAX_NODES];
    int hsz = 0, counter = 0, nn = 0;

    for (int i = 0; i < CHARSET_LEN; i++) {
        int sc = d_compute_score(i, seed);
        nodes[nn] = {sc, counter, i, -1, -1};
        HeapElem e = {sc, counter, nn};
        heap_push(heap, hsz, e);
        nn++; counter++;
    }

    while (hsz > 1) {
        HeapElem e1 = heap_pop(heap, hsz);
        HeapElem e2 = heap_pop(heap, hsz);
        int sc = nodes[e1.node_idx].score + nodes[e2.node_idx].score;
        nodes[nn] = {sc, counter, -1, e1.node_idx, e2.node_idx};
        HeapElem e = {sc, counter, nn};
        heap_push(heap, hsz, e);
        nn++; counter++;
    }
    return heap_pop(heap, hsz).node_idx;
}

// Traverse Huffman tree to get code for char_idx. Returns bit-string length.
// bits_out: each element is 0 or 1.
__device__ int d_get_code(
    const HuffNode nodes[HUFF_MAX_NODES], int root,
    int char_idx, uint8_t bits_out[CODE_MAX_LEN])
{
    // DFS with explicit stack (no recursion on GPU)
    struct Frame { int node, depth; uint8_t bits[CODE_MAX_LEN]; };
    Frame stk[HUFF_MAX_NODES];
    int sp = 0;
    stk[sp].node = root; stk[sp].depth = 0;
    sp++;

    while (sp > 0) {
        Frame fr = stk[--sp];
        const HuffNode &nd = nodes[fr.node];
        if (nd.ch != -1) {
            if (nd.ch == char_idx) {
                if (fr.depth == 0) { bits_out[0] = 0; return 1; } // single-symbol edge case
                for (int i = 0; i < fr.depth; i++) bits_out[i] = fr.bits[i];
                return fr.depth;
            }
            continue;
        }
        // Push right child (bit=1)
        if (nd.right != -1 && fr.depth < CODE_MAX_LEN-1) {
            stk[sp] = fr;
            stk[sp].node = nd.right;
            stk[sp].bits[fr.depth] = 1;
            stk[sp].depth = fr.depth + 1;
            sp++;
        }
        // Push left child (bit=0)
        if (nd.left != -1 && fr.depth < CODE_MAX_LEN-1) {
            stk[sp] = fr;
            stk[sp].node = nd.left;
            stk[sp].bits[fr.depth] = 0;
            stk[sp].depth = fr.depth + 1;
            sp++;
        }
    }
    return 0; // not found (shouldn't happen)
}

// score_tree_encode: encodes text into a bit-string (stored as char array of '0'/'1')
// Returns total bit length.
__device__ int d_score_tree_encode(
    const char *text, int n,
    const char *key, int klen,
    int round_num,
    char *out_bits, int max_out)
{
    // suffix = f"R{round_num-1}"
    // seed = sum(ord(c) for c in key) + sum(ord(c) for c in suffix) + 1
    int key_sum = 0;
    for (int i = 0; i < klen; i++) key_sum += (int)(unsigned char)key[i];
    // suffix: 'R' + digit(s) for round_num-1
    int rn1 = round_num - 1;
    char suffix[8]; int slen = 0;
    suffix[slen++] = 'R';
    if (rn1 >= 10) suffix[slen++] = (char)('0' + rn1/10);
    suffix[slen++] = (char)('0' + rn1%10);
    int suf_sum = 0;
    for (int i = 0; i < slen; i++) suf_sum += (int)(unsigned char)suffix[i];

    int seed_init = key_sum + suf_sum + 1;
    // Python seed starts as a plain int here; wrap in Seed128
    Seed128 seed; seed.hi = 0; seed.lo = (uint64_t)seed_init;

    int total_bits = 0;
    HuffNode nodes[HUFF_MAX_NODES];

    for (int ci = 0; ci < n; ci++) {
        int root = d_build_huffman(nodes, seed);
        int ch_idx = d_char_to_idx(text[ci]);
        uint8_t code[CODE_MAX_LEN];
        int clen = d_get_code(nodes, root, ch_idx, code);
        if (total_bits + clen > max_out) return -1; // overflow
        for (int b = 0; b < clen; b++)
            out_bits[total_bits++] = (char)('0' + code[b]);
        seed = d_update_seed(seed, text[ci]);
    }
    return total_bits;
}

// ─────────────────────────────────────────────────────────────────────────────
// Full encrypt function (2 rounds)
// Returns true if ciphertext matches target.
// ─────────────────────────────────────────────────────────────────────────────

// We need buffers for intermediate bit-strings.
// After round 1 score_tree_encode, the 4-char plaintext becomes a bit-string.
// That bit-string is then treated as text (each char is '0' or '1') for round 2.
// The bit-string length is variable and can be large (≤ 4*24 = 96 bits per round;
// worst case two rounds: ~96 chars → ~96*24 = 2304 bits, but in practice much less).
// We allocate 512 chars for intermediate buffers (safe upper bound).

#define BUF_SIZE 512

__device__ bool d_encrypt_matches(const char *key, int klen)
{
    uint32_t sm = d_key_to_seed(key, klen);

    char buf_a[BUF_SIZE], buf_b[BUF_SIZE], bits1[BUF_SIZE];

    // ── Round 1 ──
    d_poker_substitute(d_plain, PLAIN_LEN, buf_a, sm, 1);
    d_permute(buf_a, PLAIN_LEN, buf_b, sm, 1);
    int len1 = d_score_tree_encode(buf_b, PLAIN_LEN, key, klen, 1, bits1, BUF_SIZE);
    if (len1 < 0) return false;

    // ── Round 2: poker_substitute on the bit-string ──
    // The bit-string length (len1) can be > PLAIN_LEN; d_permute uses a fixed array.
    // We need a generic permute. Let's inline it for arbitrary n.

    // poker_substitute (round 2)
    {
        uint32_t seed_A = d_xor32(sm, 0x1111u * 2u);
        uint32_t seed_B = d_xor32(sm, 0x2222u * 2u);
        int deck_A[CHARSET_LEN], deck_B[CHARSET_LEN];
        d_fisher_yates(seed_A, deck_A);
        d_fisher_yates(seed_B, deck_B);
        int prev = (int)(sm % 36u);
        for (int i = 0; i < len1; i++) {
            int idx_c  = d_char_to_idx(bits1[i]);
            int offset = (i + 1 + prev) % CHARSET_LEN;
            int idx    = (idx_c + offset) % CHARSET_LEN;
            int out_idx = (i % 2 == 0) ? deck_A[idx] : deck_B[idx];
            buf_a[i] = d_idx_to_char(out_idx);
            prev = out_idx;
        }
    }

    // permute (round 2, generic n = len1)
    {
        uint32_t seed1 = d_xor32(sm, 2u * 0x3333u);
        uint32_t state = seed1 & 0xFFFFFFFFu;
        // We need randoms[len1]; allocate on stack (len1 ≤ ~200 safe)
        int randoms[BUF_SIZE];
        for (int i = 0; i < len1; i++) {
            state = d_lcg_next(state);
            randoms[i] = (int)(state % 10000u);
        }
        // Build sorted_idx via insertion sort
        int sorted_idx[BUF_SIZE];
        for (int i = 0; i < len1; i++) sorted_idx[i] = i;
        for (int i = 1; i < len1; i++) {
            int ki = sorted_idx[i];
            int vi = randoms[ki];
            int j = i-1;
            while (j>=0 && randoms[sorted_idx[j]] > vi) {
                sorted_idx[j+1] = sorted_idx[j]; j--;
            }
            sorted_idx[j+1] = ki;
        }
        for (int i = 0; i < len1; i++)
            buf_b[i] = buf_a[sorted_idx[i]];
    }

    // score_tree_encode (round 2)
    char bits2[BUF_SIZE * 8]; // bit-string can expand significantly
    int len2 = d_score_tree_encode(buf_b, len1, key, klen, 2, bits2, BUF_SIZE * 8);
    if (len2 < 0) return false;

    // Compare with target
    if (len2 != TARGET_LEN) return false;
    for (int i = 0; i < TARGET_LEN; i++)
        if (bits2[i] != d_target[i]) return false;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: each thread tests one key
// Keys are indexed as a linear integer 0 .. total_keys-1
// ─────────────────────────────────────────────────────────────────────────────

// Precompute offsets for each length:
//   length 1: indices   0 ..   35       (36 keys)
//   length 2: indices  36 .. 1331       (36^2 = 1296 keys)
//   length 3: indices 1332 .. 48203      (36^3 = 46656 keys)
//   length 4: indices 48204 .. 1727003   (36^4 = 1679616 keys)
// Total = 36+1296+46656+1679616 = 1727604

#define TOTAL_KEYS 1727604u

__global__ void brute_force_kernel(uint32_t total, uint32_t *d_found_flag, uint32_t *d_found_key_idx)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decode key from linear index
    char key[5]; int klen;
    uint32_t i = idx;
    const uint32_t L1 = 36u;
    const uint32_t L2 = 36u*36u;
    const uint32_t L3 = 36u*36u*36u;

    if (i < L1) {
        klen = 1;
        key[0] = d_charset[i]; key[1] = 0;
    } else if (i < L1 + L2) {
        klen = 2;
        i -= L1;
        key[0] = d_charset[i / 36];
        key[1] = d_charset[i % 36];
        key[2] = 0;
    } else if (i < L1 + L2 + L3) {
        klen = 3;
        i -= (L1 + L2);
        key[0] = d_charset[i / (36*36)];
        key[1] = d_charset[(i / 36) % 36];
        key[2] = d_charset[i % 36];
        key[3] = 0;
    } else {
        klen = 4;
        i -= (L1 + L2 + L3);
        key[0] = d_charset[i / (36*36*36)];
        key[1] = d_charset[(i / (36*36)) % 36];
        key[2] = d_charset[(i / 36) % 36];
        key[3] = d_charset[i % 36];
        key[4] = 0;
    }

    if (d_encrypt_matches(key, klen)) {
        // Atomically record first match
        if (atomicCAS(d_found_flag, 0u, 1u) == 0u) {
            *d_found_key_idx = idx;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CPU: decode linear key index back to string (mirrors kernel logic)
// ─────────────────────────────────────────────────────────────────────────────

static const char charset[] = "0123456789abcdefghijklmnopqrstuvwxyz";

void decode_key(uint32_t idx, char *key, int &klen)
{
    const uint32_t L1=36, L2=36*36, L3=36*36*36;
    if (idx < L1) {
        klen=1; key[0]=charset[idx]; key[1]=0;
    } else if (idx < L1+L2) {
        idx -= L1; klen=2;
        key[0]=charset[idx/36]; key[1]=charset[idx%36]; key[2]=0;
    } else if (idx < L1+L2+L3) {
        idx -= L1+L2; klen=3;
        key[0]=charset[idx/(36*36)];
        key[1]=charset[(idx/36)%36];
        key[2]=charset[idx%36]; key[3]=0;
    } else {
        idx -= L1+L2+L3; klen=4;
        key[0]=charset[idx/(36*36*36)];
        key[1]=charset[(idx/(36*36))%36];
        key[2]=charset[(idx/36)%36];
        key[3]=charset[idx%36]; key[4]=0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

int main()
{
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  GPU Brute-Force Key Search\n");
    printf("  Target  : %s\n", "001110110110110000011100011101011001101111000111100010010101100001001110111010001101000011110000101100111111011011110101111");
    printf("  Plain   : book\n");
    printf("  Charset : 0-9 a-z  (length 1-4)  Total keys: %u\n", TOTAL_KEYS);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Device buffers
    uint32_t *d_found_flag, *d_found_key_idx;
    CUDA_CHECK(cudaMalloc(&d_found_flag,    sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_found_key_idx, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_found_flag,    0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_found_key_idx, 0, sizeof(uint32_t)));

    // Launch config
    const int THREADS = 256;
    int blocks = (TOTAL_KEYS + THREADS - 1) / THREADS;

    printf("  Launching %d blocks × %d threads\n\n", blocks, THREADS);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    brute_force_kernel<<<blocks, THREADS>>>(TOTAL_KEYS, d_found_flag, d_found_key_idx);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    uint32_t found_flag, found_idx;
    CUDA_CHECK(cudaMemcpy(&found_flag, d_found_flag,    sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&found_idx,  d_found_key_idx, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    printf("  Done in %.3f seconds  (~%.0f keys/sec)\n", elapsed, TOTAL_KEYS / elapsed);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    if (found_flag) {
        char key[6]; int klen;
        decode_key(found_idx, key, klen);
        printf("  ★  KEY FOUND:  '%s'\n", key);
    } else {
        printf("  No matching key found in charset length 1-4.\n");
    }
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    cudaFree(d_found_flag);
    cudaFree(d_found_key_idx);
    return 0;
}
