#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// ════════════════════════════════════════════════════════════
// 0. Baseline
// ════════════════════════════════════════════════════════════
__global__ void sobel_baseline(const unsigned char* __restrict__ input,
                               unsigned char* output, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= w-1 || y < 1 || y >= h-1) return;

    float dx = -1*input[(y-1)*w+(x-1)] + 1*input[(y-1)*w+(x+1)]
               -2*input[(y  )*w+(x-1)] + 2*input[(y  )*w+(x+1)]
               -1*input[(y+1)*w+(x-1)] + 1*input[(y+1)*w+(x+1)];
    float dy = -1*input[(y-1)*w+(x-1)] - 2*input[(y-1)*w+x] - 1*input[(y-1)*w+(x+1)]
               +1*input[(y+1)*w+(x-1)] + 2*input[(y+1)*w+x] + 1*input[(y+1)*w+(x+1)];
    float g = sqrtf(dx*dx + dy*dy);
    output[y*w+x] = g > 255.f ? 255 : (unsigned char)g;
}

// ════════════════════════════════════════════════════════════
// 1. __ldg()  — read-only cache (texture cache path)
//    只需在每次 load 加 __ldg()，零架構改動
// ════════════════════════════════════════════════════════════
__global__ void sobel_ldg(const unsigned char* __restrict__ input,
                          unsigned char* output, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= w-1 || y < 1 || y >= h-1) return;

#define LDG(dy,dx) __ldg(&input[(y+(dy))*w + (x+(dx))])

    float gx = -1*LDG(-1,-1) + 1*LDG(-1,+1)
               -2*LDG( 0,-1) + 2*LDG( 0,+1)
               -1*LDG(+1,-1) + 1*LDG(+1,+1);
    float gy = -1*LDG(-1,-1) - 2*LDG(-1, 0) - 1*LDG(-1,+1)
               +1*LDG(+1,-1) + 2*LDG(+1, 0) + 1*LDG(+1,+1);
#undef LDG

    float g = sqrtf(gx*gx + gy*gy);
    output[y*w+x] = g > 255.f ? 255 : (unsigned char)g;
}

// ════════════════════════════════════════════════════════════
// 2. cudaTextureObject_t  — 2-D texture cache
//    硬體自動對 2D spatial locality 最佳化，
//    clamp-to-edge 在硬體處理，kernel 內不需邊界判斷
// ════════════════════════════════════════════════════════════
__global__ void sobel_tex(cudaTextureObject_t tex,
                          unsigned char* output, int w, int h)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    // tex2D 以 float 座標取樣，回傳 unsigned char 對應的 float (0-255)
    // clamp-to-edge 由 texture descriptor 設定，不需手動判斷邊界
#define TEX(dy,dx) tex2D<unsigned char>(tex, x+(dx), y+(dy))

    float gx = -1*TEX(-1,-1) + 1*TEX(-1,+1)
               -2*TEX( 0,-1) + 2*TEX( 0,+1)
               -1*TEX(+1,-1) + 1*TEX(+1,+1);
    float gy = -1*TEX(-1,-1) - 2*TEX(-1, 0) - 1*TEX(-1,+1)
               +1*TEX(+1,-1) + 2*TEX(+1, 0) + 1*TEX(+1,+1);
#undef TEX

    float g = sqrtf(gx*gx + gy*gy);
    output[y*w+x] = g > 255.f ? 255 : (unsigned char)g;
}

// ════════════════════════════════════════════════════════════
// 3. uchar4 向量化讀取
//    每個 thread 一次處理 4 個橫向連續像素。
//    每個 warp（32 threads）一次讀 128 bytes → 一個 cache line。
//
//    注意：width 必須是 4 的倍數，不足的部分 pad 處理。
//    我們在 host 端分配時補齊，kernel 只輸出合法範圍。
// ════════════════════════════════════════════════════════════
__global__ void sobel_vec4(const uchar4* __restrict__ input4,
                           unsigned char* output,
                           int w, int h,
                           int pitch4)   // pitch4 = padded_width / 4
{
    // 每個 thread 負責 4 個橫向像素
    int col4 = blockIdx.x * blockDim.x + threadIdx.x; // uchar4 單位的列索引
    int y    = blockIdx.y * blockDim.y + threadIdx.y;

    int x = col4 * 4;   // 對應 pixel 列索引（第一個）
    if (x < 1 || x+3 >= w-1 || y < 1 || y >= h-1) return;

    // 預先載入三行所需的 uchar4（每行需要左鄰 -1 到右鄰 +4，共 3 個 uchar4）
    // 為了存取 x-1 到 x+4，我們讀 col4-1 .. col4+1 共 3 個 uchar4
    // 但邊界處理複雜，此處用 scalar 方式從 uchar4 提取，保持向量化讀取的好處
    auto load_row = [&](int row_y, int offset_col4) -> uchar4 {
        return __ldg(&input4[row_y * pitch4 + offset_col4]);
    };

    // 讀取 3×3 uchar4 視窗（涵蓋 x-1..x+5 範圍，足夠取 3×3 鄰域）
    uchar4 r00 = load_row(y-1, col4-1), r01 = load_row(y-1, col4), r02 = load_row(y-1, col4+1);
    uchar4 r10 = load_row(y  , col4-1), r11 = load_row(y  , col4), r12 = load_row(y  , col4+1);
    uchar4 r20 = load_row(y+1, col4-1), r21 = load_row(y+1, col4), r22 = load_row(y+1, col4+1);

    // 從 uchar4 組合出 9 個橫向位置的像素陣列
    // idx: -1  0  1  2  3  4  (相對於 x)
    //       r?0.w r?1.xyzw r?2.x
    unsigned char p_top[6] = {r00.w, r01.x, r01.y, r01.z, r01.w, r02.x};
    unsigned char p_mid[6] = {r10.w, r11.x, r11.y, r11.z, r11.w, r12.x};
    unsigned char p_bot[6] = {r20.w, r21.x, r21.y, r21.z, r21.w, r22.x};

    // 對 4 個像素各自計算 Sobel（idx 1..4 對應 x+0..x+3）
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float gx = -1*p_top[i] + 1*p_top[i+2]
                   -2*p_mid[i] + 2*p_mid[i+2]
                   -1*p_bot[i] + 1*p_bot[i+2];
        float gy = -1*p_top[i] - 2*p_top[i+1] - 1*p_top[i+2]
                   +1*p_bot[i] + 2*p_bot[i+1] + 1*p_bot[i+2];
        float g = sqrtf(gx*gx + gy*gy);
        output[(y)*(w) + (x+i)] = g > 255.f ? 255 : (unsigned char)g;
    }
}

// ════════════════════════════════════════════════════════════
// 4. Shared memory（前篇的 coalesced 版本，作為對照）
// ════════════════════════════════════════════════════════════
template<int TILE_W, int TILE_H>
__global__ void sobel_smem(const unsigned char* __restrict__ input,
                           unsigned char* output, int w, int h)
{
    constexpr int SW = TILE_W + 2, SH = TILE_H + 2;
    __shared__ unsigned char smem[SH][SW];

    int tx = threadIdx.x, ty = threadIdx.y;
    int base_x = blockIdx.x * TILE_W - 1;
    int base_y = blockIdx.y * TILE_H - 1;
    int tid = ty * TILE_W + tx;
    int smem_total = SW * SH;

    for (int i = tid; i < smem_total; i += TILE_W * TILE_H) {
        int gx = max(0, min(w-1, base_x + i % SW));
        int gy = max(0, min(h-1, base_y + i / SW));
        smem[i/SW][i%SW] = __ldg(&input[gy*w+gx]);
    }
    __syncthreads();

    int out_x = blockIdx.x * TILE_W + tx;
    int out_y = blockIdx.y * TILE_H + ty;
    if (out_x < 1 || out_x >= w-1 || out_y < 1 || out_y >= h-1) return;

    int sy = ty+1, sx = tx+1;
    float dx = -1.f*smem[sy-1][sx-1] + 1.f*smem[sy-1][sx+1]
               -2.f*smem[sy  ][sx-1] + 2.f*smem[sy  ][sx+1]
               -1.f*smem[sy+1][sx-1] + 1.f*smem[sy+1][sx+1];
    float dy = -1.f*smem[sy-1][sx-1] - 2.f*smem[sy-1][sx] - 1.f*smem[sy-1][sx+1]
               +1.f*smem[sy+1][sx-1] + 2.f*smem[sy+1][sx] + 1.f*smem[sy+1][sx+1];
    float g = sqrtf(dx*dx + dy*dy);
    output[out_y*w + out_x] = g > 255.f ? 255 : (unsigned char)g;
}

// ════════════════════════════════════════════════════════════
// 計時工具
// ════════════════════════════════════════════════════════════
struct Result { float min_ms, avg_ms; };

template<typename F>
Result bench(F launch, int warmup=5, int runs=20)
{
    for (int i=0; i<warmup; i++) launch();
    cudaDeviceSynchronize();
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    float total=0, mn=1e9f;
    for (int i=0; i<runs; i++) {
        cudaEventRecord(s); launch(); cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms,s,e);
        total+=ms; mn=min(mn,ms);
    }
    cudaEventDestroy(s); cudaEventDestroy(e);
    return {mn, total/runs};
}

// ════════════════════════════════════════════════════════════
// Texture object 建立 / 釋放
// ════════════════════════════════════════════════════════════
cudaTextureObject_t make_tex(unsigned char* d_ptr, int w, int h)
{
    // 用 CUDA array 讓硬體以 2D tiling 方式存放，spatial locality 最好
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<unsigned char>();
    cudaArray_t cuArr;
    cudaMallocArray(&cuArr, &desc, w, h);
    cudaMemcpy2DToArray(cuArr, 0, 0, d_ptr, w, w, h, cudaMemcpyDeviceToDevice);

    cudaResourceDesc resDesc{}; resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArr;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeClamp;   // x 方向邊界夾住
    texDesc.addressMode[1] = cudaAddressModeClamp;   // y 方向邊界夾住
    texDesc.filterMode     = cudaFilterModePoint;    // 最近鄰（像素不插值）
    texDesc.readMode       = cudaReadModeElementType;// 讀出原始 uchar，不正規化

    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);
    return tex;
}

int main()
{
    int width, height, channels;
    unsigned char* h_in = stbi_load("input.jpg", &width, &height, &channels, 1);
    if (!h_in) { printf("找不到 input.jpg！\n"); return -1; }
    printf("圖片大小: %d x %d\n\n", width, height);

    size_t size = (size_t)width * height;

    // ── 一般 GPU buffer ──
    unsigned char *d_in, *d_out;
    cudaMalloc(&d_in,  size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // ── uchar4 用的 padded buffer（width 補齊到 4 的倍數）──
    int padded_w = (width + 3) & ~3;           // 向上對齊到 4
    int pitch4   = padded_w / 4;               // uchar4 單位的每行寬度
    size_t size4 = (size_t)padded_w * height;
    unsigned char* h_pad = (unsigned char*)calloc(size4, 1);
    for (int y=0; y<height; y++)
        memcpy(h_pad + y*padded_w, h_in + y*width, width);
    uchar4* d_in4;
    cudaMalloc(&d_in4, size4);
    cudaMemcpy(d_in4, h_pad, size4, cudaMemcpyHostToDevice);
    free(h_pad);

    // ── Texture object（使用 cudaArray，2D tiling）──
    cudaTextureObject_t tex = make_tex(d_in, width, height);

    // ── Block sizes 要測試 ──
    struct BS { int x, y; };
    std::vector<BS> blockSizes = {
        {16,16},{32,8},{8,32},{32,16},{16,32},{8,16},{16,8},{64,4},{4,64}
    };

    printf("%-12s | %-18s | %-18s | %-18s | %-18s | %-18s\n",
           "BlockSize",
           "Baseline(avg)",
           "ldg(avg)",
           "Texture(avg)",
           "Vec4(avg)",
           "Smem(avg)");
    printf("%s\n", std::string(105,'-').c_str());

    float best[5] = {1e9f,1e9f,1e9f,1e9f,1e9f};
    const char* names[5] = {"Baseline","__ldg","Texture","Vec4","Smem"};

    for (auto& bs : blockSizes) {
        if (bs.x * bs.y > 1024) continue;

        dim3 block(bs.x, bs.y);
        dim3 grid((width+bs.x-1)/bs.x, (height+bs.y-1)/bs.y);

        // vec4：x 方向以 4 pixels 為單位，block.x 對應 uchar4 列數
        dim3 block4(bs.x, bs.y);
        dim3 grid4((padded_w/4 + bs.x-1)/bs.x, (height+bs.y-1)/bs.y);

        auto r0 = bench([&]{ sobel_baseline<<<grid, block>>>(d_in,  d_out, width, height); });
        auto r1 = bench([&]{ sobel_ldg    <<<grid, block>>>(d_in,  d_out, width, height); });
        auto r2 = bench([&]{ sobel_tex    <<<grid, block>>>(tex,   d_out, width, height); });
        auto r3 = bench([&]{ sobel_vec4   <<<grid4,block4>>>(d_in4, d_out, width, height, pitch4); });

        // smem：只用 16×16 tile（模板參數固定）
        auto smem_kernel = sobel_smem<16,16>;
auto r4 = bench([&]{
    smem_kernel<<<dim3((width+15)/16,(height+15)/16),
                  dim3(16,16)>>>(d_in, d_out, width, height);
});

        float avgs[5] = {r0.avg_ms, r1.avg_ms, r2.avg_ms, r3.avg_ms, r4.avg_ms};
        printf("(%3d,%3d)    |", bs.x, bs.y);
        for (int i=0; i<5; i++) {
            printf(" %17.4f |", avgs[i]);
            best[i] = min(best[i], avgs[i]);
        }
        printf("\n");
    }

    printf("%s\n", std::string(105,'-').c_str());
    printf("%-12s |", "Best avg");
    for (int i=0; i<5; i++) printf(" %17.4f |", best[i]);
    printf("\n\n");

    // ── 找整體最快，輸出結果圖 ──
    int winner = 0;
    for (int i=1; i<5; i++) if (best[i] < best[winner]) winner = i;
    printf("最快方法: %s (%.4f ms)\n", names[winner], best[winner]);

    // 用 texture 版輸出（通常最穩定）
    dim3 b(32,8), g((width+31)/32, (height+7)/8);
    sobel_tex<<<g,b>>>(tex, d_out, width, height);
    cudaDeviceSynchronize();

    unsigned char* h_out = (unsigned char*)malloc(size);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    stbi_write_jpg("output.jpg", width, height, 1, h_out, 95);
    printf("結果已存至 output.jpg\n");

    // ── 釋放資源 ──
    cudaDestroyTextureObject(tex);
    stbi_image_free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_in4);
    return 0;
}