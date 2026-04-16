// gemm_4ways_fp16_simpler.cu
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <functional>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublasLt.h>

#ifdef ENABLE_CUTLASS
  #include "cutlass/gemm/device/gemm.h"
  #include "cutlass/layout/matrix.h"
#endif

using namespace nvcuda;

#define CHECK_CUDA(x) do{ auto e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} }while(0)
#define CHECK_LT(x) do{ auto s=(x); if(s!=CUBLAS_STATUS_SUCCESS){ \
  printf("cuBLASLt error %s:%d: status=%d\n",__FILE__,__LINE__,(int)s); std::exit(1);} }while(0)

// -------------------- timing helper --------------------
static float time_ms(std::function<void(cudaStream_t)> fn, cudaStream_t stream,
                     int iters=200, int warmup=20){
  cudaEvent_t a,b;
  CHECK_CUDA(cudaEventCreate(&a));
  CHECK_CUDA(cudaEventCreate(&b));

  for(int i=0;i<warmup;i++) fn(stream);

  CHECK_CUDA(cudaEventRecord(a,stream));
  for(int i=0;i<iters;i++) fn(stream);
  CHECK_CUDA(cudaEventRecord(b,stream));
  CHECK_CUDA(cudaEventSynchronize(b));

  float ms=0.f;
  CHECK_CUDA(cudaEventElapsedTime(&ms,a,b));
  CHECK_CUDA(cudaEventDestroy(a));
  CHECK_CUDA(cudaEventDestroy(b));
  return ms/iters;
}

// -------------------- data init --------------------
static void fill_fp16(__half* p, int n){
  for(int i=0;i<n;i++){
    float v=float((i%13)-6)/7.0f;   // small deterministic values
    p[i]=__float2half(v);
  }
}

// Convert B from row-major (KxN) to col-major (KxN) with leading dim K
// row: Brow[k*N + n]
// col: Bcol[n*K + k]
static void row_to_col_major_B(const __half* Brow, __half* Bcol, int K, int N){
  for(int k=0;k<K;k++)
    for(int n=0;n<N;n++)
      Bcol[n*K + k] = Brow[k*N + n];
}

// ======================================================
// 1) Naive CUDA GEMM (FP16 load -> FP32 FMA -> FP16 store)
// ======================================================
__global__ void naive_gemm_fp16(const half* A, const half* B, half* C, int M, int N, int K){
  int r=blockIdx.y*blockDim.y+threadIdx.y;
  int c=blockIdx.x*blockDim.x+threadIdx.x;
  if(r>=M||c>=N) return;

  float acc=0.f;
  for(int k=0;k<K;k++){
    acc += __half2float(A[r*K+k]) * __half2float(B[k*N+c]);
  }
  C[r*N+c]=__float2half(acc);
}

static void run_naive(cudaStream_t s, const half* A, const half* B, half* C,
                      int M,int N,int K){
  dim3 blk(16,16);
  dim3 grd((N+15)/16,(M+15)/16);
  naive_gemm_fp16<<<grd,blk,0,s>>>(A,B,C,M,N,K);
}

// =======================================
// 2) WMMA (A row-major, B col-major, C float row-major)
//    We store accumulator to float* because CUDA 12 mma.h
//    does not provide store(accumulator<float>) -> half* overload.
// =======================================
__global__ void wmma_gemm_kernel(const half* A, const half* Bcol, float* C, int M,int N,int K){
  int gw = (blockIdx.x*blockDim.x + threadIdx.x) / warpSize;
  int tilesN = N/16;
  int tr = gw / tilesN;
  int tc = gw % tilesN;
  if(tr*16>=M || tc*16>=N) return;

  wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> aF;
  wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> bF;
  wmma::fragment<wmma::accumulator,16,16,16,float> accF;
  wmma::fill_fragment(accF, 0.0f);

  for(int k0=0;k0<K;k0+=16){
    const half* aTile  = A    + (tr*16)*K + k0;        // row-major ld=K
    const half* bTile  = Bcol + (tc*16)*K + k0;       // col-major ld=K (n*K + k)
    wmma::load_matrix_sync(aF, aTile, K);
    wmma::load_matrix_sync(bF, bTile, K);
    wmma::mma_sync(accF, aF, bF, accF);
  }

  float* cTile = C + (tr*16)*N + (tc*16);
  wmma::store_matrix_sync(cTile, accF, N, wmma::mem_row_major);
}

static void run_wmma(cudaStream_t s, const half* A, const half* Bcol, float* C,
                     int M,int N,int K){
  int tiles = (M/16)*(N/16);
  int warpsPerBlock=4;         // 128 threads
  int threads=warpsPerBlock*32;
  int blocks=(tiles+warpsPerBlock-1)/warpsPerBlock;
  wmma_gemm_kernel<<<blocks,threads,0,s>>>(A,Bcol,C,M,N,K);
}

// ======================================================
// 3) cuBLASLt (<=30 lines; only necessary settings)
//    Row-major, FP16 inputs, FP32 accumulate, C = A*B
// ======================================================
static void run_cublaslt_30lines(cudaStream_t stream, const half* A, const half* B, half* C,
                                int M,int N,int K, void* workspace, size_t wsBytes){
  cublasLtHandle_t lt; cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t aL,bL,cL; cublasLtMatmulPreference_t pref;
  CHECK_LT(cublasLtCreate(&lt));
  CHECK_LT(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t tN=CUBLAS_OP_N;
  CHECK_LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&tN,sizeof(tN)));
  CHECK_LT(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&tN,sizeof(tN)));
  CHECK_LT(cublasLtMatrixLayoutCreate(&aL, CUDA_R_16F, M, K, K));
  CHECK_LT(cublasLtMatrixLayoutCreate(&bL, CUDA_R_16F, K, N, N));
  CHECK_LT(cublasLtMatrixLayoutCreate(&cL, CUDA_R_16F, M, N, N));
  cublasLtOrder_t row=CUBLASLT_ORDER_ROW;
  CHECK_LT(cublasLtMatrixLayoutSetAttribute(aL,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(row)));
  CHECK_LT(cublasLtMatrixLayoutSetAttribute(bL,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(row)));
  CHECK_LT(cublasLtMatrixLayoutSetAttribute(cL,CUBLASLT_MATRIX_LAYOUT_ORDER,&row,sizeof(row)));
  CHECK_LT(cublasLtMatmulPreferenceCreate(&pref));
  CHECK_LT(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsBytes,sizeof(wsBytes)));
  cublasLtMatmulHeuristicResult_t heur; int got=0;
  CHECK_LT(cublasLtMatmulAlgoGetHeuristic(lt,op,aL,bL,cL,cL,pref,1,&heur,&got));
  float alpha=1.f,beta=0.f;
  if(got) CHECK_LT(cublasLtMatmul(lt,op,&alpha,A,aL,B,bL,&beta,C,cL,C,cL,&heur.algo,workspace,wsBytes,stream));
  CHECK_LT(cublasLtMatmulPreferenceDestroy(pref));
  CHECK_LT(cublasLtMatrixLayoutDestroy(aL)); CHECK_LT(cublasLtMatrixLayoutDestroy(bL)); CHECK_LT(cublasLtMatrixLayoutDestroy(cL));
  CHECK_LT(cublasLtMatmulDescDestroy(op)); CHECK_LT(cublasLtDestroy(lt));
}

// ======================================================
// 4) CUTLASS wrapper: main only sees one-line call
// ======================================================
#ifdef ENABLE_CUTLASS
static void run_cutlass(int M,int N,int K, const half* A, const half* B, half* C, cudaStream_t stream){
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::RowMajor,
    float>;
  Gemm op;
  cutlass::gemm::GemmCoord prob(M,N,K);
  int lda=K, ldb=N, ldc=N;
  typename Gemm::Arguments args{
    prob,
    {reinterpret_cast<const cutlass::half_t*>(A), lda},
    {reinterpret_cast<const cutlass::half_t*>(B), ldb},
    {reinterpret_cast<const cutlass::half_t*>(C), ldc}, // C input
    {reinterpret_cast<cutlass::half_t*>(C), ldc},       // D output
    {1.0f, 0.0f}
  };
  if(op.can_implement(args)!=cutlass::Status::kSuccess){ printf("CUTLASS can_implement failed\n"); return; }
  size_t ws = Gemm::get_workspace_size(args);
  void* w=nullptr; if(ws) CHECK_CUDA(cudaMalloc(&w, ws));
  if(op.initialize(args, w, stream)!=cutlass::Status::kSuccess){ printf("CUTLASS init failed\n"); return; }
  if(op(stream)!=cutlass::Status::kSuccess){ printf("CUTLASS run failed\n"); return; }
  if(ws) CHECK_CUDA(cudaFree(w));
}
#endif

// (Optional) If you want WMMA result in FP16, uncomment this and include it in timing.
// __global__ void fp32_to_fp16(const float* in, half* out, int n){
//   int i=blockIdx.x*blockDim.x+threadIdx.x;
//   if(i<n) out[i]=__float2half(in[i]);
// }

int main(){
  // Orin: keep multiples of 16 for WMMA.
  const int M=512, N=512, K=512;

  printf("Jetson Orin (aarch64) | GEMM FP16(A,B) FP32(acc) | C=FP16 except WMMA stores FP32\n");
  printf("M=%d N=%d K=%d\n\n", M,N,K);

  // Host
  std::vector<__half> hA(M*K), hB(K*N), hBcol(K*N);
  fill_fp16(hA.data(), M*K);
  fill_fp16(hB.data(), K*N);
  row_to_col_major_B(hB.data(), hBcol.data(), K, N);

  // Device
  half  *dA=nullptr, *dB=nullptr, *dBcol=nullptr, *dC=nullptr;
  float *dCw=nullptr;  // WMMA output (float)
  CHECK_CUDA(cudaMalloc(&dA,    sizeof(half)*M*K));
  CHECK_CUDA(cudaMalloc(&dB,    sizeof(half)*K*N));
  CHECK_CUDA(cudaMalloc(&dBcol, sizeof(half)*K*N));
  CHECK_CUDA(cudaMalloc(&dC,    sizeof(half)*M*N));
  CHECK_CUDA(cudaMalloc(&dCw,   sizeof(float)*M*N));

  CHECK_CUDA(cudaMemcpy(dA, hA.data(),    sizeof(half)*M*K, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(),    sizeof(half)*K*N, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dBcol, hBcol.data(), sizeof(half)*K*N, cudaMemcpyHostToDevice));

  cudaStream_t stream; CHECK_CUDA(cudaStreamCreate(&stream));

  // Workspace for cuBLASLt
  size_t wsBytes = 1<<22; // 4MB
  void* dWs=nullptr; CHECK_CUDA(cudaMalloc(&dWs, wsBytes));

  // Measure
  auto ms_naive = time_ms([&](cudaStream_t s){
    CHECK_CUDA(cudaMemsetAsync(dC, 0, sizeof(half)*M*N, s));
    run_naive(s, dA, dB, dC, M,N,K);
  }, stream);

  auto ms_wmma = time_ms([&](cudaStream_t s){
    CHECK_CUDA(cudaMemsetAsync(dCw, 0, sizeof(float)*M*N, s));
    run_wmma(s, dA, dBcol, dCw, M,N,K);
  }, stream);

  auto ms_lt = time_ms([&](cudaStream_t s){
    CHECK_CUDA(cudaMemsetAsync(dC, 0, sizeof(half)*M*N, s));
    run_cublaslt_30lines(s, dA, dB, dC, M,N,K, dWs, wsBytes);
  }, stream);

#ifdef ENABLE_CUTLASS
  auto ms_cutlass = time_ms([&](cudaStream_t s){
    CHECK_CUDA(cudaMemsetAsync(dC, 0, sizeof(half)*M*N, s));
    run_cutlass(M,N,K, dA, dB, dC, s);   // one-line call
  }, stream);
#endif

  // TFLOPS for GEMM (2*M*N*K)
  double flops = 2.0 * double(M) * double(N) * double(K);
  auto tflops=[&](double ms){ return flops / (ms*1e-3) / 1e12; };

  printf("Avg time per run:\n");
  printf("  Naive CUDA : %8.4f ms  (%6.2f TFLOPS)\n", ms_naive, tflops(ms_naive));
  printf("  WMMA       : %8.4f ms  (%6.2f TFLOPS)\n", ms_wmma,  tflops(ms_wmma));
  printf("  cuBLASLt   : %8.4f ms  (%6.2f TFLOPS)\n", ms_lt,    tflops(ms_lt));
#ifdef ENABLE_CUTLASS
  printf("  CUTLASS    : %8.4f ms  (%6.2f TFLOPS)\n", ms_cutlass, tflops(ms_cutlass));
#else
  printf("  CUTLASS    : (not built) compile with -DENABLE_CUTLASS and include cutlass\n");
#endif

  // Cleanup
  CHECK_CUDA(cudaFree(dWs));
  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dBcol));
  CHECK_CUDA(cudaFree(dC));
  CHECK_CUDA(cudaFree(dCw));
  CHECK_CUDA(cudaStreamDestroy(stream));

  return 0;
}