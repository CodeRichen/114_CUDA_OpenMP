#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<time.h>

/*
 * 這個範例用來比較多種 CUDA 啟動配置在向量加法上的差異。
 *
 * 資料規模：
 *   ELEMENT_COUNT = 1024 * 1024 個元素 (float)
 *
 * 預設較佳啟動配置：
 *   THREADSPERBLOCK = 128
 *   BLOCKSPERGRID   = ELEMENT_COUNT / THREADSPERBLOCK
 *
 * 備註：
 *   下方部分 kernel 是刻意保留的「非最佳寫法」，用於教學比較。
 */
#define ELEMENT_COUNT 1024*1024
#define THREADSPERBLOCK 128
#define BLOCKSPERGRID ELEMENT_COUNT/THREADSPERBLOCK 
// 一維索引模型：一個 thread 對應一個邏輯元素索引。
// Grid 大小仍需符合裝置限制。

// 基準與實驗用 kernel 宣告。
__global__ void vecAdd_gpu_kernel_1_1(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_1_2(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_1_256(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_2_256(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_1_4096(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_256_1(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_256_2(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_better(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_2_19(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_2_18(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_2_10(float vecA[],float vecB[],float vecC[]);

void vecAdd_cpu(float vecA[],float vecB[],float vecC[]);
int check_result(float h_res[],float d_res[]);

int main(int argc, char **argv)
{

	// Host 端緩衝區：
	// h_vecA/h_vecB 為輸入，h_vecResultFromDevice 為 GPU 輸出，
	// h_vecResultFromHost 為 CPU 參考答案。
	float *h_vecA,*h_vecB,*h_vecResultFromDevice,*h_vecResultFromHost;
	h_vecA = (float*)malloc(sizeof(float)*ELEMENT_COUNT);
	h_vecB = (float*)malloc(sizeof(float)*ELEMENT_COUNT); 
	h_vecResultFromDevice = (float*)malloc(sizeof(float)*ELEMENT_COUNT);
	h_vecResultFromHost = (float*)malloc(sizeof(float)*ELEMENT_COUNT);
	// 配置 Host 記憶體。

	srand(time(0));
	// 設定亂數種子（只需一次）。

	for(int i=0;i<ELEMENT_COUNT;i++)
	{
		h_vecA[i] = rand()%100;
		h_vecB[i] = rand()%100;
	}
	// 以小範圍亂數初始化輸入資料。

	cudaError_t R; 
	// 以 R 接收每個 CUDA API 的回傳狀態。

	float *d_vecA,*d_vecB,*d_vecC;
	printf("\n ========== Check cudaMalloc ==========\n");
	R = cudaMalloc((void**)&d_vecA,sizeof(float)*ELEMENT_COUNT);
	printf(" Malloc d_vecA : %s\n",cudaGetErrorString(R));
	R = cudaMalloc((void**)&d_vecB,sizeof(float)*ELEMENT_COUNT);
	printf(" Malloc d_vecB : %s\n",cudaGetErrorString(R));
	R = cudaMalloc((void**)&d_vecC,sizeof(float)*ELEMENT_COUNT);
	printf(" Malloc d_vecC : %s\n\n",cudaGetErrorString(R));
	// 在 Device 端配置兩個輸入與一個輸出陣列。

	printf(" ======== Check Data Transfer =========\n");
	R = cudaMemcpy(d_vecA,h_vecA,sizeof(float)*ELEMENT_COUNT,cudaMemcpyHostToDevice);	
	printf(" Memory Copy d_vecA : %s\n",cudaGetErrorString(R));
	R = cudaMemcpy(d_vecB,h_vecB,sizeof(float)*ELEMENT_COUNT,cudaMemcpyHostToDevice);
	printf(" Memory Copy d_vecB : %s\n\n",cudaGetErrorString(R));
	// 將初始化完成的輸入資料由 Host 複製到 Device。

	// 先計算 CPU 參考答案，後面每個 GPU kernel 都會拿它來比對。
	clock_t t1 = clock();
	vecAdd_cpu(h_vecA,h_vecB,h_vecResultFromHost);
	clock_t t2 = clock();
	float CPU_elapsedTime;
	CPU_elapsedTime = (t2-t1)/(double)(CLOCKS_PER_SEC);
	// CPU_elapsedTime 單位為秒。

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
	// 使用 CUDA event 量測 GPU 端耗時。

	cudaEventRecord(start,0);
	// 啟動 GPU 計時（stream 0）。

	// 這裡改成一次執行測試所有「已實作」的 kernel。
	// 只要某個 kernel launch 成功，就會同步、複製結果並做比對。
	// vecAdd_gpu_kernel_2_18 目前只有宣告、沒有定義，所以不會納入測試。
	int allKernelPass = 1;
	int testedKernelCount = 0;
	#define RUN_KERNEL_TEST(KERNEL_NAME, ...) \
		do { \
			printf("\n [%s] 開始測試\n", KERNEL_NAME); \
			cudaEvent_t kernelStart, kernelStop; \
			cudaEventCreate(&kernelStart); \
			cudaEventCreate(&kernelStop); \
			cudaEventRecord(kernelStart,0); \
			cudaMemset(d_vecC, 0, sizeof(float)*ELEMENT_COUNT); \
			__VA_ARGS__; \
			cudaError_t launchErr = cudaGetLastError(); \
			if (launchErr != cudaSuccess) { \
				printf("  Launch 失敗 : %s\n", cudaGetErrorString(launchErr)); \
				allKernelPass = 0; \
			} else { \
				cudaError_t syncErr = cudaDeviceSynchronize(); \
				if (syncErr != cudaSuccess) { \
					printf("  同步失敗 : %s\n", cudaGetErrorString(syncErr)); \
					allKernelPass = 0; \
				} else { \
					cudaEventRecord(kernelStop,0); \
					cudaEventSynchronize(kernelStop); \
					float kernelElapsedMs = 0.0f; \
					cudaEventElapsedTime(&kernelElapsedMs, kernelStart, kernelStop); \
					printf("  Kernel 時間 : %3.20f s\n", kernelElapsedMs / 1000.0f); \
					R = cudaMemcpy(h_vecResultFromDevice,d_vecC,sizeof(float)*ELEMENT_COUNT,cudaMemcpyDeviceToHost); \
					printf("  Memcpy h_vecResultFromDevice : %s\n",cudaGetErrorString(R)); \
					if(check_result(h_vecResultFromHost,h_vecResultFromDevice)) \
						printf("  Result Check : OK!\n"); \
					else { \
						printf("  Result Check : QQ!\n"); \
						allKernelPass = 0; \
					} \
				} \
			} \
			cudaEventDestroy(kernelStart); \
			cudaEventDestroy(kernelStop); \
			testedKernelCount++; \
		} while(0)
	
	RUN_KERNEL_TEST("vecAdd_gpu_kernel_1_1", vecAdd_gpu_kernel_1_1<<<1,1>>>(d_vecA,d_vecB,d_vecC));
	RUN_KERNEL_TEST("vecAdd_gpu_kernel_1_2", vecAdd_gpu_kernel_1_2<<<1,2>>>(d_vecA,d_vecB,d_vecC));
	RUN_KERNEL_TEST("vecAdd_gpu_kernel_1_256", vecAdd_gpu_kernel_1_256<<<1,256>>>(d_vecA,d_vecB,d_vecC));
	RUN_KERNEL_TEST("vecAdd_gpu_kernel_2_256", vecAdd_gpu_kernel_2_256<<<2,256>>>(d_vecA,d_vecB,d_vecC));
	RUN_KERNEL_TEST("vecAdd_gpu_kernel_1_4096", vecAdd_gpu_kernel_1_4096<<<1,4096>>>(d_vecA,d_vecB,d_vecC));
	RUN_KERNEL_TEST("vecAdd_gpu_kernel_256_1", vecAdd_gpu_kernel_256_1<<<256,1>>>(d_vecA,d_vecB,d_vecC));
	RUN_KERNEL_TEST("vecAdd_gpu_kernel_256_2", vecAdd_gpu_kernel_256_2<<<256,2>>>(d_vecA,d_vecB,d_vecC));
	RUN_KERNEL_TEST("vecAdd_gpu_kernel_better", vecAdd_gpu_kernel_better<<<BLOCKSPERGRID,THREADSPERBLOCK>>>(d_vecA,d_vecB,d_vecC));
	RUN_KERNEL_TEST("vecAdd_gpu_kernel_2_19", vecAdd_gpu_kernel_2_19<<<BLOCKSPERGRID,THREADSPERBLOCK>>>(d_vecA,d_vecB,d_vecC));
	RUN_KERNEL_TEST("vecAdd_gpu_kernel_2_18", vecAdd_gpu_kernel_2_18<<<BLOCKSPERGRID,THREADSPERBLOCK>>>(d_vecA,d_vecB,d_vecC));
	RUN_KERNEL_TEST("vecAdd_gpu_kernel_2_10", vecAdd_gpu_kernel_2_10<<<BLOCKSPERGRID,THREADSPERBLOCK>>>(d_vecA,d_vecB,d_vecC));
	#undef RUN_KERNEL_TEST

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	// 停止 GPU 計時（若未啟動 kernel，則量到的是近乎空操作時間）。
	
	float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
	// elapsedTime 單位為毫秒。

	free(h_vecA);
	free(h_vecB);
	free(h_vecResultFromDevice);
	// 釋放 Host 記憶體。

	cudaFree(d_vecA);
	cudaFree(d_vecB);
	cudaFree(d_vecC);
	// 釋放 Device 記憶體。

	printf("\n ======== Execution Infomation ========\n");
	printf(" GPU total test time: %3.20f s\n",elapsedTime/1000);
	printf(" Excuetion Time on CPU: %3.20f s\n",CPU_elapsedTime);
	printf(" Tested kernel count = %d\n",testedKernelCount);
	printf(" All kernel passed = %s\n",allKernelPass ? "YES" : "NO");
	printf(" Speed up = %f\n",(CPU_elapsedTime/(elapsedTime/1000)));
	printf(" ======================================\n\n"); 
	// 印出時間與簡易加速比。
	

	//system("pause");
	return 0;
}

__global__ void vecAdd_gpu_kernel_better(float vecA[],float vecB[],float vecC[])
{
	// 標準 CUDA 寫法：一個 thread 計算一個元素。
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < ELEMENT_COUNT)
    {
        vecC[i] = vecB[i] + vecA[i];
    }

}

__global__ void vecAdd_gpu_kernel_2_19(float vecA[],float vecB[],float vecC[])
{
		// 大步長版本：每個 thread 以 1024*512 的間隔處理多個元素，
		// 用來示範「迴圈分派工作」的映射方式。
    int j = blockDim.x * blockIdx.x + threadIdx.x;
	for(int i=0;i<ELEMENT_COUNT;i+=1024*512)
		if(i+threadIdx.x+blockDim.x * blockIdx.x < ELEMENT_COUNT)
			vecC[i+j] = vecB[i+j] + vecA[i+j];    
}

__global__ void vecAdd_gpu_kernel_2_18(float vecA[],float vecB[],float vecC[])
{
		// 與 2_19 類似，但改成 1024*256 的間隔（約 2^18）。
    int j = blockDim.x * blockIdx.x + threadIdx.x;
	for(int i=0;i<ELEMENT_COUNT;i+=1024*256)
		if(i+threadIdx.x+blockDim.x * blockIdx.x < ELEMENT_COUNT)
			vecC[i+j] = vecB[i+j] + vecA[i+j];    
}

__global__ void vecAdd_gpu_kernel_2_10(float vecA[],float vecB[],float vecC[])
{
		// 與 kernel_2_19 類似，但步長縮小為 1024，
		// 因此每個 thread 的迴圈次數會增加。
    int j = blockDim.x * blockIdx.x + threadIdx.x;
	for(int i=0;i<ELEMENT_COUNT;i+=1024)
		if(i+threadIdx.x+blockDim.x * blockIdx.x < ELEMENT_COUNT)
			vecC[i+j] = vecB[i+j] + vecA[i+j];    
}

__global__ void vecAdd_gpu_kernel_256_1(float vecA[],float vecB[],float vecC[])
{
	// 啟動配置示例：很多 block、每個 block 只有一個 thread。
	// 在現代 GPU 上通常效率不佳（occupancy 偏低）。
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  for(int i=0;i<ELEMENT_COUNT;i+=256)
	if(i+threadIdx.x+blockDim.x * blockIdx.x < ELEMENT_COUNT)
		vecC[i+j] = vecB[i+j] + vecA[i+j];
}
/*
256 1
(0,0)
j=0		i=0:0, 		i=256:256,	i=512:512....

(1,0)
j=1		i=0:1, 		i=256:257, 	i=512:513....

(2,0)
j=2		i=0:2,		i=256:258,	i=512:514....

...

(255,0)
j=255	i=0;255,	i=256:511,	i=512:767....
*/

__global__ void vecAdd_gpu_kernel_256_2(float vecA[],float vecB[],float vecC[])
{
	// 以步長 512 模擬 2-way 工作分攤（搭配 launch 256,2）。
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  for(int i=0;i<ELEMENT_COUNT;i+=512)
	if(i+threadIdx.x+blockDim.x * blockIdx.x < ELEMENT_COUNT)
		vecC[i+j] = vecB[i+j] + vecA[i+j];
}

__global__ void vecAdd_gpu_kernel_2_256(float vecA[],float vecB[],float vecC[])
{
	// 與 kernel_256_2 內容等價，主要用於測試不同命名/配置。
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  for(int i=0;i<ELEMENT_COUNT;i+=512)
	if(i+threadIdx.x+blockDim.x * blockIdx.x < ELEMENT_COUNT)
		vecC[i+j] = vecB[i+j] + vecA[i+j];
}

__global__ void vecAdd_gpu_kernel_1_1(float vecA[],float vecB[],float vecC[])
{
	// 單一 thread 基準：結果正確，但在 GPU 上通常很慢。
  int i;
  for(i=0;i<ELEMENT_COUNT;i++)
	 vecC[i] = vecB[i] + vecA[i];
}

__global__ void vecAdd_gpu_kernel_1_2(float vecA[],float vecB[],float vecC[])
{
	// 兩個 thread 的基準版本：每個 thread 處理交錯元素。
  int i;
  for(i=0;i<ELEMENT_COUNT;i+=2)
	if(i+threadIdx.x < ELEMENT_COUNT)
		vecC[i+threadIdx.x] = vecB[i+threadIdx.x] + vecA[i+threadIdx.x];
}

__global__ void vecAdd_gpu_kernel_1_256(float vecA[],float vecB[],float vecC[])
{
	// 單一 block、256 threads；每個 thread 以固定步長處理多個元素。
  for(int i=0;i<ELEMENT_COUNT;i+=256)
	if(i+threadIdx.x < ELEMENT_COUNT)
		vecC[i+threadIdx.x] = vecB[i+threadIdx.x] + vecA[i+threadIdx.x];
}


__global__ void vecAdd_gpu_kernel_1_4096(float vecA[],float vecB[],float vecC[])
{
	// 僅示範用途：單一 block 啟動 4096 threads 在實體硬體上通常非法，
	// 常見上限約為每個 block 1024 threads。
  for(int i=0;i<ELEMENT_COUNT;i+=4096)
	if(i+threadIdx.x < ELEMENT_COUNT)
		vecC[i+threadIdx.x] = vecB[i+threadIdx.x] + vecA[i+threadIdx.x];
}

void vecAdd_cpu(float vecA[],float vecB[],float vecC[])
{
	// 序列化 CPU 參考實作。
  for(int i=0;i<ELEMENT_COUNT;i++)
    vecC[i] = vecB[i] + vecA[i];
}

int check_result(float h_res[],float d_res[])
{
	// 此處可用 float 精確比對：因為資料來自小整數轉 float，
	// 且 CPU/GPU 的運算形式皆為相同次序的加法。
  for(int i=0;i<ELEMENT_COUNT;i++)
    if(h_res[i] != d_res[i])
      return 0;
  return 1;
}

