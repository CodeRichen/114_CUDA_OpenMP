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
__global__ void vecAdd_gpu_kernel_2_10_a(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_2_10_b(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_2_10_c(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_2_18_a(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_2_18_b(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_2_19_a(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_2_19_b(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_better_a(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_better_b(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_better_c(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_better_d(float vecA[],float vecB[],float vecC[]);
__global__ void vecAdd_gpu_kernel_better_e(float vecA[],float vecB[],float vecC[]);

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

	// 改用 user 指定的一系列 (grid, block) 配對進行測試
	int grids[] = {
		1,1,1,2,4,256,256,256,
		1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576
	};
	int blocks[] = {
		1,2,256,256,256,1,2,4,
		1024,512,256,128,64,32,16,8,4,2,1
	};
	int configs = sizeof(grids)/sizeof(grids[0]);
	double *times = (double*)malloc(sizeof(double)*configs);
	int *ok = (int*)malloc(sizeof(int)*configs);
	for (int i=0;i<configs;i++){ times[i] = -1.0; ok[i]=0; }

	// Query device limits
	int dev; cudaGetDevice(&dev);
	cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
	int maxThreadsPerBlock = prop.maxThreadsPerBlock;

	// 為每個測試配置選擇對應的 kernel，展示不同策略的效能差異
	int successful = 0;
	for (int i=0;i<configs;i++){
		int g = grids[i];
		int b = blocks[i];
		printf("\n[Test %d] <<<%d,%d>>>\n", i+1, g, b);
		if (b > maxThreadsPerBlock){
			printf("  Skip: threads_per_block=%d > device max %d\n", b, maxThreadsPerBlock);
			continue;
		}
		cudaMemset(d_vecC, 0, sizeof(float)*ELEMENT_COUNT);
		cudaEvent_t kstart, kstop; cudaEventCreate(&kstart); cudaEventCreate(&kstop);
		cudaEventRecord(kstart,0);
		
		// 根據測試序號選擇對應的 kernel
		// 策略：少線程用小步長 kernel；多線程用大步長或 grid-stride
		switch(i) {
			case 0: vecAdd_gpu_kernel_1_1<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<1,1>>> 1執行緒
			case 1: vecAdd_gpu_kernel_1_2<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<1,2>>> 2執行緒
			case 2: vecAdd_gpu_kernel_1_256<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<1,256>>> 256執行緒單block
			case 3: vecAdd_gpu_kernel_256_1<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<2,256>>> 512執行緒, 步長256
			case 4: vecAdd_gpu_kernel_256_2<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<4,256>>> 1K執行緒, 步長512
			case 5: vecAdd_gpu_kernel_256_1<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<256,1>>> 256執行緒, 步長256
			case 6: vecAdd_gpu_kernel_256_2<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<256,2>>> 512執行緒, 步長512
			case 7: vecAdd_gpu_kernel_2_10_a<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<256,4>>> 1K執行緒, 步長1024
			case 8: vecAdd_gpu_kernel_2_10_b<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<1024,1024>>> 1M執行緒, 步長1024 (多迴圈)
			case 9: vecAdd_gpu_kernel_2_10_c<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<2048,512>>> 1M執行緒, 步長1024
			case 10: vecAdd_gpu_kernel_2_18_a<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<4096,256>>> 1M執行緒, 步長262K
			case 11: vecAdd_gpu_kernel_2_18_b<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<8192,128>>> 1M執行緒, 步長262K
			case 12: vecAdd_gpu_kernel_2_19_a<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<16384,64>>> 1M執行緒, 步長524K
			case 13: vecAdd_gpu_kernel_2_19_b<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<32768,32>>> 1M執行緒, 步長524K
			case 14: vecAdd_gpu_kernel_better_a<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<65536,16>>> 1M執行緒, 超大grid用grid-stride
			case 15: vecAdd_gpu_kernel_better_b<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<131072,8>>> 1M執行緒, 超大grid
			case 16: vecAdd_gpu_kernel_better_c<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<262144,4>>> 1M執行緒, 超大grid
			case 17: vecAdd_gpu_kernel_better_d<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<524288,2>>> 1M執行緒, 超大grid
			case 18: vecAdd_gpu_kernel_better_e<<<g,b>>>(d_vecA,d_vecB,d_vecC); break;  // <<<1048576,1>>> 1M執行緒, 極大grid
			default: vecAdd_gpu_kernel_better<<<g,b>>>(d_vecA,d_vecB,d_vecC);
		}
		
		cudaError_t launchErr = cudaGetLastError();
		if (launchErr != cudaSuccess){
			printf("  Launch failed: %s\n", cudaGetErrorString(launchErr));
			cudaEventDestroy(kstart); cudaEventDestroy(kstop);
			continue;
		}
		cudaError_t syncErr = cudaDeviceSynchronize();
		if (syncErr != cudaSuccess){
			printf("  Sync failed: %s\n", cudaGetErrorString(syncErr));
			cudaEventDestroy(kstart); cudaEventDestroy(kstop);
			continue;
		}
		cudaEventRecord(kstop,0); cudaEventSynchronize(kstop);
		float ms=0.0f; cudaEventElapsedTime(&ms,kstart,kstop);
		times[i] = (double)ms/1000.0;
		R = cudaMemcpy(h_vecResultFromDevice,d_vecC,sizeof(float)*ELEMENT_COUNT,cudaMemcpyDeviceToHost);
		printf("  Memcpy result: %s\n", cudaGetErrorString(R));
		if (check_result(h_vecResultFromHost,h_vecResultFromDevice)){
			printf("  Result: OK  time=%3.6fs\n", times[i]);
			ok[i]=1; successful++;
		} else {
			printf("  Result: QQ (mismatch)  time=%3.6fs\n", times[i]);
			ok[i]=0;
		}
		cudaEventDestroy(kstart); cudaEventDestroy(kstop);
	}

	// 統計：找最快、最慢、平均（只計成功的）
	double sum=0; double minT=1e300; double maxT=-1.0; int minIdx=-1, maxIdx=-1; int count=0;
	for (int i=0;i<configs;i++){
		if (!ok[i]) continue;
		double t = times[i]; sum += t; count++;
		if (t < minT){ minT=t; minIdx=i; }
		if (t > maxT){ maxT=t; maxIdx=i; }
	}
	double avg = (count>0)? (sum/count) : -1.0;

	// 完成測試，整理並印出統計結果（只統計成功的測試）
	free(h_vecA);
	free(h_vecB);
	free(h_vecResultFromDevice);

	cudaFree(d_vecA);
	cudaFree(d_vecB);
	cudaFree(d_vecC);

	printf("\n ======== Test Summary ========\n");
	if (count == 0) {
		printf(" No successful GPU tests.\n");
	} else {
		printf(" Successful tests: %d / %d\n", count, configs);
		printf(" Fastest: <<<%d,%d>>> %3.6fs\n", grids[minIdx], blocks[minIdx], minT);
		printf(" Slowest: <<<%d,%d>>> %3.6fs\n", grids[maxIdx], blocks[maxIdx], maxT);
		printf(" Average (successful): %3.6fs\n", avg);
		printf(" CPU time: %3.6fs\n", CPU_elapsedTime);
		printf(" CPU/GPU ratio per test (CPU_time / GPU_time):\n");
		for (int i=0;i<configs;i++){
			if (!ok[i]){
				printf("  [Test %d] <<<%d,%d>>> skipped/failed\n", i+1, grids[i], blocks[i]);
				continue;
			}
			if (times[i] > 0.0) {
				printf("  [Test %d] <<<%d,%d>>> %3.6f\n", i+1, grids[i], blocks[i], CPU_elapsedTime / times[i]);
			} else {
				printf("  [Test %d] <<<%d,%d>>> invalid time\n", i+1, grids[i], blocks[i]);
			}
		}
		if (maxT > 0.0) {
			printf(" Slowest GPU vs CPU: ratio CPU/SlowGPU = %3.6f\n", CPU_elapsedTime / maxT);
		}
	}
	printf(" ================================\n\n");

	return 0;
}

__global__ void vecAdd_gpu_kernel_better(float vecA[],float vecB[],float vecC[])
{
	// 改為 grid-stride loop，讓任意 (grid, block) 組合都能正確覆蓋整個陣列。
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = idx; i < ELEMENT_COUNT; i += stride) {
		vecC[i] = vecB[i] + vecA[i];
	}
}

__global__ void vecAdd_gpu_kernel_better_a(float vecA[],float vecB[],float vecC[])
{
	// 與 vecAdd_gpu_kernel_better 相同，但用於不同測試 case。
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = idx; i < ELEMENT_COUNT; i += stride) {
		vecC[i] = vecB[i] + vecA[i];
	}
}

__global__ void vecAdd_gpu_kernel_better_b(float vecA[],float vecB[],float vecC[])
{
	// 與 vecAdd_gpu_kernel_better 相同，但用於不同測試 case。
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = idx; i < ELEMENT_COUNT; i += stride) {
		vecC[i] = vecB[i] + vecA[i];
	}
}

__global__ void vecAdd_gpu_kernel_better_c(float vecA[],float vecB[],float vecC[])
{
	// 與 vecAdd_gpu_kernel_better 相同，但用於不同測試 case。
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = idx; i < ELEMENT_COUNT; i += stride) {
		vecC[i] = vecB[i] + vecA[i];
	}
}

__global__ void vecAdd_gpu_kernel_better_d(float vecA[],float vecB[],float vecC[])
{
	// 與 vecAdd_gpu_kernel_better 相同，但用於不同測試 case。
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = idx; i < ELEMENT_COUNT; i += stride) {
		vecC[i] = vecB[i] + vecA[i];
	}
}

__global__ void vecAdd_gpu_kernel_better_e(float vecA[],float vecB[],float vecC[])
{
	// 與 vecAdd_gpu_kernel_better 相同，但用於不同測試 case。
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;
	for (unsigned int i = idx; i < ELEMENT_COUNT; i += stride) {
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

__global__ void vecAdd_gpu_kernel_2_19_a(float vecA[],float vecB[],float vecC[])
{
	// 與 vecAdd_gpu_kernel_2_19 相同，但用於不同測試 case。
    int j = blockDim.x * blockIdx.x + threadIdx.x;
	for(int i=0;i<ELEMENT_COUNT;i+=1024*512)
		if(i+threadIdx.x+blockDim.x * blockIdx.x < ELEMENT_COUNT)
			vecC[i+j] = vecB[i+j] + vecA[i+j];
}

__global__ void vecAdd_gpu_kernel_2_19_b(float vecA[],float vecB[],float vecC[])
{
	// 與 vecAdd_gpu_kernel_2_19 相同，但用於不同測試 case。
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

__global__ void vecAdd_gpu_kernel_2_18_a(float vecA[],float vecB[],float vecC[])
{
	// 與 vecAdd_gpu_kernel_2_18 相同，但用於不同測試 case。
    int j = blockDim.x * blockIdx.x + threadIdx.x;
	for(int i=0;i<ELEMENT_COUNT;i+=1024*256)
		if(i+threadIdx.x+blockDim.x * blockIdx.x < ELEMENT_COUNT)
			vecC[i+j] = vecB[i+j] + vecA[i+j];
}

__global__ void vecAdd_gpu_kernel_2_18_b(float vecA[],float vecB[],float vecC[])
{
	// 與 vecAdd_gpu_kernel_2_18 相同，但用於不同測試 case。
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

__global__ void vecAdd_gpu_kernel_2_10_a(float vecA[],float vecB[],float vecC[])
{
	// 與 vecAdd_gpu_kernel_2_10 相同，但用於不同測試 case。
    int j = blockDim.x * blockIdx.x + threadIdx.x;
	for(int i=0;i<ELEMENT_COUNT;i+=1024)
		if(i+threadIdx.x+blockDim.x * blockIdx.x < ELEMENT_COUNT)
			vecC[i+j] = vecB[i+j] + vecA[i+j];
}

__global__ void vecAdd_gpu_kernel_2_10_b(float vecA[],float vecB[],float vecC[])
{
	// 與 vecAdd_gpu_kernel_2_10 相同，但用於不同測試 case。
    int j = blockDim.x * blockIdx.x + threadIdx.x;
	for(int i=0;i<ELEMENT_COUNT;i+=1024)
		if(i+threadIdx.x+blockDim.x * blockIdx.x < ELEMENT_COUNT)
			vecC[i+j] = vecB[i+j] + vecA[i+j];
}

__global__ void vecAdd_gpu_kernel_2_10_c(float vecA[],float vecB[],float vecC[])
{
	// 與 vecAdd_gpu_kernel_2_10 相同，但用於不同測試 case。
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

