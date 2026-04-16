#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<time.h>

#define ELEMENT_COUNT 1024*1024
#define THREADSPERBLOCK 128
#define BLOCKSPERGRID ELEMENT_COUNT/THREADSPERBLOCK 
//type : 1 thread get 1 data, 1D BLOCK ID, 1D THREAD ID
//must less then 65535

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

	float *h_vecA,*h_vecB,*h_vecResultFromDevice,*h_vecResultFromHost;
	h_vecA = (float*)malloc(sizeof(float)*ELEMENT_COUNT);
	h_vecB = (float*)malloc(sizeof(float)*ELEMENT_COUNT); 
	h_vecResultFromDevice = (float*)malloc(sizeof(float)*ELEMENT_COUNT);
	h_vecResultFromHost = (float*)malloc(sizeof(float)*ELEMENT_COUNT);
	// Create memory Space on host

	srand(time(0));
	// Set Random Table

	for(int i=0;i<ELEMENT_COUNT;i++)
	{
		h_vecA[i] = rand()%100;
		h_vecB[i] = rand()%100;
	}
	// Initialize data 

	cudaError_t R; 
	// To Get Report From Device

	float *d_vecA,*d_vecB,*d_vecC;
	printf("\n ========== Check cudaMalloc ==========\n");
	R = cudaMalloc((void**)&d_vecA,sizeof(float)*ELEMENT_COUNT);
	printf(" Malloc d_vecA : %s\n",cudaGetErrorString(R));
	R = cudaMalloc((void**)&d_vecB,sizeof(float)*ELEMENT_COUNT);
	printf(" Malloc d_vecB : %s\n",cudaGetErrorString(R));
	R = cudaMalloc((void**)&d_vecC,sizeof(float)*ELEMENT_COUNT);
	printf(" Malloc d_vecC : %s\n\n",cudaGetErrorString(R));
	// Create Memory space on device

	printf(" ======== Check Data Transfer =========\n");
	R = cudaMemcpy(d_vecA,h_vecA,sizeof(float)*ELEMENT_COUNT,cudaMemcpyHostToDevice);	
	printf(" Memory Copy d_vecA : %s\n",cudaGetErrorString(R));
	R = cudaMemcpy(d_vecB,h_vecB,sizeof(float)*ELEMENT_COUNT,cudaMemcpyHostToDevice);
	printf(" Memory Copy d_vecB : %s\n\n",cudaGetErrorString(R));
	// Copy Data from Host to Device

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Set Timmer

	cudaEventRecord(start,0);
	// Get start time
	
	//vecAdd_gpu_kernel_1_1<<<1,1>>>(d_vecA,d_vecB,d_vecC);
	//vecAdd_gpu_kernel_1_2<<<1,2>>>(d_vecA,d_vecB,d_vecC);
	//vecAdd_gpu_kernel_1_256<<<1,256>>>(d_vecA,d_vecB,d_vecC);
	//vecAdd_gpu_kernel_2_256<<<1,256>>>(d_vecA,d_vecB,d_vecC);
	//vecAdd_gpu_kernel_1_4096<<<1,4096>>>(d_vecA,d_vecB,d_vecC);
	//vecAdd_gpu_kernel_256_1<<<256,1>>>(d_vecA,d_vecB,d_vecC);
	//vecAdd_gpu_kernel_256_n<<<256,2>>>(d_vecA,d_vecB,d_vecC);
	//vecAdd_gpu_kernel_256_n<<<256,128>>>(d_vecA,d_vecB,d_vecC);
	//vecAdd_gpu_kernel_better<<<BLOCKSPERGRID,THREADSPERBLOCK>>>(d_vecA,d_vecB,d_vecC);
	//vecAdd_gpu_kernel_2_19(float vecA[],float vecB[],float vecC[]);
	//vecAdd_gpu_kernel_2_10(float vecA[],float vecB[],float vecC[]);
	
	// kernel function call

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	// Get stop time
	
	float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
  // Get Execution Time

	printf(" ============ Check Result ============\n");
  R = cudaMemcpy(h_vecResultFromDevice,d_vecC,sizeof(float)*ELEMENT_COUNT,cudaMemcpyDeviceToHost);
  printf(" Memcpy h_vecResultFromDevice : %s\n",cudaGetErrorString(R));
  // Copy Result from Device

	clock_t t1 = clock();

	vecAdd_cpu(h_vecA,h_vecB,h_vecResultFromHost);
	// host vecAdd function
	
	clock_t t2 = clock();
  float CPU_elapsedTime;
  CPU_elapsedTime = (t2-t1)/(double)(CLOCKS_PER_SEC);
  // Get Execution Time

	if(check_result(h_vecResultFromHost,h_vecResultFromDevice))
		printf(" Result Check : OK!\n\n");
	else
		printf(" Result Check : QQ!\n\n");
	// Check Result between Host and Device

	free(h_vecA);
	free(h_vecB);
	free(h_vecResultFromDevice);
	// Free Host Memory

	cudaFree(d_vecA);
	cudaFree(d_vecB);
	cudaFree(d_vecC);
	// Free Device Memory

	printf(" ======== Execution Infomation ========\n");
	printf(" Excuetion Time on GPU: %3.20f s\n",elapsedTime/1000);
	printf(" Excuetion Time on CPU: %3.20f s\n",CPU_elapsedTime);
	printf(" Speed up = %f\n",(CPU_elapsedTime/(elapsedTime/1000)));
	printf(" ======================================\n\n"); 
// Print Execution Information
	

	//system("pause");
	return 0;
}

__global__ void vecAdd_gpu_kernel_better(float vecA[],float vecB[],float vecC[])
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < ELEMENT_COUNT)
    {
        vecC[i] = vecB[i] + vecA[i];
    }

}

__global__ void vecAdd_gpu_kernel_2_19(float vecA[],float vecB[],float vecC[])
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
	for(int i=0;i<ELEMENT_COUNT;i+=1024*512)
		if(i+threadIdx.x+blockDim.x * blockIdx.x < ELEMENT_COUNT)
			vecC[i+j] = vecB[i+j] + vecA[i+j];    
}

__global__ void vecAdd_gpu_kernel_2_10(float vecA[],float vecB[],float vecC[])
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
	for(int i=0;i<ELEMENT_COUNT;i+=1024)
		if(i+threadIdx.x+blockDim.x * blockIdx.x < ELEMENT_COUNT)
			vecC[i+j] = vecB[i+j] + vecA[i+j];    
}

__global__ void vecAdd_gpu_kernel_256_1(float vecA[],float vecB[],float vecC[])
{
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
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  for(int i=0;i<ELEMENT_COUNT;i+=512)
	if(i+threadIdx.x+blockDim.x * blockIdx.x < ELEMENT_COUNT)
		vecC[i+j] = vecB[i+j] + vecA[i+j];
}

__global__ void vecAdd_gpu_kernel_2_256(float vecA[],float vecB[],float vecC[])
{
  int j = blockDim.x * blockIdx.x + threadIdx.x;
  for(int i=0;i<ELEMENT_COUNT;i+=512)
	if(i+threadIdx.x+blockDim.x * blockIdx.x < ELEMENT_COUNT)
		vecC[i+j] = vecB[i+j] + vecA[i+j];
}

__global__ void vecAdd_gpu_kernel_1_1(float vecA[],float vecB[],float vecC[])
{
  int i;
  for(i=0;i<ELEMENT_COUNT;i++)
	 vecC[i] = vecB[i] + vecA[i];
}

__global__ void vecAdd_gpu_kernel_1_2(float vecA[],float vecB[],float vecC[])
{
  int i;
  for(i=0;i<ELEMENT_COUNT;i+=2)
	if(i+threadIdx.x < ELEMENT_COUNT)
		vecC[i+threadIdx.x] = vecB[i+threadIdx.x] + vecA[i+threadIdx.x];
}

__global__ void vecAdd_gpu_kernel_1_256(float vecA[],float vecB[],float vecC[])
{
  for(int i=0;i<ELEMENT_COUNT;i+=256)
	if(i+threadIdx.x < ELEMENT_COUNT)
		vecC[i+threadIdx.x] = vecB[i+threadIdx.x] + vecA[i+threadIdx.x];
}


__global__ void vecAdd_gpu_kernel_1_4096(float vecA[],float vecB[],float vecC[])
{
  for(int i=0;i<ELEMENT_COUNT;i+=4096)
	if(i+threadIdx.x < ELEMENT_COUNT)
		vecC[i+threadIdx.x] = vecB[i+threadIdx.x] + vecA[i+threadIdx.x];
}

void vecAdd_cpu(float vecA[],float vecB[],float vecC[])
{
  for(int i=0;i<ELEMENT_COUNT;i++)
    vecC[i] = vecB[i] + vecA[i];
}

int check_result(float h_res[],float d_res[])
{
  for(int i=0;i<ELEMENT_COUNT;i++)
    if(h_res[i] != d_res[i])
      return 0;
  return 1;
}

