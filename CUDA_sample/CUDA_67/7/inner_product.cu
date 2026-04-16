#include <stdio.h>
#include<cuda.h>

#define ELEMENT_COUNT 1024*1024
#define BLOCKSPERGRID 1024
#define THREADSPERBLOCK 1024
__global__ void vecMul_gpu_kernel(float vecA[], float vecB[], float vecC[]) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<ELEMENT_COUNT)
		??;
}

__global__ void vecSum_gpu_kernel_1(float vecC[], float partial_sum[]) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    float local_sum = 0;
	
	if(tid==0)
	{
		for(int i=0;i<THREADSPERBLOCK;i++)
			??;
		partial_sum[blockIdx.x]=local_sum;
	}
}

__global__ void vecSum_gpu_kernel_2(float vecC[], float partial_sum[]) {
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    float local_sum = 0;
	?? vec[THREADSPERBLOCK];
	
	??;
	??;
	
	if(tid==0)
	{
		for(int i=0;i<THREADSPERBLOCK;i++)
			local_sum+=vec[tid+i];
		partial_sum[blockIdx.x]=local_sum;
	}
}




void vecDot_cpu(float *S, float vecA[], float vecB[]) {
    int i;
    for (i=0; i<ELEMENT_COUNT; i++)
	{
        *S += vecB[i] * vecA[i];
		//printf("S=%f A=%f B=%f\n",*S,vecA[i],vecB[i]);
	}
}
int main (int argc, char **argv) {

    cudaSetDevice(0);

    float *h_vecA, *h_vecB, *h_psum,*h_vecC;
    float *S_gpu, *S_cpu;
    h_vecA = (float*)malloc(sizeof(float)*ELEMENT_COUNT);
    h_vecB = (float*)malloc(sizeof(float)*ELEMENT_COUNT);
	h_vecC = (float*)malloc(sizeof(float)*ELEMENT_COUNT);
    h_psum = (float*)malloc(sizeof(float)*THREADSPERBLOCK);
    S_gpu = (float*)malloc(sizeof(float));
    S_cpu = (float*)malloc(sizeof(float));
    *S_gpu = *S_cpu = 0;

    srand(time(0));

    int i;
    for(i=0; i<ELEMENT_COUNT; i++) {
        h_vecA[i] = 1;//rand()%10;
        h_vecB[i] = 1;//rand()%10;
    }

    cudaError_t R;

    float *d_vecA, *d_vecB, *d_vecC, *d_psum;
    printf("\n========== Check cudaMalloc ==========\n");
    R = cudaMalloc((void **)&d_vecA, sizeof(float)*ELEMENT_COUNT);
    printf(" Malloc d_vecA: %s\n", cudaGetErrorString(R));
    R = cudaMalloc((void **)&d_vecB, sizeof(float)*ELEMENT_COUNT);
    printf(" Malloc d_vecB: %s\n", cudaGetErrorString(R));
    R = cudaMalloc((void **)&d_vecC, sizeof(float)*ELEMENT_COUNT);
    printf(" Malloc d_vecC: %s\n", cudaGetErrorString(R));
    R = cudaMalloc((void **)&d_psum, sizeof(float)*THREADSPERBLOCK);
    printf(" Malloc d_vecC: %s\n", cudaGetErrorString(R));

    printf("========== Check Data Transfer ==========\n");
    R = cudaMemcpy(d_vecA, h_vecA, sizeof(float)*ELEMENT_COUNT, cudaMemcpyHostToDevice);
    printf(" Memory Copy d_vecA: %s\n", cudaGetErrorString(R));
    R = cudaMemcpy(d_vecB, h_vecB, sizeof(float)*ELEMENT_COUNT, cudaMemcpyHostToDevice);
    printf(" Memory Copy d_vecB: %s\n", cudaGetErrorString(R));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    vecMul_gpu_kernel<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(d_vecA, d_vecB, d_vecC);
    cudaDeviceSynchronize();
    //vecSum_gpu_kernel_1<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(d_vecC, d_psum);
	vecSum_gpu_kernel_2<<<BLOCKSPERGRID, THREADSPERBLOCK>>>(d_vecC, d_psum);
	
	cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

	R = cudaMemcpy(h_vecC, d_vecC, sizeof(float)*ELEMENT_COUNT,cudaMemcpyDeviceToHost);
	printf(" Memcpy h_vecResultFromDevice : %s\n",cudaGetErrorString(R));
	
    printf("========== Check Result ===========\n");
    R = cudaMemcpy(h_psum, d_psum, sizeof(float)*BLOCKSPERGRID, cudaMemcpyDeviceToHost);
    printf(" Memcpy h_psum: %s\n", cudaGetErrorString(R));

	for (i=0; i<BLOCKSPERGRID; i++)
	{
		//printf("h[%d]=%f\n",i,h_psum[i]);
	    *S_gpu += h_psum[i];
	}

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    vecDot_cpu(S_cpu, h_vecA, h_vecB);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float CPU_ET;
    cudaEventElapsedTime(&CPU_ET, start, stop);

    if (*S_cpu==*S_gpu)
        printf(" Result Check: OK! S_cpu = %lf, S_gpu = %lf\n\n", *S_cpu, *S_gpu);
    else {
        printf(" Result Check: not equal!\n");
        printf(" Result S_cpu = %lf, S_gpu = %lf\n\n", *S_cpu, *S_gpu);
    }

    free(h_vecA);
    free(h_vecB);
    free(h_psum);
    free(S_gpu);
    free(S_cpu);

    cudaFree(d_vecA);
    cudaFree(d_vecB);
    cudaFree(d_vecC);
    cudaFree(d_psum);

    printf("\n========== Execution Info. ===========\n");
    printf(" Execution Time on GPU: %3.20f s\n", elapsedTime/1000);
    printf(" Execution Time on CPU: %3.20f s\n", CPU_ET/1000);
    printf(" Speed up = %lf\n", (CPU_ET/elapsedTime));

    return 0;
}
