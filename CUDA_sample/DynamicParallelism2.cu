#include <stdio.h>
__device__ int value;

#define DEVICE_INTRINSIC_QUALIFIERS   __device__ __forceinline__

DEVICE_INTRINSIC_QUALIFIERS
unsigned int smid()
{
  unsigned int r;

  asm("mov.u32 %0, %%smid;" : "=r"(r));

  return r;
}

__global__ void child_launch(int *data) {
    printf("parentID=%d childID=%d SMID=%d\n",*data, threadIdx.x, smid());
    //while(1);
}

__global__ void parent_launch(int *data) {
    data[blockIdx.x] = blockIdx.x;
	printf("parentID=%d SMID=%d\n",blockIdx.x,smid());
    __syncthreads();
    if(threadIdx.x == 0) {
        child_launch<<< 10, 1 >>>(&data[blockIdx.x]);
        cudaDeviceSynchronize();
    }
    __syncthreads();
}

void host_launch(int *data) {
	parent_launch<<< 10, 1 >>>(data);
    if (cudaSuccess != cudaGetLastError()) {
		return;
    }
    // wait for parent to complete
    if (cudaSuccess != cudaDeviceSynchronize()) {
		return;
    }	
}

int main(int argc, char *argv[])
{
        int *data;
        cudaMallocManaged(&data, sizeof(int)*10);
        // launch parent
        host_launch(data);
        cudaFree(data);
        return 0;
}
		