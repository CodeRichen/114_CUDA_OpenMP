#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

__device__ int ParentID[4];

__global__ void child_launch(int *PID) {
    printf("ParentID=%d Child Device Thread %d\n",*PID,threadIdx.x);
}

__global__ void parent_launch() {
   __syncthreads();
   printf ("Parent Device Thread %d\n", threadIdx.x);
   ParentID[threadIdx.x]=threadIdx.x;
   child_launch<<< 1, 4 >>>(&ParentID[threadIdx.x]);
   if (cudaSuccess != cudaGetLastError()) {
     return;
   }
   // wait for child to complete
   if (cudaSuccess != cudaDeviceSynchronize()) {
     return;
   }

}

void host_launch() {
        parent_launch<<< 1, 4>>>();
        if (cudaSuccess != cudaGetLastError()) {
                return;
        }
        // wait for parent to complete
        if (cudaSuccess != cudaDeviceSynchronize()) {
                return;
        }
}

int main()
{
        host_launch();
}
   

