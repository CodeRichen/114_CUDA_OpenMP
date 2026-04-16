#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

__global__ void child_launch() {
    printf("World\n");
}

__global__ void parent_launch() {
   __syncthreads();
   printf("Hello\n");
   child_launch<<< 1, 4 >>>();
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
   

