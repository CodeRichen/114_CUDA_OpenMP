#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#define checkCuda(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__global__ void vecMultiply(int *arr, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<size){
        for(int i = 0;i<100000;i++){
            *(arr + tid) += 10;
        }
    }
}

int main(int argc, char *argv[]){
    // Initialize
        int elementSize = 64;
    int threadsPerBlock = 32;
    int blockSize = (elementSize+threadsPerBlock-1)/threadsPerBlock;
    int *host_input_arr;
    int *device_input_arr;
    cudaHostAlloc((void**)&host_input_arr, sizeof(int) * elementSize, cudaHostAllocMapped);
    for(int i = 0;i<elementSize;i++){
        host_input_arr[i] = i;
    }
    
    cudaHostGetDevicePointer((void **)&device_input_arr,  (void *) host_input_arr , 0);
	
	cudaEvent_t startEvent, stopEvent; 
	
	checkCuda( cudaEventCreate(&startEvent) );
	checkCuda( cudaEventCreate(&stopEvent) );

	checkCuda( cudaEventRecord(startEvent, 0) );

    vecMultiply<<<blockSize, threadsPerBlock>>>(device_input_arr, elementSize);

	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );

	float time;
	checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
	printf("  Time: %f\n", time);

    for(int i = 0;i<elementSize;i++){
        printf("%d ", device_input_arr[i]);
    }
    printf("\n");
    
    cudaFree(device_input_arr);
    
    
    return 0;
}