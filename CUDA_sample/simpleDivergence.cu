#include <stdio.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void mathKernel1( float * c) {
     int tid = blockIdx.x * blockDim.x + threadIdx.x;
     float a, b;
    a = b = 0.0f ;
     if (tid % 2 == 0 ) {
        a = 100.0f ;
    } else {
        b = 200.0f ;
    }
    c[tid] = a + b;
}

__global__ void mathKernel2( float * c) {
     int tid = blockIdx.x * blockDim.x + threadIdx.x;
     float a, b;
    a = b = 0.0f ;
     if ((tid / warpSize) % 2 == 0 ) {
        a = 100.0f ;
    } else {
        b = 200.0f ;
    }
    c[tid] = a + b;
}


int main( int argc, char ** argv) {
  // set up device 
	int dev = 0 ;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties( & deviceProp, dev);
	printf( " %s using Device %d: %s\n " , argv[ 0 ],dev, deviceProp.name);
	  // set up data size 
	int size = 64 ;
	int blocksize = 64 ;
	if (argc > 1 ) 
		blocksize = atoi(argv[ 1 ]);
	if (argc > 2 ) 
		size = atoi(argv[ 2 ]);
	printf( " Data size %d " , size);
	
	// set up execution configuration  
	dim3 block (blocksize, 1 );
	dim3 grid ((size +block.x- 1 )/block.x, 1 );
	printf( " Execution Configure (block %d grid %d)\n " ,block.x, grid.x);
	
	// allocate gpu memory 
	float * d_C;
	size_t nBytes = size * sizeof( float );
	cudaMalloc(( float **)& d_C, nBytes);
	
	float ms; // elapsed time in milliseconds
	cudaEvent_t startEvent, stopEvent;
	// run kernel 1  
	checkCuda( cudaEventCreate(&startEvent) );
	checkCuda( cudaEventCreate(&stopEvent) );
    checkCuda( cudaEventRecord(startEvent,0) );	
	mathKernel1 <<<grid, block>>> (d_C);
	cudaDeviceSynchronize();
	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
	printf("mathKernel1 <<< %4d %4d >>> elapsed %f ms \n " ,grid.x,block.x, ms);
	
	// run kernel 2
    checkCuda( cudaEventRecord(startEvent,0) );	
	mathKernel2 <<<grid, block>>> (d_C);
	cudaDeviceSynchronize();
	checkCuda( cudaEventRecord(stopEvent, 0) );
	checkCuda( cudaEventSynchronize(stopEvent) );
	checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
	printf("mathKernel2 <<< %4d %4d >>> elapsed %f ms \n " ,grid.x,block.x, ms);

	// free gpu memory and reset divece 
	cudaFree(d_C);
	cudaDeviceReset();
	return EXIT_SUCCESS;
}