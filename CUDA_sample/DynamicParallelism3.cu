 
#include <stdio.h>

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__ void kernel(float *a, int offset)
{
  int i = offset + threadIdx.x + blockIdx.x*blockDim.x;
  float x = (float)i;
  float s = sinf(x); 
  float c = cosf(x);
  a[i] = a[i] + sqrtf(s*s+c*c);
}

float maxError(float *a, int n) 
{
  float maxE = 0;
  for (int i = 0; i < n; i++) {
    float error = fabs(a[i]-1.0f);
    if (error > maxE) maxE = error;
  }
  return maxE;
}

int main(int argc, char **argv)
{
  const int blockSize = 256, nStreams = 4;
  const int n = 4 * 10 *1024 * blockSize * nStreams;
  const int streamSize = n / nStreams;
  const int bytes = n * sizeof(float);
   
  int devId = 0;
  if (argc > 1) devId = atoi(argv[1]);

  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  checkCuda( cudaSetDevice(devId) );
  
  // allocate unified memory
  float *a;
  checkCuda( cudaMallocManaged((void**)&a, bytes));      
  
  float ms; // elapsed time in milliseconds
  
  // create events and streams
  cudaEvent_t startEvent, stopEvent, dummyEvent;
  cudaStream_t stream[nStreams];
  
  //============= baseline case - sequential transfer and execute===================
  float total_ms=0;
  for(int i=0;i<100;i++)
  {
	  checkCuda( cudaEventCreate(&startEvent) );
	  checkCuda( cudaEventCreate(&stopEvent) );
	  checkCuda( cudaEventCreate(&dummyEvent) );  
	  memset(a, 0, bytes);
	  checkCuda( cudaEventRecord(startEvent,0) );
	  kernel<<<n/blockSize, blockSize>>>(a, 0);
	  checkCuda( cudaEventRecord(stopEvent, 0) );
	  checkCuda( cudaEventSynchronize(stopEvent) );
	  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
	  // cleanup
	  checkCuda( cudaEventDestroy(startEvent) );
	  checkCuda( cudaEventDestroy(stopEvent) );
	  checkCuda( cudaEventDestroy(dummyEvent) );
      if (cudaSuccess != cudaGetLastError()) {
		return;
      }
      // wait for parent to complete
      if (cudaSuccess != cudaDeviceSynchronize()) {
		return;
      }	  
	  printf("Time for sequential transfer and execute (ms): %f\n", ms);
	  total_ms=total_ms+ms;
  }
  printf("Average Time for sequential transfer and execute (ms): %f\n", total_ms/100);
  printf("  max error: %e\n", maxError(a, n));

  //==============asynchronous version 1: loop over {copy, kernel, copy}============
  total_ms=0;
  for(int i=0;i<100;i++)
  {
	  for (int i = 0; i < nStreams; ++i)
        checkCuda( cudaStreamCreate(&stream[i]) );  
	  checkCuda( cudaEventCreate(&startEvent) );
	  checkCuda( cudaEventCreate(&stopEvent) );
	  checkCuda( cudaEventCreate(&dummyEvent) );   
	  memset(a, 0, bytes);
	  checkCuda( cudaEventRecord(startEvent,0) );
	  for (int i = 0; i < nStreams; ++i) {
		int offset = i * streamSize;
		kernel<<<streamSize/blockSize, blockSize, 0, stream[i]>>>(a, offset);
	  }
	  checkCuda( cudaEventRecord(stopEvent, 0) );
	  checkCuda( cudaEventSynchronize(stopEvent) );
	  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
	  // cleanup
	  checkCuda( cudaEventDestroy(startEvent) );
	  checkCuda( cudaEventDestroy(stopEvent) );
	  checkCuda( cudaEventDestroy(dummyEvent) );
	  for (int i = 0; i < nStreams; ++i)
		checkCuda( cudaStreamDestroy(stream[i]) );      
	  if (cudaSuccess != cudaGetLastError()) {
		return;
      }
      // wait for parent to complete
      if (cudaSuccess != cudaDeviceSynchronize()) {
		return;
      }		  
	  printf("Time for asynchronous V1 transfer and execute (ms): %f\n", ms);
	  total_ms=total_ms+ms;
  }
  printf("Average Time for sequential transfer and execute (ms): %f\n", total_ms/100);
  printf("  max error: %e\n", maxError(a, n));
  cudaFreeHost(a);

  return 0;
}