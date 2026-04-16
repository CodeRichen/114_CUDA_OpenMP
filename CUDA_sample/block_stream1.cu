 
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
  const int blockSize = 256, nStreams = 3;
  const int n = 4 * 1024 * blockSize * nStreams;
  const int streamSize = n / nStreams;
  const int streamBytes = streamSize * sizeof(float);
  const int bytes = n * sizeof(float);
   
  int devId = 0;
  if (argc > 1) devId = atoi(argv[1]);

  cudaDeviceProp prop;
  checkCuda( cudaGetDeviceProperties(&prop, devId));
  printf("Device : %s\n", prop.name);
  checkCuda( cudaSetDevice(devId) );
  
  // allocate pinned host memory and device memory
  float *a, *d_a;
  checkCuda( cudaMallocHost((void**)&a, bytes) );      // host pinned
  checkCuda( cudaMalloc((void**)&d_a, bytes) ); // device

  float ms; // elapsed time in milliseconds
  
  // create events and streams
  cudaEvent_t startEvent, stopEvent, dummyEvent;
  cudaStream_t stream[nStreams];
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) );
  checkCuda( cudaEventCreate(&dummyEvent) );

  
  //============= baseline case - sequential transfer and execute===================
  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent,0) );
  checkCuda( cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice) );
  kernel<<<n/blockSize, blockSize>>>(d_a, 0);
  checkCuda( cudaMemcpy(a, d_a, bytes, cudaMemcpyDeviceToHost) );
  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for sequential transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // ==================asynchronous version 2: ============================
  // loop over copy, loop over kernel, loop over copy
  for (int i = 0; i < nStreams-1; ++i)
    checkCuda( cudaStreamCreate(&stream[i]) );
	
  memset(a, 0, bytes);
  checkCuda( cudaEventRecord(startEvent,0) );

  int offset0 = 0 * streamSize;
  checkCuda( cudaMemcpyAsync(&d_a[offset0], &a[offset0], 
                               streamBytes, cudaMemcpyHostToDevice,
                               stream[0]) );
  int offset1 = 1 * streamSize;
  checkCuda( cudaMemcpyAsync(&d_a[offset1], &a[offset1], 
                               streamBytes, cudaMemcpyHostToDevice));
  int offset2 = 2 * streamSize;
  checkCuda( cudaMemcpyAsync(&d_a[offset2], &a[offset2], 
                               streamBytes, cudaMemcpyHostToDevice,
                               stream[1]) );							   

  kernel<<<streamSize/blockSize, blockSize, 0, stream[0]>>>(d_a, offset0);
  kernel<<<streamSize/blockSize, blockSize, 0>>>(d_a, offset1);
  kernel<<<streamSize/blockSize, blockSize, 0, stream[1]>>>(d_a, offset2); 

  checkCuda( cudaMemcpyAsync(&a[offset0], &d_a[offset0], 
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[0]) );
  checkCuda( cudaMemcpyAsync(&a[offset1], &d_a[offset1], 
                               streamBytes, cudaMemcpyDeviceToHost) );
  checkCuda( cudaMemcpyAsync(&a[offset2], &d_a[offset2], 
                               streamBytes, cudaMemcpyDeviceToHost,
                               stream[1]) );							   

  checkCuda( cudaEventRecord(stopEvent, 0) );
  checkCuda( cudaEventSynchronize(stopEvent) );
  checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
  printf("Time for asynchronous V2 transfer and execute (ms): %f\n", ms);
  printf("  max error: %e\n", maxError(a, n));

  // cleanup
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
  checkCuda( cudaEventDestroy(dummyEvent) );
  for (int i = 0; i < nStreams-1; ++i)
    checkCuda( cudaStreamDestroy(stream[i]) );
  cudaFree(d_a);
  cudaFreeHost(a);

  return 0;
}