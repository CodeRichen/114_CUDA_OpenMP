#include <stdio.h>
__device__ void f() {
printf ("Device Thread %d%d%d\n", threadIdx.x,blockIdx.x,blockDim.x);
}
__global__ void kernel() {
f();
}
int main() {
kernel<<<1,4>>>();
kernel<<<4,1>>>();
if (cudaDeviceSynchronize() != cudaSuccess) {
fprintf (stderr, "Cuda call failed\n");
}
return 0;
}
