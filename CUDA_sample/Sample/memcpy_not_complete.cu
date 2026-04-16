#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<time.h>




int main(int argc, char **argv)
{
	int size=1000;
	int *g;
	cudaError_t R;
	
	srand(time(0));
	// Set Random Table	
	
	int a[size],b[size];
	
	for(int k=0;k<size;k++)
	{
		a[k]=rand()%1000;
		b[k]=0;
	}
	
	printf("(1) initialized random array\n");
	printf("a[0]=%d a[1]=%d a[2]=%d \n",a[0],a[1],a[2]);
	printf("b[0]=%d b[1]=%d b[2]=%d \n",b[0],b[1],b[2]);
	
    printf("\n ========== Check cudaMalloc ==========\n");
	R = ??;
	printf("(2) Malloc g : %s\n",cudaGetErrorString(R));	
	

	printf(" ======== Check Data Transfer =========\n");
	R = ??;	
	printf("(3) cudaMemorycpy(host->device): %s\n",cudaGetErrorString(R));		
	
	R = ??;
	printf("(4) cudaMemorycpy(device->host): %s\n",cudaGetErrorString(R));		
	printf("a[0]=%d a[1]=%d a[2]=%d \n",a[0],a[1],a[2]);
	printf("b[0]=%d b[1]=%d b[2]=%d \n",b[0],b[1],b[2]);
	bool flag=true;
	for(int k=0;k<size;k++)
	{
		if(a[k]!=b[k])
		{
			flag=false;
			break;
		}
	}
	printf("(5) check a==b? : %s\n\n",flag?"pass":"error");
	
	R=??;	
	printf("(6) cudaFree: %s\n",cudaGetErrorString(R));	
	// Free Device Memory
	return 0;
}



