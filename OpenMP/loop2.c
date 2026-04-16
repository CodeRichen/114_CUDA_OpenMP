#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char* argv[])
{
	int i;
	int n=100;
	int a[100];
	int b[100];
	#pragma omp parallel shared(n,a,b) private(i) num_threads(4)
	{
		#pragma omp for nowait
		for (i=0; i<n; i++)
			a[i] = i;
		#pragma omp for nowait
		//for(i=n-1; i>=0; i--)
		for(i=0; i<n; i++)
			b[i] = 2 * a[i];
	}  
	for(i=0;i<n;i++)
		printf("b[%d]=%d\n",i,b[i]);
}
