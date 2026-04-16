#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char* argv[])
{
	int n=100000;
	int a[100000];
	int b[100000];
	int i;
	#pragma omp parallel shared(n,a,b,i)//) private(i)
	{
		#pragma omp for  nowait
		for (i=0; i<n; i++)
			a[i] = i;
		#pragma omp for
		for(i=0; i<n; i++)
			b[i] = 2 * a[i];
	}  /*-- End of parallel region --*/

	for(i=0;i<n;i++)
		printf("b[%d]=%d\n",i,b[i]);
}
