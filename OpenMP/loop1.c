#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char* argv[])
{
	int i;
	int n=100;
	#pragma omp parallel shared(n) private(i) num_threads(2)
	{
		#pragma omp for
		for (i=0; i<n; i++)
			printf("Thread %d executes loop iteration%d\n",omp_get_thread_num(), i);
	}  
}
