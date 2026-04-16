#include <omp.h>
#include <stdio.h>
int main()
{
     	#pragma omp parallel num_threads(6)	
	{
		printf("The parallel region is executed by thread %d\n",omp_get_thread_num());
		if( omp_get_thread_num() == 2 )
			printf("Thread %d does things differently\n",omp_get_thread_num());
	}
}
