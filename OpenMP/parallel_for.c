#include <omp.h>
#include <stdio.h>
int main()
{
	int i=0,n=12;
     	#pragma omp parallel num_threads(6)	
	{
		printf("1_Thread %d executes\n",omp_get_thread_num());
		#pragma omp for
		for (i=0; i<n; i++)
			printf("2_Thread %d executes loop iteration%d\n",omp_get_thread_num(), i);
	}
}
