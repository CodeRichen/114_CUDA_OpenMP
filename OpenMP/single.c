#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char* argv[])
{
	int n=10;
	int a=1;
	int b[10];
	int i;
	double j;
	#pragma omp parallel shared(a,b) private(i) num_threads(4)
	{
		#pragma omp single
		{
			a = 10;
			printf("Single construct executed by thread %d\n",omp_get_thread_num());
			for(j=0;j<1000000000;j++);			
		}
		 /* A barrier is automatically inserted here */
		printf("After %d \n",omp_get_thread_num());
		#pragma omp for
		for(i=0; i<n; i++)
			b[i] = a;
	}  /*-- End of parallel region --*/
	
		printf("After the parallel region:\n");
		for(i=0; i<n; i++)
			printf("b[%d] = %d\n",i,b[i]);


}
