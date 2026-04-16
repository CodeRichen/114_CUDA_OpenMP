#include <stdio.h>
#include <omp.h>
int main()
{
	double time;
	int j;
	int A=10;
	time= omp_get_wtime() ;
	#pragma omp parallel
	{
	  printf("hello\n");
	}
	
	#pragma omp parallel for private(j) 
	                         \shared(A)
	for(j=0;j<100;j++)
	{
		printf("A=%d (%x), TID=%d, j=%d, j's address %x\n",A,&A,omp_get_thread_num(),j,&j);
	}
	
	printf("thread number=%d \n",omp_get_num_threads());

	printf("Parallel version uses %.16g s \n", omp_get_wtime() - time);
}