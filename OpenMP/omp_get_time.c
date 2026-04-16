#include <stdio.h>
#include <omp.h>
int main()
{
	double time;
	int j; 
	time= omp_get_wtime() ;
	#pragma omp parallel for
	for(j=0;j<10;j++)//(j=0;j<10000000;j++)
	{
		printf("TID=%d i=%d\n",omp_get_thread_num(),j);
	}
	//sleep(1);
	printf("Parallel version uses %.16g s \n", omp_get_wtime() - time);
}