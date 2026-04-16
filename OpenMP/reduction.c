#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char* argv[])
{
	int sum = 0;
	int i,j;
	double time;
	time= omp_get_wtime() ;
		
	#pragma omp parallel
	{
		#pragma omp for reduction( +:sum) private(j)
		for(  i = 0; i < 10000; ++ i )
			for(  j = 0; j < 50000; ++ j )
				sum += 1;
	}
	printf("Parallel version uses %.16g s \n", omp_get_wtime() - time);
	
	printf( "%d\n",sum );



}
