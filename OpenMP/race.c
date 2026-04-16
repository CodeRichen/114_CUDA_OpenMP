#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char* argv[])
{
	int sum = 0;
	int i,j;
	#pragma omp parallel
	{
		#pragma omp for private(j)
		for(i = 0; i < 10000; ++ i )
			for(j = 0; j < 50000; ++ j )
				sum += 1;
	}
	printf( "%d\n",sum );

}
