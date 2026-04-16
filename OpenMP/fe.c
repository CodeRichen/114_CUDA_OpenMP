#include <omp.h>
#include <stdio.h>
#include <stdlib.h>




int main(int argc, char* argv[])
{
	int Fe[10];
	int i;
	Fe[0] = 0;
	Fe[1] = 1;
	#pragma omp parallel for num_threads(4)
	for( i = 2; i < 10; ++ i )
		Fe[i] = Fe[i-1] + Fe[i-2];
	for( i = 0; i < 10; ++ i )
		printf( "%d," , Fe[i] );




}
