#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
void Test2( int n, int m )
{
	printf("＜T:%d＞ - %d, %d\n", omp_get_thread_num(), n, m );
}
	
int main(int argc, char* argv[])
{
	int i, j;
	#pragma omp parallel for shared(i) private(j)
	for( i = 0; i < 3; ++ i )
		for( j = 0; j < 3; ++ j )
				Test2( i, j );
				
	
	
}
