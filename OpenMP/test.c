#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
void Test( int n )
{
	int i;
	for(i = 0; i <n; i++)
	{
		//do nothing, just waste time
	}
	printf( "%d, ", n );
}

int main(int argc, char* argv[])
{
	int i;
	#pragma omp parallel for
	for( i = 0; i <10;i++)
		Test(i);
	
}
