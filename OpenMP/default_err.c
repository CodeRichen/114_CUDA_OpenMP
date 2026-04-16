#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


void Test2( int n, int m )
{
	printf( "<T:%d> - %d, %d\n", omp_get_thread_num(), n, m );
}

int main(int argc, char* argv[])
{
	int X; 
	int i;
	#pragma omp parallel default(none) 
	{ 
	                #pragma omp for  
	                for( i = 0; i < 4; ++ i ) 
	                      for( X = 0; X < 4; ++ X ) 
	                             Test2( i, X ); 
	}


}
