#include <omp.h>
#include <stdio.h>
#include <stdlib.h>



int main(int argc, char* argv[])
{
	int a[5];
	#pragma omp parallel sections
	{
		#pragma omp section
		{	printf("thread[%d] start loop1\n",omp_get_thread_num());
			int k;
			int i;
			for(i = 0; i < 5; ++ i )
			{
				a[i] = i;
				for( k = 0; k < 10000; ++ k )
				{}
			}
			printf("thread[%d] finish loop1\n",omp_get_thread_num());
		}
	
		#pragma omp section
		{
			int i;
			printf("thread[%d] start loop2\n",omp_get_thread_num());
			for( i = 0; i < 5; ++ i )
				printf( "%d\n", a[i] );
			printf("thread[%d] finish loop2\n",omp_get_thread_num());
		}
	}

}
