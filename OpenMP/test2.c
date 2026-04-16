#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char* argv[])
{
	int i,j;
	#pragma omp parallel for //private(j)
	for( i = 0; i <3;i++)
	{
		for(j=0;j<3;j++)	
			printf("TID=%d i=%d j=%d \n",omp_get_thread_num(),i,j);
	}
	
}
