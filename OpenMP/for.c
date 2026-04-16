#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char* argv[])
{
	int n=10;
	int i,j=2;
	
	#pragma omp parallel num_threads(4) shared(n,i) //private(i)
	{
		#pragma omp for
		for (i=0; i<n; i++)
		printf("Thread %d executes loop iteration%d i_addr=%x n_addr=%x\n",omp_get_thread_num(), i,&i,&n);
	}  /*-- End of parallel region --*/
}
