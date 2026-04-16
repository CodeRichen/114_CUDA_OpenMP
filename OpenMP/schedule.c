#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char* argv[])
{
	int n=100;
	int i;
	
	#pragma omp parallel for num_threads(2) shared(n) private(i) schedule(guided,1)
	for (i=0; i<n; i++)
	   printf("Thread %d executes loop iteration%d \n",omp_get_thread_num(), i);
	/*-- End of parallel region --*/
}
