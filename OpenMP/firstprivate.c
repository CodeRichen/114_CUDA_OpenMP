#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char* argv[])
{
	int a[10];
	int n=2;
	int vlen=10;
	int i;
	int indx;
	int TID;	
	
	for(i=0; i<vlen; i++) 
		a[i] = -i-1;
	
	indx = 4;
	#pragma omp parallel default(none) firstprivate(indx) private(i, TID) shared(n,a) num_threads(3)
	{
		TID = omp_get_thread_num();
	
		indx += n*TID;
		for(i=indx; i<indx+n; i++)
			a[i] = TID +1;
	}  /*-- End of parallel region --*/
	
	printf("After the parallel region:\n");
	for(i=0; i<vlen; i++)
		printf("a[%d] = %d\n",i,a[i]);


}
