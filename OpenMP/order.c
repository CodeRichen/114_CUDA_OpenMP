#include <omp.h>
#include <stdio.h>
#include <stdlib.h>




int main(int argc, char* argv[])
{
	int TID,i;
	int n=10;
	int a[10];
	for(i=0; i<n; i++)
	 a[i]=i;
	
	#pragma omp parallel for default(none) ordered \
			private(i,TID) shared(n,a)
		for(i=0; i<n; i++)
		{
			TID = omp_get_thread_num();
	
			printf("Thread %d updates a[%d]\n",TID,i);
	
			a[i] += i;
	
			#pragma omp ordered
			{
				printf("Thread %d prints value of a[%d] =%d\n",TID,i,a[i]);
			}
		}  /*-- End of parallel for --*/



}
