#include <omp.h>
#include <stdio.h>
#include <stdlib.h>




int main(int argc, char* argv[])
{
	int TID,i;
	int n=1000000;
	int a[1000000];
	int sum,sumLocal;
	
	sum = 0;
	for(i=0;i<n;i++)
	a[i]=1;
	
	double time;
	time= omp_get_wtime();	
	#pragma omp parallel shared(n,a,sum) private(TID,sumLocal)
	{
		 TID = omp_get_thread_num();
		sumLocal = 0;
		#pragma omp for
			for(i=0; i<n; i++)
				#pragma omp critical 
				{
				 sumLocal+= a[i];
				}
		#pragma omp critical 
		{
			sum += sumLocal;
			printf("TID=%d: sumLocal=%d sum = %d\n",TID,sumLocal,sum);
		}
	} /*-- End of parallel region --*/
	printf("Value of sum after parallel region: %d\n",sum);
	printf("Parallel version uses %.16g s \n", omp_get_wtime() - time);



}
