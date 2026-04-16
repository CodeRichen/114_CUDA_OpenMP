#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char* argv[])
{
	int n=10;
	int i;
	int a[10];
	for(i=0;i<n;i++)
	{
	    a[i]=i;
	    printf("a[%d](Mem[%x])=%d\n",i,&a[i],a[i]);
	    
    }
	#pragma omp parallel for private(a,i) 			num_threads(2)
	#pragma omp parallel for private(a) shared(i) 	num_threads(2)
	#pragma omp parallel for private(i) shared(a) 	num_threads(2)
	#pragma omp parallel for shared(a,i) 			num_threads(2)
	for (i=0; i<n; i++)
	{
		a[i] =a[i]+ i;
		printf("%x,%d, TID(%d) a[%d]=%d \t locates Mem[%x]\n",&i,i,omp_get_thread_num(),i,a[i],&a[i]);
	}  /*-- End of parallel for --*/
	
	for(i=0;i<n;i++)
	    printf("a[%d](Mem[%x])=%d\n",i,&a[i],a[i]);
		

}
