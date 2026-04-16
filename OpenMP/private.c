#include <omp.h>
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char* argv[])
{
	int a;
	int n=100;
	int i,j;
	printf("0x%x\n",&a);
	#pragma omp parallel for private(i) shared(a)
	for (i=0; i<n; i++)
	{
		a = i+1;
		for(j=0;j<10000;j++);
		printf("Thread %d has a value of a = %d [0x%x]for i = %d\n",omp_get_thread_num(),a,&a,i);
	}  /*-- End of parallel for --*/
	printf("a=%d\n",a);


}
