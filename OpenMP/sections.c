#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void funcA()
{
	printf("In funcA: this sections is executed by thread %d\n",omp_get_thread_num());
}

void funcB()
{
	printf("In funcB: this sections is executed by thread %d\n",omp_get_thread_num());
}

void funcC()
{
        printf("In funcC: this section is executed by thread  %d\n",omp_get_thread_num()); 
        
}


int main(int argc, char* argv[])
{
	int i;
	int n=100;
	int a[100];
	int b[100];
	#pragma omp parallel num_threads(4)
	{
		#pragma omp sections
		{
			#pragma omp section
				(void) funcA();
			
			#pragma omp section
				(void) funcB();
			#pragma omp section
                (void) funcC();

		} /*-- End of sections block --*/
	
	}  
}
