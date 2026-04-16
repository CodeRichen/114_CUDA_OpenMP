#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void saxpy_worksharing1(float* x, float* y, float a, int N) 
{
      int i,j;
      #pragma omp parallel for
      for (i = 0; i < N; i++) 
      {
     		printf("Thread %d executes loop iteration%d\n",omp_get_thread_num(), i);
	        y[i] = y[i]+a*x[i];
	        for(j=0;j<1000000000;j++);
      }
}

void saxpy_worksharing2(float* x, float* y, float a, int N) 
{
      int i,j;
      #pragma omp parallel for
      for (i = 0; i < N; i++) 
      {
      	printf("1Thread %d=>%d\n",omp_get_thread_num(), i);
       	#pragma omp task
		{
	      		printf("Thread %d executes loop iteration%d\n",omp_get_thread_num(), i);
	         	y[i] = y[i]+a*x[i];
	         	for(j=0;j<1000000000;j++);
		}
		printf("2Thread %d=>%d\n",omp_get_thread_num(), i);
      }
}

int main ()
{
	double time;
	int N=100;
	float x[100];
	float y[100];
	float a=2;
	int i;
	
	time= omp_get_wtime();
	for(i=0;i<N;i++)
	{
		x[i]=1;
		y[i]=2;
	}
	saxpy_worksharing2(x,y,a,N);
	printf("Parallel version uses %.16g s \n", omp_get_wtime() - time);
	
	return 0;
}

