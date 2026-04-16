#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define M 100
#define N 200

void mxv(int m, int n, double *a, double *b,double *c)
{
	int i, j;
	
	for(i=0; i<m; i++)
	{
		a[i] = 0.0;
		for(j=0; j<n; j++)
			a[i] += b[i*n+j]*c[j];
	}
}



void init_B(double *B)
{
	int i,j;
	for(i=0;i<M;i++)
	{
		for(j=0;j<N;j++)
		{
			B[i*N+j]=(double)((rand()%100)+1)/(double)50;
		}
	}
}

void init_C(double *C)
{
	int j;
	for(j=0;j<N;j++)
		C[j]=(double)((rand()%100)+1)/(double)50;
}

void output_C(double *C)
{
	int j;
	for(j=0;j<N;j++)
		printf("%lf\n",C[j]);
}

void output_B(double *B)
{	
	int i,j;
	for(i=0;i<M;i++)
	{
		for(j=0;j<N;j++)
		{
			printf("%lf\n",B[i*N+j]);
		}
	}
}


void output_A(double *A)
{
	int j;
	for(j=0;j<M;j++)
		printf("%lf\n",A[j]);
}


int main()
{
	double A[M];
	double B[M*N];
	double C[N];
	srand(time(NULL));
	init_B(B);
	init_C(C);
	mxv(M,N,A,B,C);
	
	output_B(B);
	output_C(C);
	output_A(A);
}
