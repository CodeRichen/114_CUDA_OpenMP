#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define M 4000
#define N 4000

void omp_mxv(int m, int n, double *a, double *b,double *c)
{
	int i, j;
	
	#pragma omp parallel for shared(m,n,a,b,c) private(i,j)
	for(i=0; i<m; i++)
	{
		a[i] = 0.0;
		for(j=0; j<n; j++)
			a[i] += b[i*n+j]*c[j];
	}
}


void mxv(int m, int n, double *a, double *b,double *c)
{
	int i, j;
	
	for(i=0; i<m; i++)
	{
		a[i] = 0.0;
		for(j=0; j<n; j++)
		{
			//printf("B[%d]=%lf C[%d]=%lf \n",i*n+j,b[i*n+j],j,c[j]);
			a[i] += b[i*n+j]*c[j];
			//printf("A[%d]= %lf B[%d]=%lf C[%d]=%lf \n",i,a[i],i*n+j,b[i*n+j],j,c[j]);
		}
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
	//double A[M],D[M];
	//double B[M*N];
	//double C[N];
	double *A,*D;
	double *B;
	double *C;
	
	int i;
	
	A=(double *)malloc(sizeof(double)*M);	
	B=(double *)malloc(sizeof(double)*M*N);	
	C=(double *)malloc(sizeof(double)*N);	
	D=(double *)malloc(sizeof(double)*M);	
	srand(time(NULL));
	init_B(B);
	init_C(C);
	double time;
	time= omp_get_wtime();
	for(i=0;i<100;i++)
		mxv(M,N,A,B,C);
	printf("Sequential version uses %.16g s \n", omp_get_wtime() - time);
	
	time= omp_get_wtime();
	for(i=0;i<100;i++)
		omp_mxv(M,N,D,B,C);
	printf("Parallel version uses %.16g s \n", omp_get_wtime() - time);

	//output_B(B);
	//output_C(C);
	//output_A(A);
	//output_A(D);
}
