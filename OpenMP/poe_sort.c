#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main()
{
        int i, n=100000, tmp, *x, changes;

        x = (int*) malloc(n * sizeof(int));
        for(i = 0; i < n; ++i)
            x[i]=n-i;
    changes = 1;
    int nr = 0;
    double time;
    time= omp_get_wtime() ;
    while(changes)
    {
	    #pragma omp parallel private(tmp) shared(x)
	    {
	            nr++;
	            changes = 0;
	            #pragma omp for reduction(+:changes)
	            for(i = 0; i < n - 1; i = i + 2)
	            {
	                    if(x[i] > x[i+1] )
	                    {
	                            tmp = x[i];
	                            x[i] = x[i+1];
	                            x[i+1] = tmp;
	                            ++changes;
	                    }
	            }
	            #pragma omp for reduction(+:changes)
	            for(i = 1; i < n - 1; i = i + 2)
	            {
	                    if( x[i] > x[i+1] )
	                    {
	                            tmp = x[i];
	                            x[i] = x[i+1];
	                            x[i+1] = tmp;
	                            ++changes;
	                    }
	            }
	    }
    }
    
    
    printf("The Execution Time of %d Threads: %.16g s \n", omp_get_num_threads(), omp_get_wtime() - time);

 //for(i=0;i<n;i++)
    //	printf("x[i]=%d\n",i,x[i]);
	printf("%d %d\n",x[0],x[99999]);


    return 0;
}