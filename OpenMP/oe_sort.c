#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main()
{
        int i, n=100000, temp, *x, changes,even=0;

        x = (int*) malloc(n * sizeof(int));
        for(i = 0; i < n; ++i)
            x[i]=n-i;
    changes = 1;
        double time;
    time= omp_get_wtime() ;
 	while(changes) 
  	{  
  		changes=0; 
     		for(i=even; i<n; i+=2) 
     		{  
     			if (x[i]>x[i+1])
         		{  
         			temp=x[i];
            		x[i]=x[i+1];
            		x[i+1]=temp;
            		changes = 1;
         		}
	   	}
     		even=1-even;
	}  
 printf("The Execution Time of 1 Thread: %.16g s \n", omp_get_wtime() - time);

    //for(i=0;i<n;i++)
    //	printf("x[i]=%d\n",i,x[i]);
	printf("%d %d\n",x[0],x[99999]);

    return 0;
}