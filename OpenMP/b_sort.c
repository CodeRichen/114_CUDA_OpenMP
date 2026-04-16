#include <omp.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main()
{
        int i, n=10, temp, *x, changes,even=0;
	int size=n;
        x = (int*) malloc(n * sizeof(int));
        for(i = 0; i < n; ++i)
            x[i]=n-i;
    changes = 1;
    
	while (changes)
	{  
		changes=0;
      		for (i=0;i<n;i++)
	   	{  
	   		if (x[i]>x[i+1])
         		{  
         			temp=x[i];
            			x[i]=x[i+1];
            			x[i+1]=temp;
            			changes = 1;
         		}
	   	}
      		n--;
	}  


    for(i=0;i<size;i++)
    	printf("x[i]=%d\n",i,x[i]);


    return 0;
}