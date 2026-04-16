#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <sys/time.h>

int main()
{
        double start_utime,end_utime;
        struct timeval tv, tv2;
        gettimeofday(&tv,NULL);  
        start_utime = tv.tv_sec * 1000000 + tv.tv_usec;
        
        //.....
        sleep(2);
	gettimeofday(&tv2,NULL);  
	end_utime = tv2.tv_sec * 1000000 + tv2.tv_usec;  
	printf("Parallel Execution Time=  = %f(s)\n", (end_utime - start_utime)/1000000);
}	