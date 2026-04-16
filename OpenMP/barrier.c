#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

void print_time(int TID, char *str)
{
	time_t timep1;
	time(&timep1);
	printf("Thread %d %s %s\n",TID,str,ctime(&timep1));
	
}

int main(int argc, char* argv[])
{
	int TID;
	#pragma omp parallel private(TID)
	{
		TID = omp_get_thread_num();
		if (TID < omp_get_num_threads()/2 ) 
		  system("sleep 3");
		(void) print_time(TID,"before");
	
		#pragma omp barrier
	
		(void) print_time(TID,"after ");
	}  /*-- End of parallel region --*/




}
