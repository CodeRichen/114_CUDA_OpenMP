#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

//#define NUM_THREADS 1000000
int t=0;

void *PrintHello(void *);

int main (int argc, char *argv[])
{
 	pthread_t threads;
 	int rc;
 	while(1)
 	{
  		printf("In main: creating thread %d\n", t);
  		rc = pthread_create(&threads , NULL , PrintHello , NULL);
  		if(rc)
  		{
   			printf("ERROR; return code from pthread_create() is %d\n", rc);
   			perror("start");
   			return -1;
  		}
  		t++;	
 	}
}

void *PrintHello(void *threadid)
{
 
 	printf("Hello World!\n");
 	while(1);
}

