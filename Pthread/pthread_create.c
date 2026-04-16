#include<pthread.h>
#include<stdio.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#define NUM_THREADS 50000

void *PrintHello(void *);

int main (int argc, char *argv[])
{
 	pthread_t threads[NUM_THREADS];
 	int rc , t;
 	for(t=0;t<NUM_THREADS;t++)
 	{
  		printf("In main: creating thread %d\n", t);
  		rc = pthread_create(&threads[t] , NULL , PrintHello , (void *)&t );
  		usleep(1000);
  		if(rc)
  		{
   			printf("ERROR; return code from pthread_create() is %d (%s)\n", rc,strerror(errno));
   			return -1;
  		}	
 	}
}

void *PrintHello(void *threadid)
{
 	int tid = *((int *)threadid);
 	printf("Hello World! thread #%d\n", tid);
 	pthread_exit(NULL);
}

