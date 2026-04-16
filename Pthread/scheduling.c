#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

//#define NUM_THREADS 1000000
int t=0;

void *PrintHello(void *);

int main (int argc, char *argv[])
{
 	pthread_t threads;
 	pthread_attr_t tattr;
 	int rc;
 	
 	pthread_attr_init(&tattr);
 	pthread_attr_setschedpolicy(&tattr,SCHED_OTHER);
  	printf("In main: creating thread %d\n", t);
  	rc = pthread_create(&threads , &tattr, PrintHello , NULL);
  	if(rc)
  	{
   		printf("ERROR; return code from pthread_create() is %d\n", rc);
   		perror("start");
   		return -1;
  	}
  	while(1)
  	   printf("Main thread!\n");

	pthread_attr_destroy(&tattr);
}

void *PrintHello(void *threadid)
{
 	while(1)
 	  printf("child thread!\n");
}

