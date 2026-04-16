#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#define NUM_THREADS 5


struct thread_data{
   int  thread_id;
   int  sum;
   char *message;
};

struct thread_data thread_data_array[NUM_THREADS];


void *PrintHello(void *);



int main (int argc, char *argv[])
{
 	pthread_t threads[NUM_THREADS];
 	long *taskids[NUM_THREADS];
 	
 	char messages[5][10]={"000\0","111\0","222\0","333\0","444\0"};
 	int rc , t,sum=0;
 	for(t=0;t<NUM_THREADS;t++)
 	{
  		printf("In main: creating thread %d\n", t);
       		sum=sum+t;
                thread_data_array[t].thread_id = t;
                thread_data_array[t].sum = sum;
                thread_data_array[t].message = messages[t];
      
  		rc = pthread_create(&threads[t] , NULL , PrintHello , (void *) &thread_data_array[t]);
  		if(rc)
  		{
   			printf("ERROR; return code from pthread_create() is %d\n", rc);
   			return -1;
  		}	
 	}
 	
 	 for(t=0;t<NUM_THREADS;t++)
 	         pthread_join(threads[t],NULL);
 	
}

void *PrintHello(void *threadarg)
{
        int taskid;
        char *hello_msg;
        int sum;
        struct thread_data *my_data;

        my_data = (struct thread_data *) threadarg;
        taskid = my_data->thread_id;
        sum = my_data->sum;
        hello_msg = my_data->message;
                 
 	printf("thread_ID=#%d %d %s\n", taskid,sum,hello_msg);
 	pthread_exit(NULL);
}

