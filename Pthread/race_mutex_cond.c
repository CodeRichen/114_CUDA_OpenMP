#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define NUM_THREADS 5

#define Target 10000



pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
void* GoRun(void* ptr) 
{
	int a;
	int move=0;
	int tid = *((int *)ptr);
	//printf("Horse[%d] ready\n",tid);
        pthread_mutex_lock(&mutex);
        printf("Horse[%d] ready++++\n",tid);
        pthread_cond_wait(&cond, &mutex);
        printf("Horse[%d] ready----\n",tid);
        //getchar();
        if(pthread_mutex_unlock(&mutex)!=0)
          printf("unlock error\n");
	
	printf("Horse[%d] go!\n",tid);
	while(move < Target) {
		move++;
		a=(rand()%50)+1;
        usleep(a);
	}
	printf("Horse[%d] achieves target\n",tid);

}



int main() 
{
	pthread_t threads[NUM_THREADS];
	int i;
	int *taskids[NUM_THREADS];
	srand(time(NULL));
	
    //if(pthread_mutex_unlock(&mutex)!=0)
    //    printf("unlock error\n");
    //else
    //    printf("unlock success\n"); 
       	    
    //printf("unlock successxxxxx\n");     
 	for(i=0;i<NUM_THREADS;i++)
 	{	
 		taskids[i] = (int *) malloc(sizeof(int));
                *taskids[i] = i; 
                printf("%d\n",*taskids[i]);		
		pthread_create(&threads[i], NULL, GoRun, (void *) taskids[i]);
	}
	sleep(1); 
	
	    pthread_mutex_lock(&mutex);
	    pthread_cond_broadcast(&cond);
	    pthread_mutex_unlock(&mutex);

	    //getchar();
	    pthread_cond_destroy(&cond);
	    pthread_mutex_destroy(&mutex);

 	for(i=0;i<NUM_THREADS;i++)
	{		
		pthread_join(threads[i],NULL);
		printf("join %d\n",*taskids[i]);
	}
	
	return 0;
}