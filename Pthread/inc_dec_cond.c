#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>


pthread_mutex_t count_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t count_nonzero = PTHREAD_COND_INITIALIZER;
unsigned count=0;
void *decrement_count()
{	
	while(1)
	{
		pthread_mutex_lock (&count_lock);
		if(count>0)
		{			
			count=count -1;
			printf("------------------%d\n",count);
		}
		else
		{
			printf("count is equal to 0\n");
			pthread_cond_wait( &count_nonzero, &count_lock);
		}
		
		//getchar();
		pthread_mutex_unlock (&count_lock);
	}
}

void *increment_count()
{
	int i;
	while(1)
	{
		for(i=0;i<10000;i++);
		pthread_mutex_lock(&count_lock);
		if(count==0)
			pthread_cond_signal(&count_nonzero);
		count=count+1;
		printf("+++++++++++++++++%d\n",count);
		//getchar();
		pthread_mutex_unlock(&count_lock);
	}
}


int main() 
{

	pthread_t a, b;
	pthread_create(&a, NULL, decrement_count, NULL);
	pthread_create(&b, NULL, increment_count, NULL);
	pthread_join(a,NULL);
	pthread_join(b,NULL);
	return 0;
}