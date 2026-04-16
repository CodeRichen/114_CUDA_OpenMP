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
		//printf("dec1\n");
		if(count==0) 
		{	
			printf("dec2\n");
			//pthread_cond_wait( &count_nonzero, &count_lock);
		}
		else
			count=count -1;
		//printf("-\n");
		//getchar();
		pthread_mutex_unlock (&count_lock);
	}
}

void *increment_count()
{
	while(1)
	{
		pthread_mutex_lock(&count_lock);
		//printf("inc1\n");
		count=count+1;
		//printf("+\n");
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