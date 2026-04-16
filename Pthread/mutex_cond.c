#include <stdio.h>
#include <pthread.h>
#define BIG 5
#define LIT 1
int x,y;
pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
void* aThread(void* ptr) 
{
	pthread_mutex_lock(&mut);
	printf("RUN aThread\n");
	while(x <= y) {
		printf("x=%d <= y=%d, so waiting for bThread !\n", x, y);
		pthread_cond_wait(&cond, &mut);
	}
	printf("NOW aThread restarted !\n");
	//getchar();
	if(pthread_mutex_unlock(&mut)!=0)
	{
		printf("aThread unlock error----------------\n");
	}
	else
	       printf("aThread unlock success----------------\n");
}


void* bThread(void* ptr) 
{
	pthread_mutex_lock(&mut);
	printf("RUN bThread\n");
	printf("wake up aThread !\n");
	x = BIG;
	y = LIT;
	pthread_cond_broadcast(&cond);
	printf("after broadcast !\n");
	//getchar();
	if(pthread_mutex_unlock(&mut)!=0)
	{
		printf("bThread unlock error--------------\n");
	}
	else
	       printf("bThread unlock success----------------\n");	
	printf("after unlock !\n");
}

int main() 
{
	x = LIT;
	y = BIG;
	pthread_t a, b;
	pthread_create(&a, NULL, aThread, NULL);
	pthread_create(&b, NULL, bThread, NULL);
	pthread_join(a, NULL);
	pthread_join(b, NULL);
	return 0;
}