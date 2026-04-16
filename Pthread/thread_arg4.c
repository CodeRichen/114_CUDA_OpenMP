#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
/*
void* returnThread(void* ptr) 
{
	int *a;
	int b=1;
	a=(int *)malloc(sizeof(int));
	a= *((int *)ptr);
	printf("1: %d %8x\n",a,&a);
	a=b;
	pthread_exit((void*) a);
}

int main() 
{
	pthread_t th;
	int rec=0,t=11;
	pthread_create(&th, NULL , returnThread , (void *)&t );
	pthread_join(th, (void**) &rec);
	printf("%d\n",rec);
	return 0;
}

*/
void* returnThread(void* ptr) 
{
	int a,b=1;
	a= *((int *)ptr);
	a=b;
	pthread_exit((void*) &a);
}
int main() 
{
	pthread_t th;
	int *rec,t=10;
	pthread_create(&th, NULL , returnThread , (void *)&t );
	pthread_join(th, (void**) &rec);
	printf("%d\n",rec);
	return 0;
}
