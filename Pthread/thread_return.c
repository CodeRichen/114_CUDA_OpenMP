#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

struct Birthday {
	int year;
	int month;
	int day;
};
void* returnThread(void* ptr) 
{
	struct Birthday* rVal = calloc(sizeof(struct Birthday), 1);
//	(*rVal).year = 1983;
//	(*rVal).month = 5;
//	(*rVal).day = 27;
	rVal->year = 1983;
	rVal->month = 5;
	rVal->day = 27;
	free(rVal);
	printf("%8x\n",rVal);
	pthread_exit((void*) rVal);
}
int main() 
{
	pthread_t th;
	struct Birthday* rec;
	printf("Main function is started !\n");
	pthread_create(&th, NULL, returnThread, NULL);
	pthread_join(th, (void**) &rec);
	printf("%8x\n",rec);
	printf("My birthday is ");
//	printf("%d-%d-%d.\n", (*rec).year, (*rec).month, (*rec).day);
	printf("%d-%d-%d.\n", rec->year, rec->month, rec->day);
	return 0;
}