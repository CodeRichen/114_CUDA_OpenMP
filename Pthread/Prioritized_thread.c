#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>

#define handle_error_en(en, msg) \
        do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)
#define max_num 10
//需要以root權限執行
			
void *func(void *a)
{
	int policy;
	struct sched_param param;
	int i=0;
    double result=0.0;
	pthread_getschedparam(pthread_self(),&policy,&param);
	
	printf("start: policy = %d, priority = %d \n",policy,param.sched_priority);
	while(i<1000000000)
	{
		result = result + sin(i) * tan(i);
		i++;
	}
	printf("end: policy = %d, priority = %d \n",policy,param.sched_priority);
	pthread_exit(NULL);
} 

int main()
{
	pthread_attr_t tattr;
	int ret;
	int newprio = 20;
	struct sched_param param;
	int i;
	pthread_t tid[max_num];
	 
	/* initialized with default attributes */
	ret = pthread_attr_init (&tattr);

	/* safe to get existing scheduling param */
	ret = pthread_attr_getschedparam (&tattr, &param);

	pthread_attr_setschedpolicy(&tattr, SCHED_FIFO);
	printf("Usage: sudo ./exe \n");


	/* with new priority specified */
	for (i = 0 ; i < max_num ; i++)
	{
		/* set the priority; others are unchanged */
		param.sched_priority = newprio+i*3;

		/* setting the new scheduling param */
		ret = pthread_attr_setschedparam (&tattr, &param);

		/* specify explicit scheduling */
		ret = pthread_attr_setinheritsched (&tattr, PTHREAD_EXPLICIT_SCHED);		
		ret = pthread_create (&tid[i], &tattr, func, NULL); 
	}
	
	for (i = 0 ; i < max_num ; i++)
	{
		pthread_join(tid[i], NULL);	
	}  
}
