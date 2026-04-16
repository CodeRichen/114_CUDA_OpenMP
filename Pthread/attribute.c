#include <pthread.h>
#include <sched.h>
#include <stdio.h>



void *child_thread(void *arg)
{
	int i=0;
	while(i<100000)
	{
		printf("child thread\n");
		i++;
	}
	pthread_exit(NULL);
}

int main(int argc,char *argv[ ])
{
	pthread_t child_thread_id;
 	pthread_attr_t attr;
 	struct sched_param param;
	int rc,i=0;
	int policy;
	int max_priority,min_priority;
 	
  	printf("In main: creating thread \n");
  	
  	
  	
	pthread_attr_init(&attr); /*初始化線程屬性變量*/
	pthread_attr_setschedpolicy(&attr,SCHED_OTHER);
	pthread_attr_setinheritsched(&attr,PTHREAD_EXPLICIT_SCHED); /*設置線程繼承性*/
	pthread_attr_getinheritsched(&attr,&policy); /*獲得線程的繼承性*/
	
	if(policy==PTHREAD_EXPLICIT_SCHED)
		printf("Inheritsched:PTHREAD_EXPLICIT_SCHED\n");
	if(policy==PTHREAD_INHERIT_SCHED)
		printf("Inheritsched:PTHREAD_INHERIT_SCHED\n");
	
	pthread_attr_setschedpolicy(&attr,SCHED_FIFO);/*設置線程調度策略*/
	pthread_attr_getschedpolicy(&attr,&policy);/*取得線程的調度策略*/
	
	if(policy==SCHED_FIFO)
		printf("Schedpolicy:SCHED_FIFO\n");
	if(policy==SCHED_RR)
		printf("Schedpolicy:SCHED_RR\n");
	if(policy==SCHED_OTHER)
		printf("Schedpolicy:SCHED_OTHER\n");
	
	//sched_get_priority_max(max_priority);/*獲得系統支持的線程優先權的最大值*/
	//sched_get_priority_min(min_priority);/* 獲得系統支持的線程優先權的最小值*/
	//printf("Max priority:%u\n",max_priority);
	//printf("Min priority:%u\n",min_priority);
	param.sched_priority=50;
	pthread_attr_setschedparam(&attr,&param);/*設置線程的調度參數*/
	printf("sched_priority:%u\n",param.sched_priority);/*獲得線程的調度參數*/  	
  	
  	
	rc=pthread_create(&child_thread_id,&attr,child_thread,NULL);

  	if(rc)
  	{
   		printf("ERROR; return code from pthread_create() is %d\n", rc);
   		perror("start");
   		return -1;
  	}
 
	while(i<100000)
	{
		printf("main thread\n");
		i++;
	}  		
	pthread_join(child_thread_id,NULL);
	pthread_attr_destroy(&attr);
}
