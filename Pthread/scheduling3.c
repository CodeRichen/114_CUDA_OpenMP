#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>


void *func(void *a)
{

	pthread_attr_t my_attr;
	struct sched_param my_param, thread_param;
	int ret, rr_min_priority, rr_max_priority, thread_policy;
	int status;
	
	/* If the priority scheduling option is defined, set various
	* scheduling parameters. Note that it is particularly
	* important that you remember to set the inheritsched
	* attribute to PTHREAD_EXPLICIT_SCHED, or the
	* the policy and priority that you've set will be ignored!
	* The default behavior is to inherit scheduling
	* information from the creating thread.
	*/
	
	 printf("test1\n");
/*
	status = pthread_attr_init(&my_attr);
	if(status!=0)
		perror( "Init attr");
	 printf("test2\n");

	status = pthread_attr_getschedpolicy(&my_attr,&thread_policy);
	if(status!=0)
		perror( "Get policy");
	
	status = pthread_attr_getschedparam(&my_attr, &thread_param);
	if(status!=0)
		perror( "Get sched param");
	
	status = pthread_attr_setschedpolicy(&my_attr, SCHED_RR);
	if(status != 0)
		printf("Unable to set SCHED_RR policy.n");
	else {
		rr_min_priority = sched_get_priority_min(SCHED_RR);
		
		if(rr_min_priority == -1)
			perror("Get SCHED_RR min priority");
			
		rr_max_priority = sched_get_priority_max(SCHED_RR);
		if(rr_max_priority == -1)
			perror("Get SCHED_RR max priority");
			
		thread_param.sched_priority = (rr_min_priority+rr_max_priority)/2;
		printf("SCHED_RR priority range is %d to %d : using %d\n", rr_min_priority, rr_max_priority, thread_param.sched_priority);
		
		status = pthread_attr_setschedparam(&my_attr, &thread_param);
		if(status != 0)
			perror( "Set params");
		
		status = pthread_attr_setinheritsched(&my_attr, PTHREAD_EXPLICIT_SCHED);
		if(status != 0)
			perror( "Set inherit");
	}/*end of else*/
	//...
  
   pthread_exit(NULL);
} 

void *PrintHello(void *threadid)
{
 	
 	printf("Hello World! thread \n");
 	pthread_exit(NULL);
}


int main(void)
{
   int i;
   int rc;
   pthread_t thread_id[3];
   void *status1;
   pthread_attr_t my_attr;
   struct sched_param my_param, thread_param;
   int ret, rr_min_priority, rr_max_priority, thread_policy;
   int status;
	
	/* If the priority scheduling option is defined, set various
	* scheduling parameters. Note that it is particularly
	* important that you remember to set the inheritsched
	* attribute to PTHREAD_EXPLICIT_SCHED, or the
	* the policy and priority that you've set will be ignored!
	* The default behavior is to inherit scheduling
	* information from the creating thread.
	*/
	
	status = pthread_attr_init(&my_attr);
	if(status!=0)
		perror( "Init attr");
	
	status = pthread_attr_getschedpolicy(&my_attr,&thread_policy);
	if(status!=0)
		perror( "Get policy");
	
	status = pthread_attr_getschedparam(&my_attr, &thread_param);
	if(status!=0)
		perror( "Get sched param");
	
	status = pthread_attr_setschedpolicy(&my_attr, SCHED_FIFO);
	if(status != 0)
		printf("Unable to set SCHED_FIFO policy.n");
	else {
		rr_min_priority = sched_get_priority_min(SCHED_FIFO);
		
		if(rr_min_priority == -1)
			perror("Get SCHED_FIFO min priority");
			
		rr_max_priority = sched_get_priority_max(SCHED_FIFO);
		if(rr_max_priority == -1)
			perror("Get SCHED_FIFO max priority");
			
		thread_param.sched_priority = (rr_min_priority+rr_max_priority)/2;
		printf("SCHED_FIFO priority range is %d to %d : using %d\n", rr_min_priority, rr_max_priority, thread_param.sched_priority);
		
		status = pthread_attr_setschedparam(&my_attr, &thread_param);
		if(status != 0)
			perror( "Set params");
		
		status = pthread_attr_setinheritsched(&my_attr, PTHREAD_EXPLICIT_SCHED);
		if(status != 0)
			perror( "Set inherit");
		
	}/*end of else*/
	//...    
   for (i = 0 ; i < 3 ; i++)
   {
     //pthread_create(&thread_id[i],&attr_obj,(void *(*)(void *))func,(void *)NULL);
     pthread_create(&thread_id[i],&my_attr,(void *)func,(void *)NULL);
   }
   
   for (i = 0 ; i < 3 ; i++)
   {
     rc=pthread_join(thread_id[i], &status1);
     if(rc)
	{
	   printf("ERROR; return code from pthread_join() is %d\n", rc);
           exit(-1);
	}
   }
}

