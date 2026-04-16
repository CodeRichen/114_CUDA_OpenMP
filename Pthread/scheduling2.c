#include <pthread.h>
#include <sched.h>
#include <stdio.h>


void *func(void *a)
{
   int policy;
   struct sched_param param;

   pthread_getschedparam(pthread_self(),&policy,&param);
   printf("policy = %d, priority = %d \n",policy,param.sched_priority);
   pthread_exit(NULL);
} 

int main(void)
{
   int i;
   pthread_t thread_id[3];
   pthread_attr_t attr_obj;
   struct sched_param param;
   void *status;
   int policy;


   pthread_getschedparam(pthread_self(),&policy,&param);
   printf("policy = %d, priority = %d \n",policy,param.sched_priority);
   
   pthread_attr_init(&attr_obj);
   for (i = 0 ; i < 3 ; i++)
   {
     pthread_attr_setschedpolicy(&attr_obj,SCHED_FIFO);

     param.sched_priority = i+1;
     pthread_attr_setschedparam(&attr_obj,&param);

     pthread_create(&thread_id[i],&attr_obj,(void *(*)(void *))func,(void *)NULL);
   }
   
   for (i = 0 ; i < 3 ; i++)
   {
     pthread_join(thread_id[i], &status);	
   }  
}

