#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <errno.h>

//需要以superuser權限執行

int showSchedParam(pthread_t thread)
{
  struct sched_param   param;
  int                  policy;
  int                  rc;

  printf("Get scheduling parameters\n");
  rc = pthread_getschedparam(thread, &policy, &param);
  printf("pthread_getschedparam()\n");

  printf("The thread scheduling parameters indicate:\n"
         "priority = %d\n", param.sched_priority);
  return param.sched_priority;
}

void *threadfunc(void *parm)
{
  int           rc,thePriority;

  printf("Inside secondary thread\n");
  thePriority = showSchedParam(pthread_self());
  sleep(5);  /* Sleep is not a very robust way to serialize threads */
  return NULL;
}

int main(int argc, char **argv)
{
  pthread_t             thread;
  int                   rc=0;
  struct sched_param    param;
  int                   policy = SCHED_FIFO;
  int                   theChangedPriority=0,thePriority;



  printf("Create thread using default attributes\n");
  rc = pthread_create(&thread, NULL, threadfunc, NULL);
  printf("pthread_create()\n");

  sleep(2);  /* Sleep is not a very robust way to serialize threads */

  param.sched_priority = 50;
 

  printf("Set scheduling parameters, prio=%d\n",param.sched_priority);
  rc = pthread_setschedparam(thread, policy, &param);
  if (rc) {
         printf("ERROR; return code from pthread_setschedparam is %d\n", rc);
		 perror("close"); //假設close錯誤，會印出"close: Resource temporarily unavailable"。
		 fprintf(stderr,"%d %s\\n",rc,strerror(rc));//會印出"11 Resource temporarily unavailable"。
         exit(-1);
  }
  printf("pthread_setschedparam()\n");

  /* Let the thread fill in its own last priority */
  theChangedPriority = showSchedParam(thread);
 
  if (thePriority == theChangedPriority ||
      param.sched_priority != theChangedPriority) {
      printf("The thread did not get priority set correctly, "
             "first=%d last=%d expected=%d\n",
             thePriority, theChangedPriority, param.sched_priority);
      return -1;
  }
  
  sleep(5);  /* Sleep is not a very robust way to serialize threads */
  printf("Main completed\n");
  return 0;
}

