#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>


int count=0;
pthread_mutex_t mutex;
void aThread_function()
{
     int i;
  struct timeval tv,end;
  unsigned long long start_utime, end_utime;
  gettimeofday(&tv,NULL);
  start_utime = tv.tv_sec * 1000000 + tv.tv_usec;

    
     for(i=0;i<100;i++)
     {
//         pthread_mutex_lock(&mutex);
         count++;
//         pthread_mutex_unlock(&mutex);
     }
     
    printf("thread1\n");
	gettimeofday(&end, NULL);
	//end_utime=end.tv_sec * 1000000 + end.tv_usec;
	end_utime=end.tv_sec * 0 + end.tv_usec;
    printf("aThreads finish Time : %llu	  \n",end_utime);     
    pthread_exit(0);
}


void bThread_function()
{
     int i;
  struct timeval tv,end;
  unsigned long long start_utime, end_utime;
  gettimeofday(&tv,NULL);
  start_utime = tv.tv_sec * 1000000 + tv.tv_usec;

    
     for(i=0;i<100000000;i++)
     {
//         pthread_mutex_lock(&mutex);
         count++;
//         pthread_mutex_unlock(&mutex);
     }
     
    printf("thread2\n");
	gettimeofday(&end, NULL);
	//end_utime=end.tv_sec * 1000000 + end.tv_usec;
	end_utime=end.tv_sec * 0 + end.tv_usec;
    printf("bThreads finish Time : %llu	  \n",end_utime);     
    pthread_exit(0);
}
                                                                                         
                                                                                         


int main(void)
{
  pthread_t thread1, thread2;
  char *message1="thread1";
  char *message2="thread2";
  struct timeval tv,end;
  unsigned long long start_utime, end_utime;

  pthread_create(&thread1, NULL,(void*)&aThread_function, NULL);
  pthread_create(&thread2, NULL,(void*)&bThread_function, NULL);
  gettimeofday(&end, NULL);
  //end_utime=end.tv_sec * 1000000 + end.tv_usec;
  end_utime=end.tv_sec * 0 + end.tv_usec;
  printf("before  join thread2 Time : %llu	  \n",end_utime);     
  pthread_join(thread2,NULL);
  gettimeofday(&end, NULL);
  //end_utime=end.tv_sec * 1000000 + end.tv_usec;
  end_utime=end.tv_sec * 0 + end.tv_usec;
  printf("before join thread1 Time : %llu	  \n",end_utime);     
  pthread_join(thread1,NULL);
  gettimeofday(&end, NULL);
  //end_utime=end.tv_sec * 1000000 + end.tv_usec;
  end_utime=end.tv_sec * 0 + end.tv_usec;
  printf("after join thread1 Time : %llu	  \n",end_utime);     
  printf("count= %d\n",count);
  return 0; 
}
