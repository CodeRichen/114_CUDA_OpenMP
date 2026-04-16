#include <stdio.h>
#include <pthread.h>
int count=0;
pthread_mutex_t mutex;
void *count_function(void *thread_id)
{
        int i;
        for(i=0;i<1000000;i++)
        {
                pthread_mutex_lock(&mutex);
                count++;
                pthread_mutex_unlock(&mutex);
         }
        pthread_exit(0);
}
int main(void)
{
  pthread_t thread1, thread2;
  char *message1="thread1";
  char *message2="thread2";
  pthread_create(&thread1, NULL,count_function, NULL);
  pthread_create(&thread2, NULL,count_function, NULL);
  pthread_join(thread1,NULL);
  pthread_join(thread2,NULL);
  printf("count= %d\n",count);
  return 0; 
}
