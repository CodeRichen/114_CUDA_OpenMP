/* begin */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int count=0;

void *count_function(void *thread_id)
{
        int i;
        for(i=0;i<1000000;i++)
                count++;
}

int main(void)
{
  pthread_t thread1, thread2;
  int error;
  error=pthread_create(&thread1, NULL, count_function, NULL);
  assert(error == 0);
  error=pthread_create(&thread2, NULL,count_function, NULL);
  assert(error == 0);
  printf("count= %d\n",count);
  //system("pause");
  return 0;
}

