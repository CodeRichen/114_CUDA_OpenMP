#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

void print_message_function( void *ptr )
{
  char *message;
  int i=0;
  message = (char *) ptr;
  pthread_t thread=pthread_self();
  //for(i=0;i<1000000;i++)
  	printf("TID=%u:  %s \n",(unsigned int)thread, message);		
}

int main(void)
{
  pthread_t thread1, thread2;
  char message1[10] = "Hello";
  char message2[10] = "World";	

  pthread_mutex_t mutex;
  pthread_cond_t cond;
  pthread_cond_t mycond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mymutex = PTHREAD_MUTEX_INITIALIZER;
  struct timespec delay;
  delay.tv_sec = time(NULL)+1;
  delay.tv_nsec = 0;
  pthread_mutex_init(&mutex,NULL);
  pthread_cond_init(&cond,NULL);
                

  pthread_create(&thread1, NULL, (void*)&print_message_function, (void*)message1);
  printf("wait 1 seconds\n");                     
  pthread_mutex_lock(&mutex);
  pthread_cond_timedwait(&cond, &mutex, &delay);
  pthread_mutex_unlock(&mutex);
  printf("1wait finish\n");
  pthread_create(&thread2, NULL, (void*)&print_message_function, (void*)message2);
  
  delay.tv_sec = time(NULL)+1;
  delay.tv_nsec = 0;  
  printf("wait 1 seconds\n"); 
  pthread_mutex_lock(&mutex);
  pthread_cond_timedwait(&cond, &mutex, &delay);
  pthread_mutex_unlock(&mutex);  
  printf("2wait finish\n");
  //printf("end\n");                                
  return 0;
}
                