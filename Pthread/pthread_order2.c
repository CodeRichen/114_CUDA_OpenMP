#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

	
void print_message_function( void *ptr )
{
  char *message;message = (char *) ptr;
  int i;
  pthread_t thread=pthread_self();
  //for(i=0;i<1000000;i++)
  {
  	printf("TID=%u: %s \n",(unsigned int)thread, message);	
  	//printf ("thread_one id is %lld\n", pthread_self());
  }
}

int main(void)
{
  pthread_t thread1, thread2;
  char *message1 = "Hello";
  char *message2 = "World";
  printf("starting\n");	
  pthread_create(&thread1, NULL,(void*)&print_message_function, (void*)message1);
  printf("Sleep 5 seconds\n"); 
  sleep(5); 
  printf("1 Sleep finish\n");
  pthread_create(&thread2, NULL,(void*)&print_message_function, (void*)message2);	 
  printf("Sleep 5 seconds\n"); 
  sleep(5);
  printf("2 Sleep finish\n");
  printf("ending\n");
  return 0;
}
                