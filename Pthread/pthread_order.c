#include <stdio.h>
#include <pthread.h>
void print_message_function( void *ptr )
{
  char *message;message = (char *) ptr;
  printf("%s ", message);	
}

int main(void)
{
  pthread_t thread1, thread2;
  char *message1 = "Hello";
  char *message2 = "World";	
  pthread_create(&thread1, NULL,(void*)&print_message_function, (void*)message1);
  pthread_create(&thread2, NULL,(void*)&print_message_function, (void*)message2);	
  return 0;
}
                