#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>

#define true 1
#define false 0
int flag = true;
void * thr_fn(void * arg) {
  while (flag){
    printf(".............................\n");
    sleep(1);
  }
  printf("thread exit\n");
}
 
int main() {
  pthread_t thread;
  if (0 != pthread_create(&thread, NULL, thr_fn, NULL)) {
    printf("error when create pthread,%d\n", errno);
    return 1;
  }
 
  char c ;
  while (flag){
    printf("ooooooooooooooooooooooooooooooooo\n");
  }  
  //while ((c = getchar()) != 'q');
 
  printf("Now terminate the thread!\n");
  flag = false;
  printf("Wait for thread to exit\n");
  pthread_join(thread, NULL);
  printf("Bye\n");
  return 0;
}