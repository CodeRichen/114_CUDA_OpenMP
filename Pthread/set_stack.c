#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <limits.h>

static void *threadFunc1(void *arg)
{
    char arr[5000];
    printf("Hello fromt threadFunc1\n");
    return NULL;
}

static void *threadFunc2(void *arg)
{
    char arr[10000];
    printf("Hello fromt threadFunc2\n");
    return NULL;
}

int main(int argc, char *argv[])
{
   pthread_t t1,t2;
   pthread_attr_t thread_attr;
   int s = 0;
   size_t tmp_size=0;

   s = pthread_attr_init(&thread_attr);
   assert(s==0);

   s = pthread_attr_setstacksize(&thread_attr , PTHREAD_STACK_MIN + 0x1000);
   assert(s==0);

   s = pthread_attr_getstacksize(&thread_attr , &tmp_size );
   assert(s==0);

   printf("forced stack size of pthread is:%zu\n", tmp_size);

   s = pthread_create(&t1, &thread_attr, threadFunc1, NULL);
   assert(s==0);

   sleep(1);    /* not the "right" way, but ok for a quick test */
   s = pthread_attr_setstacksize(&thread_attr , PTHREAD_STACK_MIN + 0x4000);
   assert(s==0);

   s = pthread_attr_getstacksize(&thread_attr , &tmp_size );
   assert(s==0);

   printf("forced stack size of pthread is:%zu\n", tmp_size);
   
   s = pthread_create(&t2, &thread_attr, threadFunc2, NULL);
   assert(s==0);

   sleep(1);

   printf("Main done()\n");
   return 0;
}
