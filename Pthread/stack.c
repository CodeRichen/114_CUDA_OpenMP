#include <pthread.h>
#include <stdio.h>

void func( void *ptr )
{
  printf ("thread_one id is %lld\n", pthread_self());
  pthread_exit(NULL);
}

int main(void)
{
  pthread_t thread_id;

  int ret, stacksize = 20480; /*thread 堆疊設定為20K，stacksize以位元組為單位。*/

  pthread_attr_t attr;
  ret = pthread_attr_init(&attr); /*初始化執行緒屬性*/
  if (ret != 0)
	return -1;

  ret = pthread_attr_setstacksize(&attr, stacksize);
  if(ret != 0)
    return -1;
  
  ret = pthread_create (&thread_id, &attr, (void*)&func, NULL);
  if(ret != 0)
    return -1;
  
  pthread_join(thread_id,NULL);

  ret = pthread_attr_destroy(&attr); /*不再使用執行緒屬性，將其銷燬*/
  if(ret != 0)
    return -1;
}