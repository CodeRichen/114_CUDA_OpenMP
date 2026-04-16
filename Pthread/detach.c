#include <pthread.h> 
#include <stdlib.h> 
#include <stdio.h> 
#include <unistd.h> 
#include <string.h>
 
// 线程ID pthread_t ntid_joinable; pthread_t ntid_detached; // 互斥对象 
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
 
int count;
pthread_t ntid_joinable,ntid_detached;

void printids(const char *s) {
    pid_t pid;
    pthread_t tid;
 
    pid = getpid();
    tid = pthread_self();
 
    printf("%s pid %u tid %u (0x%x)\n", s, (unsigned int)pid,
        (unsigned int)tid, (unsigned int)tid);
}
 
 
// 线程函数 
void *thr_joinablefn(void *arg) {
    printids("new thread thr_joinablefn begin\n");
 
    // 加锁
    pthread_mutex_lock(&mutex);
 
    printids("new thread thr_joinablefn:\n");
 
    int i=0;
    for ( ; i<5; ++i)
    {
        printf("thr_joinablefn runing %d\n", count++);
        sleep(1);
    }
 
    // 释放互斥锁
    pthread_mutex_unlock(&mutex);
 
	pthread_exit( (void*)555);
}
 
// 线程函数 
void *thr_detachedfn(void *arg) {
    printids("new thread thr_detachedfn begin\n");
 
    // 加锁
    //pthread_mutex_lock(&mutex);
 
    int err;
    int **ret;
    err = pthread_join(ntid_joinable, (void**)ret);
    if ( err == 0 )
    {
        printf("thr_joinablefn return in thr_detachedfn %d\n", *ret);
    }
    else
    {
        printf("can't pthread_join ntid_joinable thread in thr_detachedfn :%s\n", strerror(err));
    }
 
    printids("new thread thr_detachedfn:\n");
 
    int i=0;
    for ( ; i<10; ++i)
    {
        printf("thr_detachedfn runing %d\n", count++);
        sleep(1);
    }
 
    // 释放互斥锁
    //pthread_mutex_unlock(&mutex);
 
    return ( (void*)666);
}
 
int main(void) {
    int err;
	
    count = 0;
    // 初始化互斥对象
    pthread_mutex_init(&mutex, NULL);
 
    // 创建joinable线程
    // pthread_craete第二个参数为NULL，则创建默认属性的线程，此时为PTHREAD_CREATE_JOINABLE
    err = pthread_create(&ntid_joinable, NULL, thr_joinablefn, NULL);
    if ( 0 != err )
    {
        printf("can't create ntid_joinable thread:%s\n", strerror(err));
    }
 
    // 创建detached线程
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    // 如果下句的 PTHREAD_CREATE_DETACHED 改为 PTHREAD_CREATE_JOINABLE
    // 则该线程与上面创建的线程一样
    pthread_attr_setdetachstate (&attr, PTHREAD_CREATE_DETACHED);
    err = pthread_create(&ntid_detached, &attr, thr_detachedfn, NULL);
    if ( 0 != err )
    {
        printf("can't create thr_detachedfn thread:%s\n", strerror(err));
    }
    pthread_attr_destroy (&attr);
 
    int **ret;
    err = pthread_join(ntid_joinable, (void**)ret);
    if ( err == 0 )
    {
        printf("thr_joinablefn return %d\n", *ret);
    }
    else
    {
        printf("can't pthread_join ntid_joinable thread:%s\n", strerror(err));
    }
 
    err = pthread_join(ntid_detached, (void**)ret);
    if ( err == 0 )
    {
        printf("ntid_detached return %d\n", *ret);
    }
    else
    {
        printf("can't pthread_join ntid_detached thread:%s\n", 
strerror(err));
    }
 
    pthread_mutex_destroy(&mutex);
 
    return 0;
}

