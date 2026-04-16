#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#define BUFFER_SIZE 8

struct Products
{
    int buffer[BUFFER_SIZE];
    pthread_mutex_t locker;           //保證存取操作的原子性 互斥性
    pthread_cond_t notEmpty;        //是否可讀              
    pthread_cond_t notFull;                 //是否可寫
    int posReadFrom;
    int posWriteTo;
};

int BufferIsFull(struct Products* products)
{
    if ((products->posWriteTo + 1) % BUFFER_SIZE == products->posReadFrom)
    {
        return (1);
    }
    return (0);
}

int BufferIsEmpty(struct Products* products)
{
    if (products->posWriteTo == products->posReadFrom)
    {
        return (1);
    }

    return (0);
}

//製造產品。

void Produce(struct Products* products, int item)
{
    pthread_mutex_lock(&products->locker); //原子操作

    while (BufferIsFull(products))
    {
        pthread_cond_wait(&products->notFull, &products->locker);
    } //無空間可寫入

    //寫入數據
    products->buffer[products->posWriteTo] = item;
    products->posWriteTo++;

    if (products->posWriteTo >= BUFFER_SIZE)
        products->posWriteTo = 0;

    pthread_cond_signal(&products->notEmpty);     //發信
    pthread_mutex_unlock(&products->locker);      //解鎖
}

int Consume(struct Products* products)
{
    int item;

    pthread_mutex_lock(&products->locker);

    while (BufferIsEmpty(products))
    {
        pthread_cond_wait(&products->notEmpty, &products->locker);
    } //為空時持續等待,無數據可讀

    //提取數據
    item = products->buffer[products->posReadFrom];
    products->posReadFrom++;

    if (products->posReadFrom >= BUFFER_SIZE) //如果到末尾,從頭讀取
        products->posReadFrom = 0;

    pthread_cond_signal(&products->notFull); 
    pthread_mutex_unlock(&products->locker);

    return item;
}


#define END_FLAG (-1)

struct Products products;

void* ProducerThread(void* data)
{
    int i;
    for (i = 0; i < 16; ++i)
    {
        printf("producer: %d\n", i);
        Produce(&products, i);
    }
    Produce(&products, END_FLAG);
    return NULL;
}

void* ConsumerThread(void* data)
{
    int item;

    while (1)
    {
        item = Consume(&products);
        if (END_FLAG == item)
            break;
        printf("consumer: %d\n", item);
    }
    return (NULL);
}

int main(int argc, char* argv[])
{
    pthread_t producer;
    pthread_t consumer;
    int result;

    pthread_create(&producer, NULL, &ProducerThread, NULL);
    pthread_create(&consumer, NULL, &ConsumerThread, NULL);

    pthread_join(producer, (void *)&result);
    pthread_join(consumer, (void *)&result);


    exit(EXIT_SUCCESS);
}
