#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>



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


sem_t blank_number, product_number;


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
    while(!BufferIsFull(products))
    {
        sem_wait(&blank_number);
    
        products->buffer[products->posWriteTo] = item;
        products->posWriteTo++;
        
        sem_post(&product_number);
        
        
        if (products->posWriteTo >= BUFFER_SIZE)
            products->posWriteTo = 0;
    }

}

int Consume(struct Products* products)
{
    int item;

    while(!BufferIsEmpty(products))
    {
        sem_wait(&product_number);
    
        item = products->buffer[products->posReadFrom];
        products->posReadFrom++;
        
        sem_post(&blank_number);

        if (products->posReadFrom >= BUFFER_SIZE) //如果到末尾,從頭讀取
            products->posReadFrom = 0;
    }

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
    sem_init(&blank_number, 0, BUFFER_SIZE);
    sem_init(&product_number, 0, 0);
        
    pthread_create(&producer, NULL, &ProducerThread, NULL);
    pthread_create(&consumer, NULL, &ConsumerThread, NULL);

    pthread_join(producer, (void *)&result);
    pthread_join(consumer, (void *)&result);

    sem_destroy(&blank_number);
    sem_destroy(&product_number);
        
    exit(EXIT_SUCCESS);
}
