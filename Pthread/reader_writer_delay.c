  
  #include <stdio.h>
  #include <pthread.h>

  void reader_function(void);
  void writer_function(void);
  
  unsigned char buffer;
  int buffer_has_item = 0;
  pthread_mutex_t mutex;
  struct timespec delay;
  pthread_cond_t cond;
  
  struct timespec delay;
    
int  main()
  {
     pthread_t reader;
  
     delay.tv_sec = 2;
     delay.tv_nsec = 0;

  pthread_mutex_t mutex;
  //pthread_cond_t cond;
  pthread_cond_t mycond = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t mymutex = PTHREAD_MUTEX_INITIALIZER;

  pthread_mutex_init(&mutex,NULL);
  pthread_cond_init(&cond,NULL);
                  
  
     pthread_mutex_init(&mutex, NULL);
     pthread_create( &reader, NULL, (void*)&reader_function,
                    NULL);
     writer_function();
     return 0;
  }
  
  void writer_function(void)
  {
        
    char item=0;
     while(1)
     {
          pthread_mutex_lock( &mutex );
          if ( buffer_has_item == 0 )
          {
               buffer = item;
               printf("produce item %d\n",buffer);                
               buffer_has_item = 1;
               if(item++==-1)
                 item=0;
               
          }
          pthread_mutex_unlock( &mutex );
          
          delay.tv_sec = time(NULL)+1;
		  delay.tv_nsec = 0;
          pthread_mutex_lock(&mutex);
          pthread_cond_timedwait(&cond, &mutex, &delay);
          pthread_mutex_unlock(&mutex);
     }
  }
  
  void reader_function(void)
  {
     while(1)
     {
          pthread_mutex_lock( &mutex );
          if ( buffer_has_item == 1)
          {
               printf("consume item %d\n",buffer);
               buffer_has_item = 0;
          }
          pthread_mutex_unlock( &mutex );

     }
  }
