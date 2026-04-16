  
  #include <stdio.h>
  #include <pthread.h>

  void reader_function(void);
  void writer_function(void);
  
  unsigned char buffer;
  int buffer_has_item = 0;
  pthread_mutex_t mutex1, mutex2;
  struct timespec delay;
  
int  main()
  {
     pthread_t reader;
  
  
     pthread_mutex_init(&mutex1, NULL);
     pthread_mutex_init(&mutex2, NULL);
     
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
          pthread_mutex_lock( &mutex1 );
          if ( buffer_has_item == 0 )
          {
               buffer = item;
               printf("produce item %d\n",buffer);                
               buffer_has_item = 1;
               if(item++==-1)
                 item=0;
               
          }
          pthread_mutex_unlock( &mutex1 );
     }
  }
  
  void reader_function(void)
  {
    int i;
     while(1)
     {
          i=0;
          while(i<10000)
            i++;
          
          pthread_mutex_lock( &mutex2 );
          if ( buffer_has_item == 1)
          {
               printf("consume item %d\n ",buffer);
               buffer_has_item = 0;
          }
          pthread_mutex_unlock( &mutex2 );
     }
  }
