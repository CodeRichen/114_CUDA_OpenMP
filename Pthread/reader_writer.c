  
  #include <stdio.h>
  #include <pthread.h>

  void reader_function(void);
  void writer_function(void);
  
  unsigned char buffer;
  int buffer_has_item = 0;
  pthread_mutex_t mutex;
  struct timespec delay;
  
int  main()
  {
     pthread_t reader;
  
  
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
			   //for(int i=0;i<100000;i++);
               buffer = item;
               printf("produce item %d\n",buffer);                
               buffer_has_item = 1;
               if(item++==-1)
                 item=0;
               
          }
		  else
             printf("buffer is full\n");

          pthread_mutex_unlock( &mutex );

     }
  }
  
  void reader_function(void)
  {
     while(1)
     {
          pthread_mutex_lock( &mutex );
          if ( buffer_has_item == 1)
          {
               printf("consume item %d\n ",buffer);
               buffer_has_item = 0;
          }
		  else
             printf("buffer is empty\n");
		  
          pthread_mutex_unlock( &mutex );
     }
  }
