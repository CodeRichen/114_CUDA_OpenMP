  
  #include <stdio.h>
  #include <pthread.h>

  void reader_function(void);
  void writer_function(void);
  
  unsigned char buffer;
  int buffer_has_item = 0;
  
int  main()
  {
     pthread_t reader;
  
  
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
          if ( buffer_has_item == 0 )
          {
               buffer = item;
               printf("produce item %d\n",buffer);                
               buffer_has_item = 1;
               if(item++==-1)
                 item=0;
               
          }
     }
  }
  
  void reader_function(void)
  {
     while(1)
     {
          if ( buffer_has_item == 1)
          {
               printf("consume item %d\n ",buffer);
               buffer_has_item = 0;
          }
     }
  }
