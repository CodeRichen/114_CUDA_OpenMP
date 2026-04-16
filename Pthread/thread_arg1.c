#include<pthread.h>
#include<stdio.h>
#include <unistd.h>

void* sendMessage(void* ptr) 
{ 
	char* pStr = (char*) ptr; 
	printf("Get messages : %s\n", pStr);
	pthread_exit(NULL); 
} 

int main() 
{ 
	pthread_t th; 
	char* message = "Hello"; 
	printf("Main function is started !\n"); 
	pthread_create(&th, NULL, sendMessage, (void*)message); 
	pthread_join(th, NULL); 
	return 0; 
} 	

