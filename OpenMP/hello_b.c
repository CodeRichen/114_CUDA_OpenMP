#include <omp.h>
#include <stdio.h>
int main()
{
	int ID = 0;
     #pragma omp parallel private(ID) //shared(ID)
     {
         //int ID = 0;
         printf("TID=%d\n",omp_get_thread_num());
         printf("hello(%d) %x ", ID,&ID);         
         printf("world(%d) \n", ID);     
     }
}
