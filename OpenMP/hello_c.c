#include <omp.h>
#include <stdio.h>
int main()
{
     int a;
     #pragma omp parallel private(a) num_threads(8)
     {
         int ID = omp_get_thread_num();
         printf("hello(%d) %x %x\n", ID, &ID,&a);        
         printf("world(%d) \n", ID);    
     }
     //expoer OMP_NUM_THREADS=6
}
