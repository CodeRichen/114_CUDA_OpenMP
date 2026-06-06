#include <omp.h>
#include <stdio.h>
int main()
{
     int a,b;
     #pragma omp parallel private(a,b) num_threads(8)
     {
         int ID = omp_get_thread_num();
         printf("hello(%d) %x %x %x\n", ID, &ID,&a,&b);        
         printf("world(%d) \n", ID);    
     }
     //expoer OMP_NUM_THREADS=6
}
