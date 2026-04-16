#include <stdio.h>
#include <omp.h>

static omp_lock_t lock;

/*
int main()
{
    int i; 
    omp_init_lock(&lock); // ªì©l¤Æ¤¬¥¸Âê

    #pragma omp parallel for
    for (i = 0; i < 5; ++i) 
    {
        omp_set_lock(&lock); //Àò±o¤¬¥¸Âê
        printf("%d +\n", omp_get_thread_num());
	printf("%d -\n", omp_get_thread_num());
	omp_unset_lock(&lock); //ÄÀ©ñ¤¬¥¸Âê
    }

    omp_destroy_lock(&lock); //¾P·´¤¬¥¸Âê
    return 0;
}
*/


int main(int argc, char* argv[])
{
	int sum = 0;
	int i,j;
	omp_init_lock(&lock); // ªì©l¤Æ¤¬¥¸Âê
	double time;
	time= omp_get_wtime() ;	
	#pragma omp parallel
	{
		#pragma omp for private(j)
		for( i = 0; i < 10000; ++ i )
			for( j = 0; j < 50000; ++ j )
			{
				omp_set_lock(&lock); //Àò±o¤¬¥¸Âê
				sum += 1;
				//printf("sum=%d\n",sum);
				omp_unset_lock(&lock); //ÄÀ©ñ¤¬¥¸Âê
			}
	}
	omp_destroy_lock(&lock); //¾P·´¤¬¥¸Âê
	printf("Parallel version uses %.16g s \n", omp_get_wtime() - time);
	printf( "%d\n",sum );
	return 0;

}
