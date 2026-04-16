#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define MaxFileNum 1000

void ReadFile(char *filename)
{
	FILE *fp;
	int i;
	char buffer[1000];
	fp=fopen(filename,"r");
	while(fgets(buffer,1000,fp)!=NULL);
	
	fclose(fp);
}

int main(int argc, char *argv[])
{
	int i;
	char filename[30];
	int filesize;
	double time1;
	assert(argc == 2);
	omp_set_num_threads(atoi(argv[1]));
	srand((unsigned)time(NULL));
	
    time1= omp_get_wtime() ;	
	#pragma omp parallel for private(i,filesize,filename)
	for(i=0;i<MaxFileNum;i++)
	{
		sprintf(filename,"%5d.txt",i);
		printf("FileName=%s \n",filename);
		ReadFile(filename);		
	}
	printf("The Execution Time of %s Threads: %.16g s \n", argv[1], omp_get_wtime() - time1);

}
