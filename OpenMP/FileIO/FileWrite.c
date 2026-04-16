#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define MaxFileNum 1000
#define MaxFileSize 4000000

char Letter()
{
	return rand()%(125-48+1)+48;  //ASCII 48-125
}

void WriteFile(char *filename,int filesize)
{
	FILE *fp;
	int i;
	fp=fopen(filename,"w");
	for(i=0;i<filesize;i++)
	{
		fprintf(fp,"%c",Letter());
	}
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
		filesize=rand()%MaxFileSize;
		sprintf(filename,"%5d.txt",i);
		printf("FileName=%s Size=%d\n",filename,filesize);
		WriteFile(filename,filesize);		
	}
	printf("The Execution Time of %s Threads: %.16g s \n", argv[1], omp_get_wtime() - time1);

}
