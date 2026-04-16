#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define MaxFileNum 1000
#define MaxFileSize 4000000



void WriteFile(char *filename,int filesize)
{
	FILE *fp;
	int i;
	char ch=48;
		
	fp=fopen(filename,"w");
	for(i=0;i<filesize;i++)
	{
		fprintf(fp,"%c",ch);
		ch++;
		if(ch>125)
			ch=48;
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
		sprintf(filename,"%d.txt",i);
		//printf("FileName=%s Size=%d\n",filename,filesize);
		WriteFile(filename,filesize);		
	}
	printf("The Execution Time of %s Threads: %.16g s \n", argv[1], omp_get_wtime() - time1);

}
