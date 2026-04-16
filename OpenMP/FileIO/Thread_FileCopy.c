#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define MaxFileNum 1000

void CopyFile(char *filename)
{
	FILE *fp1,*fp2;
	int i;
	char outfilename[30];
	char ch;
	
	fp1=fopen(filename,"r");
	
	
	sprintf(outfilename,"./Dest/%s",filename);
	fp2=fopen(outfilename,"w");
	if(fp1==NULL || fp2==NULL)
	{
		printf("fopen error \n");
		exit(0);
	}	
	
	printf("copy %s to %s \n",filename,outfilename);
	while ((ch = fgetc(fp1)) != EOF)
      fputc(ch, fp2);
	
	fclose(fp1);
	fclose(fp2);
}

int main(int argc, char *argv[])
{
	int i;
	char filename[30];
	int filesize;
	double time1;
	assert(argc == 2);
	omp_set_num_threads(atoi(argv[1]));

	
    time1= omp_get_wtime() ;	
	#pragma omp parallel for private(i,filesize,filename)
	for(i=0;i<MaxFileNum;i++)
	{
		sprintf(filename,"%d.txt",i);
		CopyFile(filename);		
	}
	printf("The Execution Time of %s Threads: %.16g s \n", argv[1], omp_get_wtime() - time1);

}
