#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <iostream>
#include <windows.h>
#include<iomanip>
using namespace std;
#define YEAR_SET 1900
#define MON_SET 1

void print_time(int TID, char *str)
{
	SYSTEMTIME sys;
	GetLocalTime(&sys);
	printf("Thread %d %s ", TID, str);
	printf("%4d/%02d/%02d %02d:%02d:%02d.%03d ¬P´Á%1d\n", sys.wYear, sys.wMonth, sys.wDay, sys.wHour, sys.wMinute, sys.wSecond, sys.wMilliseconds, sys.wDayOfWeek);

}

int main(int argc, char* argv[])
{
	int TID;
#pragma omp parallel private(TID) num_threads(4)
	{
		TID = omp_get_thread_num();
		if (TID < omp_get_num_threads() / 2)
			Sleep(3000);
		(void)print_time(TID, "before");

#pragma omp barrier

		(void) print_time(TID, "after ");
	}  /*-- End of parallel region --*/


system("pause");
	//output_B(B);
	//output_C(C);
	//output_A(A);
}
