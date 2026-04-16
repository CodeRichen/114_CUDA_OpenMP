#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define Max 10

struct list
{
	struct list *next;
	int number;
	int  no;
};

// ListInit(): 輸入個數和name，使用鏈結串列將name串接起。
struct list * ListInit()
{
	struct list *h	= NULL;
	struct list *current	= NULL;
	struct list *prev	= NULL;
	int i=0;
	int no;
	int number;

	printf("No. = ");
    	scanf("%d", &no);
	srand((unsigned)time(NULL));
	
	while( 1 )
	{
		if( no == 0 )
			break;

		number=rand()%Max;

		current = (struct list *)malloc(sizeof(struct list));

		if( current == NULL )
			exit(EXIT_FAILURE);

		current->no = i++;
		current->next = NULL;	
		current->number = number;

		if( h == NULL )
		{
			h = current;
			h->no = no;
		}
		else
			prev->next=current;

		prev = current;

		no--;
	}

	return h;
}

// processwork: 輸出name。
void processwork (struct list *p)
{
	int i;
	if( p != NULL )
	{
		printf("addr %x TID = %d no= %d number = %d\n", p,omp_get_thread_num(), p->no, p->number);
	}
	for(i=0;i<1000000000;i++);
}

int main ()
{
	struct list *head=NULL;
	head = ListInit();		//只有#pragma omp parallel 時要放在這
	double time;
	time= omp_get_wtime() ;
	
	struct list *tp = head;
	while( tp )
	{
		printf("no=%d mnumber=%d\n",tp->no,tp->number);
		tp=tp->next;
	}
	#pragma omp parallel //num_threads(4) 
	{
		#pragma omp single
		{
			
			//head = ListInit();
			struct list *p = head;
			while( p )
			{
			
				#pragma omp task
				{
					processwork(p);
				}
				printf("<TID = %d> changes pointer\n", omp_get_thread_num());
				p = p->next;
			}
		}
	}
	printf("Parallel version uses %.16g s \n", omp_get_wtime() - time);
	
	return 0;
}

