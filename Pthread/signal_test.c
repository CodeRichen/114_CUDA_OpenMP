#include <signal.h>
#include <unistd.h>
#include <stdio.h>

void sigroutine(int dunno) { /* 信號處理常式，其中dunno將會得到信號的值 */
   switch (dunno) {
        case SIGUSR1:
        	printf("Get a signal -- SIGUSR1 \n");
        	break;
        case SIGUSR2:
        	printf("Get a signal -- SIGUSR2 \n");
        	break;
        defalut:
        	printf("Get a signal %d\n",dunno);
        	break;
   }
   return;
}

int main() {
        printf("process id is %d \n",getpid());
        signal(SIGHUP, sigroutine); //* 下面設置三個信號的處理方法
        signal(SIGINT, sigroutine);
        signal(SIGQUIT, sigroutine);
        pause();
}
