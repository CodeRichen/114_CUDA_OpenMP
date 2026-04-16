#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

int main(){
    // 從呼叫 fork 開始, 會分成兩支行程多工進行
    pid_t PID = fork();

    switch(PID){
        // PID == -1 代表 fork 出錯
        case -1:
            perror("fork()");
            exit(-1);
        
        // PID == 0 代表是子行程
        case 0:
            printf("I'm Child process\n");
            printf("Child's PID is %d\n", getpid());
            break;
        
        // PID > 0 代表是父行程
        default:
            printf("I'm Parent process\n");
            printf("Parent's PID is %d\n", getpid());
    }

    return 0;
}