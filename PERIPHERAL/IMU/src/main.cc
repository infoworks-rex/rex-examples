#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <pthread.h>
#include <getopt.h>             /* getopt_long() */
#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <sys/time.h>
#include <signal.h>
#include <linux/videodev2.h>



#include <termios.h>



#define BUF_MAX 512

const static char string_setup[] = "printmask accelp_trigger gyrop_trigger or yaw_trigger or quat_trigger or set drop\r\n";
const static char string_request[] = "p.\r\n";
const static char string_reset[] = "reset\r\n";

void *IMU_loop(void *ptr)
{
	int comfd; 
	struct termios oldtio, newtio;
	struct termios oldkey, newkey;
	char *device_name = "/dev/ttyUSB0"; // IMU Device 인식
	int need_exit = 0;
	int speed = B115200;

	float buf[BUF_MAX];

	comfd = open(device_name, O_RDWR | O_NOCTTY | O_NONBLOCK);
	if(comfd < 0)
	{
		perror(device_name);
		exit(-1);
	}

    // newtio <-- serial port setting.
    memset(&newtio, 0, sizeof(struct termios));
    newtio.c_cflag = B115200 | CS8 | CLOCAL | CREAD;
    newtio.c_iflag    = IGNPAR | ICRNL;
    newtio.c_oflag = 0;
    newtio.c_lflag = ~(ICANON | ECHO | ECHOE | ISIG);

	tcflush(comfd, TCIFLUSH);
    tcsetattr(comfd, TCSANOW, &newtio);




	write(comfd, string_setup, sizeof(string_setup));		
	printf("comport setup \n");
	int i = 0;
    while (i<6) {
		i++;
		write(comfd,  string_request, sizeof(string_request)); // IMU에데이터 요청
		read(comfd,buf,BUF_MAX); //해당데이터를 String 으로 수신
		printf(" %s \n",buf); //내용 출력
		sleep(1);

    }   


	close(comfd);
	printf("Thread END \n");

}

int main(int argc, char **argv)
{   

   	//thread setting
	int ret;
	pthread_t IMU_thread;


   
	ret = pthread_create(&IMU_thread, NULL, IMU_loop, NULL);
    if (ret < 0) {
        perror("pthread create error: ");
        exit(EXIT_FAILURE);
    }


	printf("%s main pthread start \n",__func__);    
    int status;
    ret = pthread_join(IMU_thread, NULL);
  
    return 0;

    
}


