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
#include <xf86drm.h>
#include <xf86drmMode.h>

#include "rknn_helper.h"
#include "drm_module.h"
#include "camera_helper.h"
#include "rga_helper.h"

#include "configs.h"

uint8_t *camera_buf;
uint8_t *display_buf;
uint8_t *buf;
uint8_t *out;


pthread_cond_t thread_cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex_lock = PTHREAD_MUTEX_INITIALIZER;
uint8_t status = 0;
int drm_fd, cam_fd;
drm_dev_t *modeset_list = NULL;
const char* card = CARD_DEV;
rknn_context rknn_ctx;



void *camera_loop(void *ptr)
{
	int ret;
	ssize_t size;


	ret = camera_streamon(cam_fd, V4L2_BUF_LEN);

    while (1) {
        pthread_mutex_lock(&mutex_lock);

        printf("Camera get frame helper _start \n");
        ret = camera_get_frame_helper(cam_fd, &camera_buf, &size);
        printf("Camera get frame helper _end\n");


        status = 1;
        pthread_cond_signal(&thread_cond);
        pthread_mutex_unlock(&mutex_lock);
    }   
}

void *display_loop(void *ptr)
{

    int ret;
    ssize_t size;
    rga_transform_t src, dst;
    CLEAR(src); CLEAR(dst);

    struct timeval start, end;
    double diffTime;

    src.data        = camera_buf;
    src.width   = CAM_H;
    src.height  = CAM_W;
    src.format  = CAM_FMT;
    src.direction = 0;
	//src.direction = HAL_TRANSFORM_ROT_270;

    dst.data        = modeset_list->map;
    dst.width   = DISP_H;
    dst.height  = DISP_W;
    dst.format  = DISP_FMT;
    
	//init drm
	
	printf("Display Thread Set");
    while (1) {
        pthread_mutex_lock(&mutex_lock);

	    while(!status) {
            pthread_cond_wait(&thread_cond, &mutex_lock);
        }   
		gettimeofday(&start, NULL);
        ret = rknn_run_helper(rknn_ctx, (void *)camera_buf, IMG_SIZE, out);
        rga_transform(&src, &dst);

        gettimeofday(&end, NULL);
        diffTime = (end.tv_sec - start.tv_sec) * 1000.0;      // sec to ms
        diffTime += (end.tv_usec - start.tv_usec) / 1000.0;   // us to ms

        printf("Display Done\n");
        printf("Display Time : %f\n", diffTime);

        printf("Display Done\n");
        status = 0;
        pthread_mutex_unlock(&mutex_lock);
    }
}

int main(int argc, char **argv)
{   

   	//thread setting
	int ret;
	pthread_t camera_thread, display_thread;

	camera_buf = (uint8_t *)malloc(CAM_W * CAM_H * CAM_BPP);
	display_buf = (uint8_t *)malloc(DISP_W * DISP_H * DISP_BPP);
	buf = (uint8_t *)malloc(CAM_W * CAM_H * CAM_BPP);
	out = (uint8_t *)malloc(IMG_SIZE);

	cam_fd = camera_init_helper(CAM_DEV, V4L2_BUF_LEN, CAM_W, CAM_H, CAM_FMT);
	printf("camera_init _finish \n");
	drm_fd = drm_init(card, &modeset_list);
	printf("drm_init _finish \n");
	rga_init_helper();
	printf("rga_init _finish \n");
   
	ret = pthread_create(&camera_thread, NULL, camera_loop, NULL);
    if (ret < 0) {
        perror("pthread create error: ");
        exit(EXIT_FAILURE);
    }

    ret = pthread_create(&display_thread, NULL, display_loop, NULL);
    if (ret < 0) {
        perror("pthread create error: ");
        exit(EXIT_FAILURE);
    }

	printf("%s main pthread start \n",__func__);    
    int status;
    ret = pthread_join(camera_thread, NULL);
//  ret = pthread_join(display_thread, NULL);
  
    return 0;

    
}


