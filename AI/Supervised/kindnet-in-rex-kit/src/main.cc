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

//common/include
#include "rknn_helper.h"
#include "drm_module.h"
#include "camera_helper.h"
#include "rga_helper.h"
#include "configs.h"
#include "rknn_helper.h"
//inc
//extern "C"{
#include "ssd.h"
//};
uint8_t *camera_buf;
uint8_t *display_buf;
uint8_t *buf;
uint8_t *out;

ssd_ctx ctx; //for ssd data save

pthread_cond_t thread_cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t mutex_lock = PTHREAD_MUTEX_INITIALIZER;
uint8_t status = 0;
int drm_fd, cam_fd;
drm_dev_t *modeset_list = NULL;
const char* card = CARD_DEV;

pthread_key_t a_key;

#define VIDEO_PACKET_SIZE CAM_H*CAM_W*CAM_CH
#define MODEL_NAME	"/kindnet.rknn"
uint8_t buff[600 * 400 * 3];


void *ssd_loop(void *ptr)
{	


	void* dst_v = NULL;
	dst_v = malloc(VIDEO_PACKET_SIZE);
	
	int ret;
	c_RkRgaInit();
	
	rga_info_t r_src;	
	rga_info_t r_dst;	
	
	memset(&r_src, 0, sizeof(rga_info_t));
	r_src.fd = -1;
	r_src.mmuFlag = 1;
	r_src.virAddr = camera_buf;
	memset(&r_dst, 0, sizeof(rga_info_t));
	r_dst.fd = -1;
	r_dst.mmuFlag = 1;
	r_dst.virAddr = dst_v;
	r_src.rotation = 0;
	
	//FILE *image = fopen("k1.raw", "rb");
	//ret = fread(buff, sizeof(uint8_t), 600 * 400 * 3, image);	
	//fclose(image);

	ssd_init(MODEL_NAME, &ctx);
	printf("init rknn complete \n");	
	while(1){
		rga_set_rect(&r_src.rect, 0, 0, CAM_W, CAM_H, CAM_W, CAM_H, RK_FORMAT_YCrCb_420_P);
		rga_set_rect(&r_dst.rect, 0, 0, 600, 400, 600, 400, RK_FORMAT_RGB_888 );
		ret = c_RkRgaBlit(&r_src, &r_dst, NULL);

		ssd_run(&ctx,(uint8_t*)dst_v,400,600,400 * 600 * 3);

		//ssd_run(&ctx,buff,400,600,400 * 600 * 3);
		
	}

} 


void *camera_loop(void *ptr)
{
	int ret;
	ssize_t size;


	ret = camera_streamon(cam_fd, V4L2_BUF_LEN);

    while (1) {
	pthread_mutex_lock(&mutex_lock);
        ret = camera_get_frame_helper(cam_fd, &camera_buf, &size);

        status = 1;
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

    src.data = display_kindnet();
    src.width = 400;
    src.height = 600;
    src.format = RK_FORMAT_RGB_888;
    src.direction = 0;

    dst.data    = modeset_list->map;
    dst.width   = DISP_H;
    dst.height  = DISP_W;
    dst.format  = RK_FORMAT_RGBA_8888;
   
	
	printf("Display Thread Set\n");
    while (1) {

	gettimeofday(&start, NULL);


        rga_transform(&src, &dst);

        gettimeofday(&end, NULL);
        diffTime = (end.tv_sec - start.tv_sec) * 1000.0;      // sec to ms
        diffTime += (end.tv_usec - start.tv_usec) / 1000.0;   // us to ms
    }
}

int main(int argc, char **argv)
{   

   	//thread setting
	int ret;
	pthread_t camera_thread, display_thread;
	pthread_t ssd_thread;

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

	ret = pthread_setspecific(a_key, (uint8_t *)camera_buf);
   
	ret = pthread_create(&camera_thread, NULL, camera_loop, NULL);
    if (ret < 0) {
        perror("camera pthread create error: ");
        exit(EXIT_FAILURE);
    }

	ret = pthread_create(&ssd_thread, NULL, ssd_loop, NULL);
    if (ret < 0) {
        perror("ssd_run pthread create error: ");
        exit(EXIT_FAILURE);
    }

    ret = pthread_create(&display_thread, NULL, display_loop, NULL);
    if (ret < 0) {
        perror("display pthread create error: ");
        exit(EXIT_FAILURE);
    }

	printf("%s main pthread start \n",__func__);    
    int status;
    ret = pthread_join(camera_thread, NULL);
	ret = pthread_join(display_thread, NULL);
	ret = pthread_join(ssd_thread, NULL);
  
    return 0;

    
}

