#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#include <getopt.h>             /* getopt_long() */

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/select.h>

#include <signal.h>

#include <linux/videodev2.h>

#include <xf86drm.h>
#include <xf86drmMode.h>

#include "capture.h"
#include "display.h"
#include "buffer.h"
#include "yuv.h"

void video_receive(void *p, int length, int camera_number);
void display_draw();
int checkstdin(int timeout);
void func_exit();

callback_for_v4l2 v4l2_callback = video_receive;
struct v4l2_camera cam;

bo_t g_rga_buf_bo;
int g_rga_buf_fd;

struct buffer raw_video;
struct buffer buf;

int main(int argc, char **argv)
{   
    signal(SIGINT, func_exit);

    //init buffer
    buf.length = 1280 * 720 * 2;
    buf.start = calloc(buf.length, sizeof(uint8_t));

    //init v4l2    
    cam.dev_name = "/dev/video6";
    cam.io = IO_METHOD_MMAP;
    cam.width = 1280;   //hd hardcoded
    cam.height = 720;   //hd hardcoded
    cam.format = V4L2_PIX_FMT_YUYV;

    register_callback_for_v4l2(v4l2_callback);
    open_device(&cam);
    init_device(&cam);
    start_capturing(&cam);    

    //init drm
    drm_init("/dev/dri/card0");

    //init rga
    buffer_init(dev->width, dev->height, 24, &g_rga_buf_bo, &g_rga_buf_fd);

    while(1){
        mainloop_nowhile(&cam);
        if(checkstdin(10000)){
            if (getchar() == 'q') {
                break;
            }
        }
    }
    func_exit();
    return 0;
}


void video_receive(void *p, int length, int camera_number){
    raw_video.start = p;
    raw_video.length = length;
    display_draw();
}

void display_draw(){
    YUYVtoYUV422(1280, 720, raw_video.start, buf.start);
    YUV420toRGB24_RGA(RK_FORMAT_YCbCr_422_P, buf.start, 1280, 720,
			RK_FORMAT_RGB_888, g_rga_buf_fd, 1280, 720);
    drm_draw(g_rga_buf_bo.ptr, dev->buf, 1280, 720, 0, 0);
}

int checkstdin(int timeout){
    fd_set rfds;
    struct timeval tv;
    FD_ZERO(&rfds);
    FD_SET(0, &rfds);

    tv.tv_sec = timeout / 1000000;
    tv.tv_usec = timeout % 1000000;

    return select(1, &rfds, NULL, NULL, &tv) > 0;
}

void func_exit(){
    stop_capturing(&cam);
    uninit_device(&cam);
    close_device(&cam);
    free(buf.start);
    buffer_deinit(&g_rga_buf_bo, g_rga_buf_fd);
    drm_destroy(drm_fd, dev);
}