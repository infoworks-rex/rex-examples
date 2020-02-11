/* please refer better example: https://github.com/dvdhrm/docs/tree/master/drm-howto/ */
#define _XOPEN_SOURCE 600

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <pthread.h>
#include <math.h>

#include "buffer.h"
#include "rga/RgaApi.h"
#include "display.h"
#include "yuv.h"
#include "capture.h"

#define SRC_RKRGA_FMT RK_FORMAT_YCbCr_422_P
#define DST_RKRGA_FMT RK_FORMAT_RGB_888

struct drm_dev_t *dev;
struct drm_dev_t *dev_head;

bo_t g_rga_buf_bo_drm;
int g_rga_buf_fd_drm;

int drm_fd;

void fatal(char *str)
{
	fprintf(stderr, "%s\n", str);
	exit(EXIT_FAILURE);
}

void error(char *str)
{
	perror(str);
	exit(EXIT_FAILURE);
}

int eopen(const char *path, int flag)
{
	int fd;

	if ((fd = open(path, flag)) < 0) {
		fprintf(stderr, "cannot open \"%s\"\n", path);
		error("open");
	}
	return fd;
}

void *emmap(int addr, size_t len, int prot, int flag, int fd, off_t offset)
{
	uint32_t *fp;

	if ((fp = (uint32_t *) mmap(0, len, prot, flag, fd, offset)) == MAP_FAILED)
		error("mmap");
	return fp;
}

int drm_open(const char *path)
{
	int fd, flags;
	uint64_t has_dumb;

	fd = eopen(path, O_RDWR);

	/* set FD_CLOEXEC flag */
	if ((flags = fcntl(fd, F_GETFD)) < 0
		|| fcntl(fd, F_SETFD, flags | FD_CLOEXEC) < 0)
		fatal("fcntl FD_CLOEXEC failed");

	/* check capability */
	if (drmGetCap(fd, DRM_CAP_DUMB_BUFFER, &has_dumb) < 0 || has_dumb == 0)
		fatal("drmGetCap DRM_CAP_DUMB_BUFFER failed or doesn't have dumb buffer");

	return fd;
}

struct drm_dev_t *drm_find_dev(int fd)
{
	int i;
	struct drm_dev_t *dev = NULL, *dev_head = NULL;
	drmModeRes *res;
	drmModeConnector *conn;
	drmModeEncoder *enc;

	if ((res = drmModeGetResources(fd)) == NULL)
		fatal("drmModeGetResources() failed");

	/* find all available connectors */
	for (i = 0; i < res->count_connectors; i++) {
		conn = drmModeGetConnector(fd, res->connectors[i]);

		if (conn != NULL && conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
			dev = (struct drm_dev_t *) malloc(sizeof(struct drm_dev_t));
			memset(dev, 0, sizeof(struct drm_dev_t));

			dev->conn_id = conn->connector_id;
			dev->enc_id = conn->encoder_id;
			dev->next = NULL;

			memcpy(&dev->mode, &conn->modes[0], sizeof(drmModeModeInfo));
			dev->width = conn->modes[0].hdisplay;
			dev->height = conn->modes[0].vdisplay;

			/* FIXME: use default encoder/crtc pair */
			if ((enc = drmModeGetEncoder(fd, dev->enc_id)) == NULL)
				fatal("drmModeGetEncoder() faild");
			dev->crtc_id = enc->crtc_id;
			drmModeFreeEncoder(enc);

			dev->saved_crtc = NULL;

			/* create dev list */
			dev->next = dev_head;
			dev_head = dev;
		}
		drmModeFreeConnector(conn);
	}
	drmModeFreeResources(res);

	return dev_head;
}

void drm_setup_fb(int fd, struct drm_dev_t *dev)
{
	struct drm_mode_create_dumb creq;
	struct drm_mode_map_dumb mreq;

	memset(&creq, 0, sizeof(struct drm_mode_create_dumb));
	creq.width = dev->width;
	creq.height = dev->height;
	creq.bpp = BPP; // hard conding

	if (drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &creq) < 0)
		fatal("drmIoctl DRM_IOCTL_MODE_CREATE_DUMB failed");

	dev->pitch = creq.pitch;
	dev->size = creq.size;
	dev->handle = creq.handle;

	if (drmModeAddFB(fd, dev->width, dev->height,
		DEPTH, BPP, dev->pitch, dev->handle, &dev->fb_id))
		fatal("drmModeAddFB failed");

	memset(&mreq, 0, sizeof(struct drm_mode_map_dumb));
	mreq.handle = dev->handle;

	if (drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &mreq))
		fatal("drmIoctl DRM_IOCTL_MODE_MAP_DUMB failed");

	dev->buf = (uint32_t *) emmap(0, dev->size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, mreq.offset);

	dev->saved_crtc = drmModeGetCrtc(fd, dev->crtc_id); /* must store crtc data */
	if (drmModeSetCrtc(fd, dev->crtc_id, dev->fb_id, 0, 0, &dev->conn_id, 1, &dev->mode))
		fatal("drmModeSetCrtc() failed");
}

int drm_init(char *dri_path){
    /* init */
	int fd = drm_open(dri_path);
    drm_fd = fd;
	dev_head = drm_find_dev(drm_fd);

    if (dev_head == NULL) {
		fprintf(stderr, "available drm_dev not found\n");
		dev = NULL;
        return EXIT_FAILURE;
	}

	printf("available connector(s)\n\n");
	for (dev = dev_head; dev != NULL; dev = dev->next) {
		printf("connector id:%d\n", dev->conn_id);
		printf("\tencoder id:%d crtc id:%d fb id:%d\n", dev->enc_id, dev->crtc_id, dev->fb_id);
		printf("\twidth:%d height:%d\n", dev->width, dev->height);
	}

	/* FIXME: use first drm_dev */
	dev = dev_head;
	drm_setup_fb(drm_fd, dev);
    return 0;
}

void drm_draw(uint8_t* ptr, uint32_t* dev_buf, int width, int height, int x, int y)
{
	int i, j;
	int dstwidth = DISPLAY_WIDTH;
	int dstheight = DISPLAY_HEIGHT;
	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++) {
			*(dev_buf + (i + y) * dstwidth + j + x) = (uint32_t)(ptr[(i * width + j)*3] << 16) + (uint32_t)(ptr[(i * width + j)*3 + 1] << 8) + (uint32_t)(ptr[(i * width + j)*3 + 2]);
			//*(dev_buf + i * dstwidth + j) =0x00FFFF;
		}
    }
	//usleep(200000);
}

void drm_boxdraw(uint32_t color, uint32_t* dev_buf, int top, int bot, int left, int right){
    int i;
    int dstwidth = DISPLAY_WIDTH;
    int dstheight = DISPLAY_HEIGHT;
    for(i = left; i <= right; i++){
        *(dev_buf + top * dstwidth + i) = color;
    }
    for(i = left; i <= right; i++){
        *(dev_buf + bot * dstwidth + i) = color;
    }
    for(i = top; i <= bot; i++){
        *(dev_buf + i * dstwidth + left) = color;
    }
    for(i = top; i <= bot; i++){
        *(dev_buf + i * dstwidth + right) = color;
    }
}

void drm_rectdraw(uint32_t color, uint32_t* dev_buf, int top, int bot, int left, int right){
    int i, j;
    int dstwidth = DISPLAY_WIDTH;
    int dstheight = DISPLAY_HEIGHT;
    for(i = top; i <= bot; i++){
        for(j = left; j <= right; j++){
            *(dev_buf + i * dstwidth + j) = color;
        }
    }
}

void drm_flush(uint32_t *src_buf){
    memcpy(dev->buf, src_buf, dev->width * dev->height * 4);
    return;
}

void drm_destroy(int fd, struct drm_dev_t *dev_head)
{
	struct drm_dev_t *devp, *devp_tmp;
	struct drm_mode_destroy_dumb dreq;

	for (devp = dev_head; devp != NULL;) {
		if (devp->saved_crtc)
			drmModeSetCrtc(fd, devp->saved_crtc->crtc_id, devp->saved_crtc->buffer_id,
				devp->saved_crtc->x, devp->saved_crtc->y, &devp->conn_id, 1, &devp->saved_crtc->mode);
		drmModeFreeCrtc(devp->saved_crtc);

		munmap(devp->buf, devp->size);

		drmModeRmFB(fd, devp->fb_id);

		memset(&dreq, 0, sizeof(dreq));
		dreq.handle = devp->handle;
		drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &dreq);

		devp_tmp = devp;
		devp = devp->next;
		free(devp_tmp);
	}

	close(fd);
}


/*
int drm_run(struct buffer *vbuffer, int *index, struct ssd_group **boxes, void *flag, pthread_mutex_t *mutex_lock, pthread_cond_t *cond_var, pthread_cond_t *cond_v4l2){
    //uint32_t *f_buffer;
    char *dri_path = "/dev/dri/card0";

    drm_init(dri_path);
    //printf("finish drm_init\n");
    buffer_init(SINGLE_WIDTH, SINGLE_HEIGHT, 24, &g_rga_buf_bo_drm,
                &g_rga_buf_fd_drm);
    
    
    //f_buffer = calloc(dev->width * dev->height, sizeof(uint32_t));
    
    //int leftmargin = (dev->width - (SINGLE_WIDTH * 2 + 60)) / 2;
    //int topmargin = (dev->height - (SINGLE_HEIGHT * 2 + 60)) / 2;
    int startx = 0, starty = 0;

    //printf("finish drm_init2\n");

    struct ssd_object *object;
    int valid_object_count;

    sleep(2);

    //pthread_cond_signal(cond_var);
    while(*(int*)flag){
        int i;
        //pthread_mutex_lock(mutex_lock);
        //for(i = 0; i < 2; i++){
        
        //if(*index == 0){
        if(i == 0){
            //startx = leftmargin;
            startx = 290;
            //starty = topmargin;
            starty = 30;
        }
        //else if(*index == 1){
        else if(i == 1){
            //startx = leftmargin + SINGLE_WIDTH * 2 + 60;
            startx = 990;
            //starty = topmargin;
            starty = 30;
        }
        else if(*index == 2){
            //startx = leftmargin;
            startx = 290;
            //starty = topmargin + SINGLE_HEIGHT * 2 + 60;
            starty = 570;
        }
        else if(*index == 3){
            //startx = leftmargin + SINGLE_WIDTH * 2 + 60;
            startx = 990;
            //starty = topmargin + SINGLE_HEIGHT * 2 + 60;
            starty = 570;
        }

        //pthread_cond_wait(cond_v4l2, mutex_lock);
        YUV420toRGB24_RGA(SRC_RKRGA_FMT, vbuffer[i].start, SINGLE_WIDTH, SINGLE_HEIGHT,
                          DST_RKRGA_FMT, g_rga_buf_fd_drm, SINGLE_WIDTH, SINGLE_HEIGHT);
        //drm_draw(g_rga_buf_bo_drm.ptr, dev->buf, SINGLE_WIDTH, SINGLE_HEIGHT, 290, 30);
        if(*index == 1){
            pthread_cond_signal(cond_var);
        }
        //pthread_mutex_unlock(mutex_lock);
        printf("frame number = %d\n", *index);
        //drm_draw(vbuffer->start, dev->buf, SINGLE_WIDTH , SINGLE_HEIGHT, startx, starty);
        //drm_draw(g_rga_buf_bo_drm.ptr, f_buffer, 640, 480, startx, starty);
        drm_draw(g_rga_buf_bo_drm.ptr, dev->buf, SINGLE_WIDTH, SINGLE_HEIGHT, startx, starty);
        
        
        if(boxes != NULL){
            valid_object_count = (*boxes)->count;
            if (valid_object_count > 10) {
                valid_object_count = 10;
            }
        }

        if(boxes != NULL){
            for (int i = 0; i < valid_object_count; i++) { 
                object = &((*boxes)->objects[i]);
                //drm_boxdraw(f_buffer, 
                drm_boxdraw(0x50bcdf, dev->buf,
                    map(object->select.top, 0, 299, 0, SINGLE_HEIGHT - 1) + starty,
                    map(object->select.bottom, 0, 299, 0, SINGLE_HEIGHT - 1) + starty,
                    map(object->select.left, 0, 299, 0, SINGLE_WIDTH - 1) + startx,
                    map(object->select.right, 0, 299, 0, SINGLE_WIDTH - 1) + startx);
            }
        }
        //}
        //pthread_mutex_unlock(mutex_lock);
        //drm_flush(f_buffer);
    }
    buffer_deinit(&g_rga_buf_bo_drm, g_rga_buf_fd_drm);
    drm_destroy(drm_fd, dev);
    return 0;
}
*/