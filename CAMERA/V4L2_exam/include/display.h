#ifndef __DISPLAY_H__
#define __DISPLAY_H__


#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <pthread.h>

#define DISPLAY_WIDTH 1920
#define DISPLAY_HEIGHT 1080

enum {
	DEPTH = 24,
	BPP = 32,
};

struct drm_dev_t {
	uint32_t *buf;
	uint32_t conn_id, enc_id, crtc_id, fb_id;
	uint32_t width, height;
	uint32_t pitch, size, handle;
	drmModeModeInfo mode;
	drmModeCrtc *saved_crtc;
	struct drm_dev_t *next;
};

extern struct drm_dev_t *dev;
extern struct drm_dev_t *dev_head;

extern int drm_fd;

int drm_init(char *dri_path);
void drm_draw(uint8_t* ptr, uint32_t* dev_buf, int width, int height, int x, int y);
void drm_boxdraw(uint32_t color, uint32_t* dev_buf, int top, int bot, int left, int right);
void drm_rectdraw(uint32_t color, uint32_t* dev_buf, int top, int bot, int left, int right);
void drm_flush(uint32_t* src_buf);
void drm_destroy(int fd, struct drm_dev_t *dev_head);
int drm_run(struct buffer *vbuffer, int *index, struct ssd_group **boxes, void *flag, pthread_mutex_t *mutex_lock, pthread_cond_t *cond_var, pthread_cond_t *cond_v4l2);


#endif