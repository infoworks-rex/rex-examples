#ifndef _CONFIGS_H
#define _CONFIGS_H

#include <rga/RgaApi.h>
#include <rga/rga.h>
#define DISP_W		1920
#define DISP_H		1080
#define DISP_SIZE	DISP_W * DISP_H
#define DISP_BPP	4
#define DISP_FMT	RK_FORMAT_BGRA_8888

#define CAM_CH		3
#define CAM_W		640
#define CAM_H		480
#define CAM_SIZE	CAM_W * CAM_H
#define CAM_FMT 	V4L2_PIX_FMT_YUV420
#define CAM_BPP	2

#define IMG_SIZE CAM_W * CAM_H * CAM_BPP

#define VENDOR 0x1d6b
#define PRODUCT 0x0105

#define CARD_DEV "/dev/dri/card0"
#define CAM_DEV "/dev/video6"

#define V4L2_BUF_LEN 3

#endif
