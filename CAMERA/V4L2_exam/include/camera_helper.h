#pragma once
#ifndef __CAMERA_HELPER_H__
#define __CAMERA_HELPER_H__

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <libv4l2.h>
#include "utils.h"

#define CAMERA_ERR(x,arg...)	\
	ERR_MSG("[Camera Error] " x,##arg)

#define CAMERA_DBG(x,arg...)	\
	DBG_MSG("[Camera Debug] " x,##arg)

/**
* @brief V4L2 mmap camera buffer
*/
typedef struct {
	void *offset;
	size_t length;
} __attribute__((packed)) img_buffer;


/**
* @brief 	Initialize camera device
*
* @param dev_name		Device name to initialize
* @param buf_len		V4L2 Buffer length
* @param width			Camera frame width to initialize
* @param height		Camera frame height to initialize
* @param format		Camera frame V4L2 format to initialize
*
* @return	When initialize success return initialized camera file descriptor, Otherwise return -1 
*/
int camera_init_helper(const char *dev_name, int buf_len, int width, int height, int format);

/**
* @brief 	Deinitialize camera device
*
* @param fd		Camera device file descriptor to deinitialize
* @param buf_len	Camera device buffer length
*
* @return 	When deinitialize success return 0, Otherwise return -1
*/
int camera_deinit_helper(int fd, int buf_len);

/**
* @brief 	Get camera frame	(DQBUF and QBUF)
*
* @param fd		Camera file descriptor to grab frame
* @param out	Buffer to store captured frame
* @param size	Size of captured frame
*
* @return 	When grab frame success return 0, Otherwise return -1
*/
int camera_get_frame_helper(int fd, uint8_t **out, ssize_t *size);


///////////////////	Internal Functions	/////////////////////
static int xioctl(int fd, int request, void *arg);
int camera_format_helper(int fd, int width, int height, int format);
int camera_mmap_helper(int fd, int req_cnt);
int camera_streamon(int fd, int buf_len);
int camera_streamoff(int fd);
#endif
