#ifndef _DRM_MODULE_H
#define _DRM_MODULE_H

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#include "utils.h"

#define DRM_ERR(x,arg...)							\
	ERR_MSG("[DRM Error]" x,##arg)

#define DRM_DBG(x,arg...)							\
	DBG_MSG("[DRM Debug]" x,##arg)

/**
* @brief Struct about drm devices mode information
*/
typedef struct drm_dev {
	struct drm_dev *next;

	uint32_t width;
	uint32_t height;
	uint32_t stride;
	uint32_t size;
	uint32_t handle;
	uint8_t *map;

	uint32_t fb;
	uint32_t conn;		// Connector ID
	uint32_t crtc;

	drmModeModeInfo mode;
	drmModeCrtc *saved_crtc;
} drm_dev_t;

/**
* @brief Initialize DRM Device
*
* @param device	DRM Device name to initialize
* @param modeset_list	linked list to store devices display mode lists
*
* @return When success return DRM devices file descriptor, Otherwise return errno
*/
int drm_init(const char *device, drm_dev_t **modeset_list);

/**
* @brief Deinitialize & Cleanup DRM device
*
* @param fd		DRM Device file descriptor to deinitialize
* @param modeset_list	modeset linked list dependent on fd
*
* @return When success return 0, Otherwise return negative value
*/
int drm_cleanup(int fd, drm_dev_t *modeset_list);

/**
* @brief Display image data into modeset_list
*
* @param modeset_list DRM Struct to display image
* @param buf	Image pointer
* @param size	Image size
*
* @return When success return 0, Otherwise return -1 
*/
int drm_draw(drm_dev_t *modeset_list, uint8_t *buf, uint32_t size);

///////////////// Internal Function ////////////////////
int drm_create_fb(int fd, drm_dev_t *dev);
int drm_find_crtc(int fd, drmModeRes *res, drmModeConnector *conn, drm_dev_t *dev, drm_dev_t* modeset_list);
int drm_setup_dev(int fd, drmModeRes *res, drmModeConnector *conn, drm_dev_t *dev, drm_dev_t *modeset_list);
int drm_prepare(int fd, drm_dev_t **modeset_list);
int drm_open(int *out_fd, const char *device);
#endif
