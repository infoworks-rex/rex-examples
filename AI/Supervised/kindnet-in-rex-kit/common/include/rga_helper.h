#ifndef _RGA_HELPER_H
#define _RGA_HELPER_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <rga/RgaApi.h>

#include "utils.h"

typedef struct {
	uint8_t *data;
	int width;
	int height;
	int format;
	int direction;
} rga_transform_t;

int rga_init_helper(void);
int rga_transform(rga_transform_t *source, rga_transform_t *dest);

#endif
