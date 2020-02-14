#include "rga_helper.h"

int rga_init_helper(void)
{
	// Initializing RGA
	c_RkRgaInit();
}

int rga_transform(rga_transform_t *source, rga_transform_t *dest)
{
	int ret;
	rga_info_t src, dst;

	CLEAR(src); 
	src.fd = -1;
	src.mmuFlag = 1;
	src.virAddr = source->data;

	CLEAR(dst);
	dst.fd = -1;
	dst.mmuFlag = 1;
	dst.virAddr = dest->data;

	if (source->direction != 0) {
		src.rotation = source->direction;
	}

	rga_set_rect(&src.rect, 0, 0, source->height, source->width, source->height, source->width, source->format);
	rga_set_rect(&dst.rect, 0, 0, dest->height, dest->width, dest->height, dest->width, dest->format);

	ret = c_RkRgaBlit(&src, &dst, NULL);

	return 0;
}

int rga_flip(uint8_t *src, uint8_t*dst, int src_w, int src_h, int src_fmt, int dst_w, int dst_h, int dst_fmt, int direction)
{
	rga_info_t src_info, dst_info;

	memset(&src_info, 0x00, sizeof(rga_info_t));
	src_info.fd = -1;
	src_info.mmuFlag = 1;
	src_info.virAddr = src;

	memset(&dst_info, 0x00, sizeof(rga_info_t));
	dst_info.fd = -1;
	dst_info.mmuFlag = 1;
	dst_info.virAddr = dst;



	c_RkRgaBlit(&src_info, &dst_info, NULL);
}

