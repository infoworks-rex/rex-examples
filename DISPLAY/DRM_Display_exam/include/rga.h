#ifndef RGA_H
#define RGA_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

#include <rga/RgaApi.h>

int InitRga(int32_t width, int32_t height, int32_t bpp, bo_t *rga_buf_bo, int *rga_buf_fd);
int DeinitRga(bo_t *rga_buf_bo, int rga_buf_fd);
int ConvertRga(uint32_t src_fmt, uint8_t* src_buf, int32_t src_w, int32_t src_h,
					uint32_t dst_fmt, int32_t dst_fd, int32_t dst_w, int32_t dst_h, uint16_t rotate);

#endif
