/***************************************************************************
 *   Copyright (C) 2012 by Tobias Müller                                   *
 *   Tobias_Mueller@twam.info                                              *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

/**
	Convert from YUV420 format to YUV444.

	\param width width of image
	\param height height of image
	\param src source
	\param dst destination
*/
#include "rga/RgaApi.h"
#include <stdlib.h>

void YUV420toYUV444(int width, int height, unsigned char *src, unsigned char *dst) {
    int line, column;
    unsigned char *py, *pu, *pv;
    unsigned char *tmp = dst;

    // In this format each four bytes is two pixels. Each four bytes is two Y's, a Cb and a Cr.
    // Each Y goes to one of the pixels, and the Cb and Cr belong to both pixels.
    unsigned char *base_py = src;
    unsigned char *base_pu = src + (height * width);
    unsigned char *base_pv = src + (height * width) + (height * width) / 4;

    for (line = 0; line < height; ++line) {
        for (column = 0; column < width; ++column) {
            py = base_py + (line * width) + column;
            pu = base_pu + (line / 2 * width / 2) + column / 2;
            pv = base_pv + (line / 2 * width / 2) + column / 2;

            *tmp++ = *py;
            *tmp++ = *pu;
            *tmp++ = *pv;
        }
    }
}

int YUV420toRGB24(int width, int height, unsigned char *src, unsigned char *dst) {
    if (width < 1 || height < 1 || src == NULL || dst == NULL)
        return -1;
    const long len = width * height;
    unsigned char *yData = src;
    unsigned char *vData = &yData[len];
    unsigned char *uData = &vData[len >> 2];

    int bgr[3];
    int yIdx, uIdx, vIdx, idx;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            yIdx = i * width + j;
            vIdx = (i / 2) * (width / 2) + (j / 2);
            uIdx = vIdx;
            /*  YUV420 转 BGR24
            bgr[0] = (int)(yData[yIdx] + 1.732446 * (uData[vIdx] - 128)); // b分量
            bgr[1] = (int)(yData[yIdx] - 0.698001 * (uData[uIdx] - 128) - 0.703125 * (vData[vIdx] - 128));// g分量
            bgr[2] = (int)(yData[yIdx] + 1.370705 * (vData[uIdx] - 128)); // r分量
 */
            /*  YUV420 转 RGB24 注意如转换格式不对应会导致颜色失真*/
            bgr[0] = (int)(yData[yIdx] + 1.370705 * (vData[uIdx] - 128));                                  // r分量
            bgr[1] = (int)(yData[yIdx] - 0.698001 * (uData[uIdx] - 128) - 0.703125 * (vData[vIdx] - 128)); // g分量
            bgr[2] = (int)(yData[yIdx] + 1.732446 * (uData[vIdx] - 128));                                  // b分量

            for (int k = 0; k < 3; k++) {
                idx = (i * width + j) * 3 + k;
                if (bgr[k] >= 0 && bgr[k] <= 255)
                    dst[idx] = bgr[k];
                else
                    dst[idx] = (bgr[k] < 0) ? 0 : 255;
            }
        }
    }
    return 0;
}

int YUV420toRGB24_RGA(unsigned int src_fmt, unsigned char *src_buf,
                      int src_w, int src_h,
                      unsigned int dst_fmt, int dst_fd,
                      int dst_w, int dst_h) {
    int ret;
    rga_info_t src;
    rga_info_t dst;

    //puts("src memset start");
    memset(&src, 0, sizeof(rga_info_t));
    src.fd = -1; //rga_src_fd;
    src.virAddr = src_buf;
    src.mmuFlag = 1;

    //puts("dst memset start");
    memset(&dst, 0, sizeof(rga_info_t));
    dst.fd = dst_fd;
    dst.mmuFlag = 1;

    //puts("rga_set_rect src start");
    rga_set_rect(&src.rect, 0, 0, src_w, src_h, src_w, src_h, src_fmt);
    //puts("rga_set_rect dst start");
    rga_set_rect(&dst.rect, 0, 0, dst_w, dst_h, dst_w, dst_h, dst_fmt);
    //puts("c_RkRgaBlit start");
    ret = c_RkRgaBlit(&src, &dst, NULL);
    if (ret)
        printf("c_RkRgaBlit0 error : %s\n", strerror(errno));
    return ret;
}


int YUYVtoYUV422(int width, int height, unsigned char *src, unsigned char *dst){
    //unsigned char YUY2Source[width * height * 2] = src;
    //unsigned char YV12Dest[width * height * 3/2];
    int x, y;

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < (width/2); x++)
        {
            dst[(x * 2) + y * width] = src[2 * ((x * 2) + y * width)];
            dst[(x * 2 + 1) + y * width] = src[2 * ((x * 2 + 1) + y * width)];

            dst[width * height + x + (y * width) / 2] = src[1 + 2 * ((x * 2) + y * width)];
            //fprintf(stderr, "u0 : %d ", width * height + x + (y * width) / 2);
            dst[(width * height * 3)/2 + x + (y * width) / 2] = src[3 + 2 * ((x * 2) + y * width)];
            //fprintf(stderr, "v0 : %d \n", (width * height * 3)/2 + x + (y * width) / 2);
        }
    
    }

    return 0;
}

int YUYVtoYUV422_split(int width, int height, unsigned char *src, unsigned char *dst1, unsigned char *dst2){
    //unsigned char YUY2Source[width * height * 2] = src;
    //unsigned char YV12Dest[width * height * 3/2];
    int x, y;

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < (width/4); x++)
        {
            dst1[(x * 2) + y * (width/2)] = src[2 * ((x * 2) + y * width)];
            dst1[(x * 2 + 1) + y * (width/2)] = src[2 * ((x * 2 + 1) + y * width)];

            dst1[(width/2) * height + x + y * (width/2) / 2] = src[1 + 2 * ((x * 2) + y * width)];
            
            dst1[((width/2) * height * 3)/2 + x + y * (width/2) / 2] = src[3 + 2 * ((x * 2) + y * width)];

            dst2[(x * 2) + y * (width/2)] = src[2 * ((x * 2) + (width/2) + y * width)];
            dst2[(x * 2 + 1) + y *(width/2)] = src[2 * ((x * 2 + 1) + (width/2) + y * width)];

            dst2[(width/2) * height + x + y * (width/2) / 2] = src[1 + 2 * ((x * 2) + (width/2) + y * width)];
            
            dst2[((width/2) * height * 3)/2 + x + y * (width/2) / 2] = src[3 + 2 * ((x * 2) + (width/2) +  y * width)];
        }
    
    }

    return 0;
}


int YUYVtoYUV420(int width, int height, unsigned char *src, unsigned char *dst){
    //unsigned char YUY2Source[width * height * 2] = src;
    //unsigned char YV12Dest[width * height * 3/2];

    int x, y;

    for (y = 0; y < (height/2); y++)
    {
        for (x = 0; x < (width/2); x++)
        {
            
            dst[(x * 2) + (y * 2) * width] = src[2 * ((x * 2) + (y * 2) * width)];
            dst[(x * 2 + 1) + (y * 2) * width] = src[2 * ((x * 2 + 1) + (y * 2) * width)];
            dst[(x * 2) + (y * 2 + 1) * width] = src[2 * ((x * 2) + (y * 2 + 1) * width)];
            dst[(x * 2 + 1) + (y * 2 + 1) * width] = src[2 * ((x * 2 + 1) + (y * 2 + 1) * width)];

            dst[width * height + x + (y * width) / 2] = (src[1 + 2 * ((x * 2) + (y * 2) * width)] + src[1 + 2 * ((x * 2) + (y * 2 + 1) * width)]) / 2;
            
            dst[(width * height * 5)/4 + x + (y * width) / 2] = (src[3 + 2 * ((x * 2) + (y + 2) * width)] + src[3 + 2 * ((x * 2) + (y * 2 + 1) * width)]) / 2;
            
        }
    
    }
    return 0;
}


int YUYVtoRGB24(int width, int height, unsigned char *src, unsigned char *dst) {
    if (width < 1 || height < 1 || src == NULL || dst == NULL)
        return -1;
    //const long len = width * height;

    int rgb[6];
    int idx = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width/2; j++) {
            // YUYV first pixel
            rgb[0] = (int)((1.164 * (src[idx]-16)) + 1.596 * (src[idx+3] - 128));                                  // r分量
            rgb[1] = (int)((1.164 * (src[idx]-16)) - 0.391 * (src[idx+1] - 128) - 0.813 * (src[idx+3] - 128)); // g分量
            rgb[2] = (int)((1.164 * (src[idx]-16)) + 2.018 * (src[idx+1] - 128));                                  // b分量
 
            // YUYV second pixel
            rgb[3] = (int)((1.164 * (src[idx+2]-16)) + 1.596 * (src[idx+3] - 128));                                  // r分量
            rgb[4] = (int)((1.164 * (src[idx+2]-16)) - 0.391 * (src[idx+1] - 128) - 0.813 * (src[idx+3] - 128)); // g分量
            rgb[5] = (int)((1.164 * (src[idx+2]-16)) + 2.018 * (src[idx+1] - 128));                                  // b分量

            for (int k = 0; k < 6; k++) {
                idx = (i * width + j * 2) * 2;
                if (rgb[k] >= 0 && rgb[k] <= 255)
                    dst[i * width + j * 2 + k] = rgb[k];
                else
                    dst[i * width + j * 2 + k] = (rgb[k] < 0) ? 0 : 255;
            }
        }
    }
    return 0;
}