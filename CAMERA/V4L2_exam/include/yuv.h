#ifndef _YUV_H_
#define _YUV_H_

#ifdef __cplusplus
extern "C"{
#endif
void YUV420toYUV444(int width, int height, unsigned char* src, unsigned char* dst);
int YUV420toRGB24(int width, int height, unsigned char* src, unsigned char* dst);

int YUV420toRGB24_RGA(unsigned int src_fmt, unsigned char* src_buf,
                      int src_w, int src_h,
                      unsigned int dst_fmt, int dst_fd,
                      int dst_w, int dst_h);

int YUYVtoYUV422(int width, int height, unsigned char *src, unsigned char *dst);

int YUYVtoYUV420(int width, int height, unsigned char *src, unsigned char *dst);
int YUYVtoYUV422_split(int width, int height, unsigned char *src, unsigned char *dst1, unsigned char *dst2);

int YUYVtoRGB24(int width, int height, unsigned char *src, unsigned char *dst);

#ifdef __cplusplus
}
#endif

#endif
