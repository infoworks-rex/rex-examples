#ifndef __CAPTURE_H__
#define __CAPTURE_H__

#define CLEAR(x) memset(&(x), 0, sizeof(x))

#define SINGLE_WIDTH 640
#define SINGLE_HEIGHT 480
#define STEREO_WIDTH 1280
#define STEREO_HEIGHT 480

enum io_method {
    IO_METHOD_READ,
    IO_METHOD_MMAP,
    IO_METHOD_USERPTR,
};

struct buffer {
    void   *start;
    size_t  length;
};
//extern struct buffer *buffers;

struct v4l2_camera{
    char            *dev_name;
    enum io_method   io;
    int              fd;
    struct buffer   *buffers;
    unsigned int     n_buffers;
    unsigned int     width;
    unsigned int     height;
    unsigned int     format;
    int              cam_no;
};

typedef void (*callback_for_v4l2)(void *in_data, int length, int c_n);
typedef int (*camera_run_t)(struct v4l2_camera *cams, int num_of_cam, int *flag);

void register_callback_for_v4l2(callback_for_v4l2 cb);

int open_device(struct v4l2_camera *cam);
void init_device(struct v4l2_camera *cam);
void start_capturing(struct v4l2_camera *cam);
void mainloop(struct v4l2_camera *cam, int *run_flag);
void mainloop_nowhile(struct v4l2_camera *cam);
void stop_capturing(struct v4l2_camera *cam);
void uninit_device(struct v4l2_camera *cam);
void close_device(struct v4l2_camera *cam);


void rawframe_save(uint8_t *p, int size);

int cameraRun(struct v4l2_camera *cams, int num_of_cam, int *flag);

#endif