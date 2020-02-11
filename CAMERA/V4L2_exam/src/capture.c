/*
 *  V4L2 video capture example
 *
 *  This program can be used and distributed without restrictions.
 *
 *      This program is provided with the V4L2 API
 * see https://linuxtv.org/docs.php for more information
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#include <getopt.h>             /* getopt_long() */

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <pthread.h>

#include <linux/videodev2.h>

#include "capture.h"

//#define DEBUG

static int              force_format = 1;
callback_for_v4l2       cb_for_v4l2;


void register_callback_for_v4l2(callback_for_v4l2 cb) {
    cb_for_v4l2 = cb;
}


void errno_exit(const char *s)
{
        fprintf(stderr, "%s error %d, %s\\n", s, errno, strerror(errno));
        exit(EXIT_FAILURE);
}

static int xioctl(int fh, int request, void *arg)
{
        int r;

        do {
                r = ioctl(fh, request, arg);
        } while (-1 == r && EINTR == errno);

        return r;
}


int open_device(struct v4l2_camera *cam)
{
    int fd;
    struct stat st;
    //dev_name = dev_str;

    if (-1 == stat(cam->dev_name, &st)) {
        fprintf(stderr, "Cannot identify '%s': %d, %s\\n",
            cam->dev_name, errno, strerror(errno));
        return -1;
        //exit(EXIT_FAILURE);
    }

    if (!S_ISCHR(st.st_mode)) {
        fprintf(stderr, "%s is no device\\n", cam->dev_name);
        exit(EXIT_FAILURE);
    }

#ifdef DEBUG
    printf("open device : %s\n", cam->dev_name);
#endif
    fd = open(cam->dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

    if (-1 == fd) {
        fprintf(stderr, "Cannot open '%s': %d, %s\\n",
            cam->dev_name, errno, strerror(errno));
        return -1;
        //exit(EXIT_FAILURE);
    }
    cam->fd = fd;
    return fd;
}

/*
static void init_read(unsigned int buffer_size)
{
        buffers = calloc(1, sizeof(*buffers));

        if (!buffers) {
                fprintf(stderr, "Out of memory\\n");
                exit(EXIT_FAILURE);
        }

        buffers[0].length = buffer_size;
        buffers[0].start = malloc(buffer_size);

        if (!buffers[0].start) {
                fprintf(stderr, "Out of memory\\n");
                exit(EXIT_FAILURE);
        }
}
*/

static void init_mmap(struct v4l2_camera *cam)
{
    struct v4l2_requestbuffers req;

    CLEAR(req);

    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

#ifdef DEBUG
    printf("request buffers : %s\n", cam->dev_name);
#endif
    if (-1 == xioctl(cam->fd, VIDIOC_REQBUFS, &req)) {
        if (EINVAL == errno) {
            fprintf(stderr, "%s does not support memory mappingn", cam->dev_name);
            exit(EXIT_FAILURE);
        } else {
            errno_exit("VIDIOC_REQBUFS");
        }
    }
        
    if (req.count < 2) {
        fprintf(stderr, "Insufficient buffer memory on %s\\n", cam->dev_name);
        exit(EXIT_FAILURE);
    }
        
    cam->buffers = calloc(req.count, sizeof(*cam->buffers));

    if (!cam->buffers) {
        fprintf(stderr, "Out of memory\\n");
        exit(EXIT_FAILURE);
    }

#ifdef DEBUG
    printf("query buffers : %s\n", cam->dev_name);
#endif
    for (cam->n_buffers = 0; cam->n_buffers < req.count; ++cam->n_buffers) {
        struct v4l2_buffer buf;

        CLEAR(buf);

        buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory      = V4L2_MEMORY_MMAP;
        buf.index       = cam->n_buffers;

        if (-1 == xioctl(cam->fd, VIDIOC_QUERYBUF, &buf)){
            errno_exit("VIDIOC_QUERYBUF");
        }

        cam->buffers[cam->n_buffers].length = buf.length;
        fprintf(stderr, "buffer index : %d \n buffer length : %ld\n", 
                cam->n_buffers, cam->buffers[cam->n_buffers].length);
        cam->buffers[cam->n_buffers].start = 
            mmap(NULL /* start anywhere */,
                    buf.length,
                    PROT_READ | PROT_WRITE /* required */,
                    MAP_SHARED /* recommended */,
                    cam->fd, 
                    buf.m.offset);

        if (MAP_FAILED == cam->buffers[cam->n_buffers].start){
            errno_exit("mmap");
        }
    }
}

/*
static void init_userp(unsigned int buffer_size, int fd)
{
        struct v4l2_requestbuffers req;

        CLEAR(req);

        req.count  = 4;
        req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_USERPTR;

        if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
                if (EINVAL == errno) {
                        fprintf(stderr, "%s does not support "
                                 "user pointer i/on", dev_name);
                        exit(EXIT_FAILURE);
                } else {
                        errno_exit("VIDIOC_REQBUFS");
                }
        }

        buffers = calloc(4, sizeof(*buffers));

        if (!buffers) {
                fprintf(stderr, "Out of memory\\n");
                exit(EXIT_FAILURE);
        }

        for (n_buffers = 0; n_buffers < 4; ++n_buffers) {
                buffers[n_buffers].length = buffer_size;
                buffers[n_buffers].start = malloc(buffer_size);

                if (!buffers[n_buffers].start) {
                        fprintf(stderr, "Out of memory\\n");
                        exit(EXIT_FAILURE);
                }
        }
}
*/

void init_device(struct v4l2_camera *cam)
{
    struct v4l2_capability cap;
    struct v4l2_cropcap cropcap;
    struct v4l2_crop crop;
    struct v4l2_format fmt;
    unsigned int min;

#ifdef DEBUG
    printf("query cap : %s\n", cam->dev_name);
#endif
    if (-1 == xioctl(cam->fd, VIDIOC_QUERYCAP, &cap)) {
        if (EINVAL == errno) {
            fprintf(stderr, "%s is no V4L2 device\\n", cam->dev_name);
            exit(EXIT_FAILURE);
        } else {
            errno_exit("VIDIOC_QUERYCAP");
        }
    }

#ifdef DEBUG
    printf("capture capabilities : %s\n", cam->dev_name);
#endif
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "%s is no video capture device\\n", cam->dev_name);
        exit(EXIT_FAILURE);
    }

    switch (cam->io) {
        case IO_METHOD_READ:
            if (!(cap.capabilities & V4L2_CAP_READWRITE)) {
                fprintf(stderr, "%s does not support read i/o\\n", cam->dev_name);
                exit(EXIT_FAILURE);
            }
            break;

        case IO_METHOD_MMAP:
        case IO_METHOD_USERPTR:
            if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
                fprintf(stderr, "%s does not support streaming i/o\\n", cam->dev_name);
                exit(EXIT_FAILURE);
            }
            break;
    }


    /* Select video input, video standard and tune here. */
    CLEAR(cropcap);

    cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (0 == xioctl(cam->fd, VIDIOC_CROPCAP, &cropcap)) {
        crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        crop.c = cropcap.defrect; /* reset to default */

        if (-1 == xioctl(cam->fd, VIDIOC_S_CROP, &crop)) {
            switch (errno) {
                case EINVAL:
                    /* Cropping not supported. */
                    break;
                default:
                    /* Errors ignored. */
                    break;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
            }
        }
    } else {
                /* Errors ignored. */
    }


    CLEAR(fmt);

#ifdef DEBUG
    printf("set formats : %s\n", cam->dev_name);
#endif
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (force_format) {
        fmt.fmt.pix.width       = cam->width;
        fmt.fmt.pix.height      = cam->height;
        fmt.fmt.pix.pixelformat = cam->format;
        fmt.fmt.pix.field       = V4L2_FIELD_NONE;
        //fmt.fmt.pix.field       = V4L2_FIELD_INTERLACED;

        if (-1 == xioctl(cam->fd, VIDIOC_S_FMT, &fmt))
            errno_exit("VIDIOC_S_FMT");

    /* Note VIDIOC_S_FMT may change width and height. */
    } else {
        /* Preserve original settings as set by v4l2-ctl for example */
        if (-1 == xioctl(cam->fd, VIDIOC_G_FMT, &fmt))
            errno_exit("VIDIOC_G_FMT");
    }

    /* Buggy driver paranoia. */
        
    min = fmt.fmt.pix.width * 2;
    if (fmt.fmt.pix.bytesperline < min)
        fmt.fmt.pix.bytesperline = min;
    
    min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
    if (fmt.fmt.pix.sizeimage < min)
        fmt.fmt.pix.sizeimage = min;
        
    switch (cam->io) {
        case IO_METHOD_READ:
            //init_read(fmt.fmt.pix.sizeimage);
            break;

        case IO_METHOD_MMAP:
#ifdef DEBUG
            printf("init mmap : %s\n", cam->dev_name);
#endif
            init_mmap(cam);
            break;

        case IO_METHOD_USERPTR:
            //init_userp(fmt.fmt.pix.sizeimage, fd);
            break;
    }
    
}

void start_capturing(struct v4l2_camera *cam)
{
    unsigned int i;
    enum v4l2_buf_type type;

    switch (cam->io) {
    case IO_METHOD_READ:
        /* Nothing to do. */
        break;

    case IO_METHOD_MMAP:
        for (i = 0; i < cam->n_buffers; ++i) {
            struct v4l2_buffer buf;

            CLEAR(buf);
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;

#ifdef DEBUG
            printf("vidioc qbuf : %s\n", cam->dev_name);
#endif
            if (-1 == xioctl(cam->fd, VIDIOC_QBUF, &buf))
                errno_exit("VIDIOC_QBUF");
        }

        type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

#ifdef DEBUG
        printf("stream on : %s\n", cam->dev_name);
#endif
        if (-1 == xioctl(cam->fd, VIDIOC_STREAMON, &type))
            errno_exit("VIDIOC_STREAMON");

        break;

    case IO_METHOD_USERPTR:
        for (i = 0; i < cam->n_buffers; ++i) {
            struct v4l2_buffer buf;

            CLEAR(buf);
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_USERPTR;
            buf.index = i;
            buf.m.userptr = (unsigned long)cam->buffers[i].start;
            buf.length = cam->buffers[i].length;

            if (-1 == xioctl(cam->fd, VIDIOC_QBUF, &buf))
                errno_exit("VIDIOC_QBUF");
        }

        type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        if (-1 == xioctl(cam->fd, VIDIOC_STREAMON, &type))
            errno_exit("VIDIOC_STREAMON");

        break;
    }
    
}

void rawframe_save(uint8_t *p, int size){
    static int frame_number = 0;
    char filename[32];
    sprintf(filename, "/userdata/frame-%d.raw", frame_number);
    int fp= open(filename, O_WRONLY | O_CREAT, 0660);
        
    write(fp, p, size);

    close(fp);

    // if frame number conunt needed
    frame_number++;
    printf("frame saved\n");
}

static void process_image(uint8_t *p, int length, int camera_number)
{  
    //rawframe_save(p, size);

    if(cb_for_v4l2){
#ifdef DEBUG
        printf("callback v4l2\n");
#endif
        cb_for_v4l2(p, length, camera_number);
    }

}

int read_frame(struct v4l2_camera *cam)
{
    struct v4l2_buffer buf;
    unsigned int i;

#ifdef DEBUG
        //printf("read frame\n");
#endif
    switch (cam->io) {
    case IO_METHOD_READ:
        if (-1 == read(cam->fd, cam->buffers[0].start, cam->buffers[0].length)) {
            switch (errno) {
            case EAGAIN:
                return 0;

            case EIO:
                /* Could ignore EIO, see spec. */

                /* fall through */

            default:
                errno_exit("read");
            }
        }

        //process_image(buffers[0].start, buffers[0].length);
        break;

    case IO_METHOD_MMAP:
        CLEAR(buf);

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (-1 == xioctl(cam->fd, VIDIOC_DQBUF, &buf)) {
            switch (errno) {
            case EAGAIN:
                return 0;

            case EIO:
                /* Could ignore EIO, see spec. */

                /* fall through */

            default:
                errno_exit("VIDIOC_DQBUF");
            }
        }

        assert(buf.index < cam->n_buffers);
#ifdef DEBUG
        printf("process image\n");
#endif
        process_image(cam->buffers[buf.index].start, buf.bytesused, cam->cam_no);

        if (-1 == xioctl(cam->fd, VIDIOC_QBUF, &buf))
            errno_exit("VIDIOC_QBUF");
    
        break;

    case IO_METHOD_USERPTR:
        CLEAR(buf);

        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_USERPTR;

        if (-1 == xioctl(cam->fd, VIDIOC_DQBUF, &buf)) {
            switch (errno) {
            case EAGAIN:
                return 0;
            case EIO:
                /* Could ignore EIO, see spec. */

                /* fall through */

            default:
                errno_exit("VIDIOC_DQBUF");
            }
        }

        for (i = 0; i < cam->n_buffers; ++i)
            if (buf.m.userptr == (unsigned long)cam->buffers[i].start
                && buf.length == cam->buffers[i].length)
                break;

        assert(i < cam->n_buffers);

        //process_image((void *)buf.m.userptr, buf.bytesused);

        if (-1 == xioctl(cam->fd, VIDIOC_QBUF, &buf))
            errno_exit("VIDIOC_QBUF");
        break;
    }

    return 1;
}


void mainloop(struct v4l2_camera *cam, int *run_flag)
{

    while (*run_flag) {
        for (;;) {
            fd_set fds;
            struct timeval tv;
            int r;

            FD_ZERO(&fds);
            FD_SET(cam->fd, &fds);

            /* Timeout. */
            tv.tv_sec = 2;
            tv.tv_usec = 0;

            r = select(cam->fd + 1, &fds, NULL, NULL, &tv);

            if (-1 == r) {
                if (EINTR == errno)
                    continue;
                errno_exit("select");
            }

            if (0 == r) {
                fprintf(stderr, "select timeout\n");
                exit(EXIT_FAILURE);
            }

            if (read_frame(cam)){
                                
                               
                break; 
            }
        }
    }
}

void mainloop_nowhile(struct v4l2_camera *cam)
{
#ifdef DEBUG
    printf("loop begin : %s\n", cam->dev_name);
#endif
    for(;;){
        fd_set fds;
        struct timeval tv;
        int r;

        FD_ZERO(&fds);
        FD_SET(cam->fd, &fds);
#ifdef DEBUG
        //printf("fds : %dn", fds);
#endif
        /* Timeout. */
        tv.tv_sec = 2;
        tv.tv_usec = 0;

        r = select(cam->fd + 1, &fds, NULL, NULL, &tv);

        if (-1 == r) {
            if (EINTR == errno)
                continue;
            errno_exit("select");
        }

        if (0 == r) {
            fprintf(stderr, "select timeout\n");
            exit(EXIT_FAILURE); 
        }
#ifdef DEBUG
        //printf("r : %d\n", r);
#endif
        if (read_frame(cam)){
                                
                               
            break;
        }
        /* EAGAIN - continue select loop. */
    }
#ifdef DEBUG
    printf("loop end : %s\n", cam->dev_name);
#endif
}

void stop_capturing(struct v4l2_camera *cam)
{
    enum v4l2_buf_type type;

    switch (cam->io) {
    case IO_METHOD_READ:
        /* Nothing to do. */
        break;

    case IO_METHOD_MMAP:
    case IO_METHOD_USERPTR:
        type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (-1 == xioctl(cam->fd, VIDIOC_STREAMOFF, &type))
            errno_exit("VIDIOC_STREAMOFF");
        break;
    }
}

void uninit_device(struct v4l2_camera *cam)
{
        unsigned int i;

        switch (cam->io) {
        case IO_METHOD_READ:
                free(cam->buffers[0].start);
                break;

        case IO_METHOD_MMAP:
                for (i = 0; i < cam->n_buffers; ++i)
                        if (-1 == munmap(cam->buffers[i].start, cam->buffers[i].length))
                                errno_exit("munmap");
                break;

        case IO_METHOD_USERPTR:
                for (i = 0; i < cam->n_buffers; ++i)
                        free(cam->buffers[i].start);
                break;
        }
        free(cam->buffers);
}


void close_device(struct v4l2_camera *cam)
{
        if (-1 == close(cam->fd))
                errno_exit("close");

        cam->fd = -1;
}

int cameraRun(struct v4l2_camera *cams, int num_of_cam, int *flag)
{
    int i;

    for(i = 0; i < num_of_cam; i++){
        // open and initialize device
	    cams[i].fd = open_device(&cams[i]);
        if(cams[i].fd == -1){
            num_of_cam -=1;
            break;
        }
#ifdef DEBUG
        printf("file descripter : %d\n", cams[i].fd);
#endif
        init_device(&cams[i]);
        cams[i].cam_no = i;          
	    // start capturing
	    start_capturing(&cams[i]);
    }

    while(*flag){
        for(i = 0; i < num_of_cam; i++){
            mainloop_nowhile(&cams[i]);
        }
    }
    for (i = 0; i < num_of_cam; i++){
	    // stop capturing
	    stop_capturing(&cams[i]);

	    // close device
	    uninit_device(&cams[i]);
	    close_device(&cams[i]);
    }
	return 0;
}
