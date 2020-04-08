#include "camera_helper.h"

img_buffer *buffers = NULL;

static int xioctl(int fd, int request, void *arg)
{
	int r;

	do r = v4l2_ioctl (fd, request, arg);
	while (-1 == r && EINTR == errno);

	return r;
}

//DUMMY_FUNC(int, camera_init_helper, NULL, const char *dev_name)
int camera_init_helper(const char *dev_name, int buf_len, int width, int height, int format)
{
	int fd, ret;
		
	struct v4l2_capability cap;
	struct v4l2_cropcap cropcap;
	struct v4l2_crop crop;
	struct v4l2_streamparm frameint;

	fd = v4l2_open(dev_name, O_RDWR | O_NONBLOCK, 0);
	if (-1 == fd) {
		CAMERA_ERR("Cannot open %s\n", dev_name);
		return -1;
	}

	ret = xioctl(fd, VIDIOC_QUERYCAP, &cap);
	if (ret == -1) {
		if (errno == EINVAL) {
			CAMERA_ERR("%s is no V4L2 Deivce\n", dev_name);
			return -1;
		}
		else {
			CAMERA_ERR("VIDIOC_QUERYCAP Error : %s\n", strerror(errno));
			return -1;
		}
	}

	if ( !(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) ) {
		CAMERA_ERR("%s is cannot support capture\n", dev_name);
		return -1;
	}

	if ( !(cap.capabilities & V4L2_CAP_STREAMING) ) {
		CAMERA_ERR("%s is cannot support streaming\n", dev_name);
		return -1;
	}

   CLEAR(cropcap);

   cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

   if (0 == xioctl(fd, VIDIOC_CROPCAP, &cropcap)) {
      crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      crop.c = cropcap.defrect; /* reset to default */

      if (-1 == xioctl(fd, VIDIOC_S_CROP, &crop)) {
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

	ret = camera_format_helper(fd, width, height, format);
	if (ret < 0) {
		CAMERA_ERR("Can't set device format\n");
		return -1;
	}
#if 0
    frameint.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    frameint.parm.capture.timeperframe.numerator = 1;
    frameint.parm.capture.timeperframe.denominator = 9;
    if (-1 == xioctl(fd, VIDIOC_S_PARM, &frameint))
      fprintf(stderr,"Unable to set frame interval.\n");
#endif

	ret = camera_mmap_helper(fd, buf_len);
	if (ret < 0) {
		CAMERA_ERR("Memory mapping error\n");
		return -1;
	}

	return fd;
}

int camera_deinit_helper(int fd, int buf_len)
{
	int ret, i;

	ret = camera_streamoff(fd);
	if (ret < 0) {
		CAMERA_ERR("Camera stream off error\n");
		return -1;
	}

	for (i = 0; i < buf_len; ++i) {
		ret = munmap(buffers[i].offset, buffers[i].length);

		if (ret == -1) {
			CAMERA_ERR("Memory unmap error : %s\n", strerror(errno));
			return -1;
		}
	}

	ret = close(fd);
	if (ret == -1) {
		CAMERA_ERR("Device close error : %s\n", strerror(errno));
		return -1;
	}

	return 0;
}

int camera_get_frame_helper(int fd, uint8_t **out, ssize_t *size)
{
	int ret;
	struct v4l2_buffer buf;
	CLEAR(buf);

	fd_set fds;
   struct timeval tv; 
   int r;

   FD_ZERO(&fds);
   FD_SET(fd, &fds);

   /* Timeout. */
   tv.tv_sec = 2;
   tv.tv_usec = 0;

   r = select(fd + 1, &fds, NULL, NULL, &tv);

   if (-1 == r) {
      if (EINTR != errno)
         perror("Select : ");
   }   

   if (0 == r) {
      fprintf(stderr, "select timeout\n");
      exit(EXIT_FAILURE);
   }

	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;

	ret = xioctl(fd, VIDIOC_DQBUF, &buf);
	if (ret < 0) {
		CAMERA_ERR("VIDIOC_DQBUF : %s\n", strerror(errno));
		return -1;
	}
	
//	CAMERA_DBG("Byte used : %d\n", buf.bytesused);
	memcpy(*out, buffers[buf.index].offset, buf.bytesused);
	*size = buf.bytesused;

	ret = xioctl(fd, VIDIOC_QBUF, &buf);
	if (ret < 0) {
		CAMERA_ERR("VIDIOC_QBUF : %s\n", strerror(errno));
		return -1;
	}

	return 0;
}

int camera_format_helper(int fd, int width, int height, int format)
{
	int ret;
	struct v4l2_format fmt;

	CLEAR(fmt);

	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	fmt.fmt.pix.width = width;
	fmt.fmt.pix.height = height;
	fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
	fmt.fmt.pix.pixelformat = format;

	ret = xioctl(fd, VIDIOC_S_FMT, &fmt);
	if (ret == -1) {
		CAMERA_ERR("Set format error : %s\n", strerror(errno));
		return -1;
	}

	return 0;
}

int camera_mmap_helper(int fd, int req_cnt)
{
	int ret, i;
	struct v4l2_requestbuffers req;

	CLEAR(req);

	req.count = req_cnt;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;

	ret = xioctl(fd, VIDIOC_REQBUFS, &req);
	if ( ret == -1 ) {
		if (errno == EINVAL) {
			CAMERA_ERR("Device not support mmap\n");
			return -1;
		}
		else {
			CAMERA_ERR("VIDIOC_REQBUFS : %s\n", strerror(errno));
			return -1;
		}
	}

	if (req.count != req_cnt) {
		CAMERA_ERR("Insufficient buffer memory\n");
		return -1;
	}

	buffers = (img_buffer *)calloc(req.count, sizeof(img_buffer));

	if ( !(buffers) ) {
		CAMERA_ERR("Out of memory\n");
		return -1;
	}

	for(i = 0; i < req.count; ++i) {
		struct v4l2_buffer buf;

		CLEAR(buf);

		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = i;

		ret = xioctl(fd, VIDIOC_QUERYBUF, &buf);
		if (ret == -1) {
			CAMERA_ERR("VIDIOC_REQBUF : %s\n", strerror(errno));
			return -1;
		}

		buffers[i].length = buf.length;
		buffers[i].offset = v4l2_mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);

		if (buffers[i].offset == MAP_FAILED) {
			CAMERA_ERR("mmap failed : %s\n", strerror(errno));
			return -1;
		}
	}

	return 0;
}



int camera_streamon(int fd, int buf_len)
{
	enum v4l2_buf_type type;
	struct v4l2_buffer buf;
	int ret, i;

//	ret = ioctl(fd, VIDIOC_S_INPUT, 0);

	for (i = 0; i < buf_len; ++i) {
		CLEAR(buf);

		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_MMAP;
		buf.index = i;

		ret = xioctl(fd, VIDIOC_QBUF, &buf);
		if (ret == -1) {
			CAMERA_ERR("VIDIOC_QBUF error : %s\n", strerror(errno));
			return -1;
		}
	}

	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		
	ret = xioctl(fd, VIDIOC_STREAMON, &type);
	if (ret == -1) {
		CAMERA_ERR("VIDIOC_STREAMON error : %s\n", strerror(errno));
		return -1;
	}

	return 0;
}

int camera_streamoff(int fd)
{
	int ret;

	enum v4l2_buf_type type;
	type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	ret = ioctl(fd, VIDIOC_STREAMOFF, &type);
	if (ret == -1) {
		CAMERA_ERR("VIDIOC_STREAMOFF : %s\n", strerror(errno));
		return -1;
	}

	return 0;
}
