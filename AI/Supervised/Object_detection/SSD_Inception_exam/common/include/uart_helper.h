#ifndef _UART_HELPER_H_
#define _UART_HELPER_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/signal.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>

#include "utils.h"

#define UART_ERR(x, arg...) \
	ERR_MSG("[UART Error] " x,##arg)

#define UART_DBG(x, arg...) \
	DBG_MSG("[UART Debug] " x,##arg)

typedef void (*sa_handler_t)(int);

int uart_init_helper(const char* device, speed_t baudrate, sa_handler_t hnd, int *fd);

int uart_deinit_helper(int fd);

#endif
