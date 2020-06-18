#ifndef _RKNN_HELPER_H
#define _RKNN_HELPER_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>

#include <rknn_api.h>
#include "utils.h"

extern "C"{
#define RKNN_ERR(x,arg...)							\
	ERR_MSG("[RKNN Error]" x,##arg)

#define RKNN_DBG(x,arg...)							\
	DBG_MSG("[RKNN Debug]" x,##arg)

}

/**
* @brief Read rknn model (Not load to NPU)
*
* @param filename	RKNN Filename
* @param model_size Pointer to store model size
*
* @return When success return model's file pointer, Otherwise return NULL
*/
static uint8_t *load_model(const char *filename, int *model_size);

/**
* @brief Initialize RKNN Context
*
* @param filename	RKNN Model filename
* @param ctx pointer to store initialized rknn_context
*
* @return When success return 0, Otherwise return -1
*/
int rknn_init_helper(const char* filename, rknn_context *ctx);

/**
* @brief Deinitialize RKNN Context
*
* @param ctx structure to deinitialize
*
* @return When success return 0, Otherwise return -1 
*/
int rknn_deinit_helper(rknn_context ctx);

/**
* @brief Set inputs to rknn context, run inference, store rknn output
*
* @param ctx	RKNN Context
* @param data	Input data
* @param size	Input data size
* @param out	pointer to store rknn output
*
* @return When success return 0, Otherwise return -1
*/
int rknn_run_helper(rknn_context ctx, void *data, uint32_t size, void *out);

#endif
