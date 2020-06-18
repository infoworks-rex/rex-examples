#ifndef __SSD_H__
#define __SSD_H__

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include <rknn_api.h>
#include "rknn_helper.h"
#include "rga_helper.h"
#include "configs.h"

/*
	Detection Classes 91
*/

#define NUM_CLASS 7

/*
	(Grid 19x19, 3 Boxes) + (Grid 10x10, 6 Boxes) + (Grid 5x5, 6 Boxes)
	(Grid 3x3, 6 Boxes) + (Grid 2x2, 6 Boxes) + (Grid 1x1, 6 Boxes)
	= 1917
*/

/**
* @brief SSD Bounding Box coord information
*/
typedef struct
{
	int16_t left;
	int16_t top;
	int16_t right;
	int16_t bottom;
} bbox_t;

/**
* @brief 
*/
typedef struct
{
	char name[10];
	bbox_t rect;
} ssd_object;

typedef struct
{
	int count;
} ssd_detections;

typedef struct 
{
	rknn_context rknn;
	char *labels[NUM_CLASS];
	ssd_detections detections;
} ssd_ctx;


void draw_rect( rga_transform_t dst, int16_t left, int16_t top, int16_t right, int16_t bottom);
void draw_boundingbox(ssd_ctx ctx , rga_transform_t dst);

int ssd_init(const char *model_name, const char *label_path, ssd_ctx *ctx);
int ssd_run(ssd_ctx *ctx, uint8_t *img, int w, int h, ssize_t size);
void ssd_post(float *outputs, int w, int h, ssd_ctx *ctx);
int loadLabelName(const char *locationFilename, char *labels[]);
int readLines(const char *fileName, char *lines[]);
char *readLine(FILE *fp, char *buffer, int *len);
float sigmoid(float x);
float calculate_overlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1);


#endif
