 #include "ssd.h"

void ssd_post(float *predictions, float *output_classes, int w, int h, ssd_ctx *ctx)
{
	
	int output[2][NUM_RESULTS];
	int valid_count = 0, result_count = 0, i;

	ctx->detections.count = 0;
	
	decode_center_boxes(predictions, ctx->box_priors);
	
	valid_count = filter_vaild_result(output_classes, output, NUM_CLASS);
	if (valid_count > MAX_COUNT) {	// if valid_count is too much
		return;
	}
	

	/* detect nest box */
	nms(valid_count, predictions, output);

	

	for (i = 0; i < valid_count; ++i) {
		if (output[0][i] == -1) {
			continue;
		}

		int n = output[0][i];
		int topclass_index = output[1][i];

		int x1 = (int)(predictions[n*4 + 1] * w);
		int y1 = (int)(predictions[n*4 + 0] * h);
		int x2 = (int)(predictions[n*4 + 3] * w);
		int y2 = (int)(predictions[n*4 + 2] * h);

		if (x1 == 0 && x2 == 0 && y1 == 0 && y2 == 0)
			continue;

		memcpy(ctx->detections.objects[result_count].name, ctx->labels[topclass_index], 10);
		ctx->detections.objects[result_count].rect.left		= x1;
		ctx->detections.objects[result_count].rect.top		= y1;
		ctx->detections.objects[result_count].rect.right	= x2;
		ctx->detections.objects[result_count].rect.bottom	= y2;
		
		printf("(%d, %d) (%d, %d) @%d ",x1, y1, x2, y2, topclass_index);
		result_count++;
	}	

	ctx->detections.count = result_count;
}

int nms(int valid_count, float *box_locations, int (*classes)[NUM_RESULTS])
{
	int i, j, n, m;
	float xmin0, ymin0, xmax0, ymax0;
	float xmin1, ymin1, xmax1, ymax1;
	float iou;

	for (i = 0; i < valid_count; ++i) {
		if (classes[0][i] == -1) {
			continue;
		}

		n = classes[0][i];

		for (j = i+1; j < valid_count; ++j) {
			m = classes[0][j];
			if (m == -1) {
				continue;
			}

			ymin0 = box_locations[n*4 + 0];
			xmin0 = box_locations[n*4 + 1];
			ymax0 = box_locations[n*4 + 2];
			xmax0 = box_locations[n*4 + 3];

			ymin1 = box_locations[m*4 + 0];
			xmin1 = box_locations[m*4 + 1];
			ymax1 = box_locations[m*4 + 2];
			xmax1 = box_locations[m*4 + 3];

			iou = calculate_overlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);
			if (iou >= NMS_THRESHOLD) {
				classes[0][j] = -1;
			}
		}
	}
}

float calculate_overlap(
			float xmin0, float ymin0, float xmax0, float ymax0, 
			float xmin1, float ymin1, float xmax1, float ymax1)
{
	float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1));
	float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1));
	float i = w * h;
	float u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i;

	return u <= 0.f ? 0.f : (i / u);
}

float sigmoid(float x)
{
	return (float) (1.0 / (1.0 + expf(-x)));
}

int filter_vaild_result(float *output_classes, int (*output)[NUM_RESULTS], int num_classes)
{
	int valid_count = 0;
	int i, j;
	float topclass_score = -1000.0, score;
	int topclass_index = -1;

	
	// Scale them back to the input size (=> 무슨 말인지 모르겠음)
	for (i = 0; i < NUM_RESULTS; ++i) {
		topclass_score = (float)(-1000.0);
		topclass_index = -1;

		// Calculate the class index with the highest score for each grid
		// Skip the first catch-all class.
		for (j = 1; j < num_classes; ++j) {
			score = sigmoid(output_classes[i*num_classes+j]);
			if (score > topclass_score) {
				topclass_index = j;
				topclass_score = score;
			}
		}
		
		// If topclass score is larger than min threshold, increase valid_count
		if (topclass_score >= MIN_SCORE) {
			output[0][valid_count] = i;
			output[1][valid_count] = topclass_index;
			printf("%f %d\n", topclass_score, topclass_index);
			++valid_count;
		}
	}

	return valid_count;
}

void decode_center_boxes(float *predictions, float(*boxPriors)[NUM_RESULTS])
{
	/*
		Predictions

		+-------- 0 --------+-------- 1 --------+      +------ 1917 -------+

		+----+----+----+----+----+----+----+----+      +----+----+----+----+
		| cy | cx | h  | w  | cy | cx | h  | w  | .... | cy | cx | h  | w  |
		+----+----+----+----+----+----+----+----+      +----+----+----+----+ 

		Default Boxes format
		+-------+-------+       +----------+
		| cy[0] | cy[1] |  ...  | cy[1917] |
		+-------+-------+       +----------+
		| cx[0] | cx[1] |  ...  | cx[1917] |
		+-------+-------+       +----------+
		|  h[0] |  h[1] |  ...  |  h[1917] |
		+-------+-------+       +----------+
		|  w[0] |  w[1] |  ...  |  w[1917] |
		+-------+-------+       +----------+
	*/

	int i;
	float cx, cy, h, w;
	float ymin, xmin, ymax, xmax;

	for (i = 0; i < NUM_RESULTS; ++i) {
		cy = predictions[i * 4 + 0] / Y_SCALE * boxPriors[2][i] + boxPriors[0][i];
		cx = predictions[i * 4 + 1] / X_SCALE * boxPriors[3][i] + boxPriors[1][i];
		h = (float) expf(predictions[i * 4 + 2] / H_SCALE) * boxPriors[2][i];
		w = (float) expf(predictions[i * 4 + 3] / W_SCALE) * boxPriors[3][i];

		ymin = cy - (h / 2.0f);
		xmin = cx - (w / 2.0f);
		ymax = cy + (h / 2.0f);
		xmax = cx + (w / 2.0f);

		predictions[i*4 + 0] = ymin;
		predictions[i*4 + 1] = xmin;
		predictions[i*4 + 2] = ymax;
		predictions[i*4 + 3] = xmax;
	}
}

char *readLine(FILE *fp, char *buffer, int *len) 
{
    int ch;
    int i = 0;
    size_t buff_len = 0;

    buffer = (char *)malloc(buff_len + 1);
    if (!buffer)
        return NULL; // Out of memory

    while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
        buff_len++;
        void *tmp = realloc(buffer, buff_len + 1);
        if (tmp == NULL) {
            free(buffer);
            return NULL; // Out of memory
        }
        buffer = (char *)tmp;

        buffer[i] = (char)ch;
        i++;
    }
    buffer[i] = '\0';

    *len = buff_len;

    // Detect end
    if (ch == EOF && (i == 0 || ferror(fp))) {
        free(buffer);
        return NULL;
    }
    return buffer;
}

int readLines(const char *fileName, char *lines[]) 
{
    FILE *file = fopen(fileName, "r");
    char *s;
    int i = 0;
    int n = 0;
    while ((s = readLine(file, s, &n)) != NULL) {
        lines[i++] = s;
    }
    return i;
}

int loadLabelName(const char *locationFilename, char *labels[]) 
{
    readLines(locationFilename, labels);
    return 0;
}

int loadBoxPriors(const char *locationFilename, float (*boxPriors)[NUM_RESULTS]) {
    const char *d = " ";
    char *lines[4];
    int count = readLines(locationFilename, lines);
    // printf("line count %d\n", count);
    // for (int i = 0; i < count; i++) {
    // printf("%s\n", lines[i]);
    // }
    for (int i = 0; i < 4; i++) {
        char *line_str = lines[i];
        char *p;
        p = strtok(line_str, d);
        int priorIndex = 0;
        while (p) {
            float number = (float)(atof(p));
            boxPriors[i][priorIndex++] = number;
            p = strtok(NULL, d);
        }
        if (priorIndex != NUM_RESULTS) {
            printf("error\n");
            return -1;
        }
    }
    return 0;
}

void draw_rect( rga_transform_t dst, int16_t left, int16_t top, int16_t right, int16_t bottom)
{
    int x,y;
    
            if(top < 0 + 2)             top =0 + 2;
            if(top > DISP_H - 2)        top = DISP_H - 2;
            if(bottom < 0 + 2)          bottom =0 + 2 ; 
            if(bottom > DISP_H - 2)     bottom = DISP_H - 2;

            if(left < 0 + 2)            left =0 + 2;
            if(left > DISP_W - 2 )      left = DISP_W - 2;
            if(right < 0 + 2)           right =0 + 2;
            if(right > DISP_W - 2)      right = DISP_W - 2;

            for(y = top*DISP_H/300; y <  bottom*DISP_H/300; y++)
            {   
                for(x = left*DISP_W/300; x< right*DISP_W/300; x++)
                {   
                    if(x == left*DISP_W/300 || x == right*DISP_W/300-1
                         || y ==  top*DISP_H/300 || y == bottom*DISP_H/300-1)
                    {   
//                      dst.data[(x*4)+(DISP_W*y*4)]=0xFF;
//                      dst.data[((x+1)*4)+(DISP_W*(y+1)*4)]=0xFF;
//                      dst.data[((x+2)*4)+(DISP_W*(y+2)*4)]=0xFF;

                        dst.data[(x*4)+1+(DISP_W*y*4)]=0xFF;
                        dst.data[((x+1)*4+1)+(DISP_W*(y+1)*4)]=0xFF;
                        dst.data[((x+2)*4+1)+(DISP_W*(y+2)*4)]=0xFF;

//                      dst.data[(x*4)+2+(DISP_W*y*4)]=0xFF;
//                      dst.data[((x+1)*4+2)+(DISP_W*(y+1)*4)]=0xFF;
//                      dst.data[((x+2)*4+2)+(DISP_W*(y+2)*4)]=0xFF;

                        dst.data[(x*4)+3+(DISP_W*y*4)]=0xFF;
                        dst.data[((x+1)*4+3)+(DISP_W*(y+1)*4)]=0xFF;
                        dst.data[((x+2)*4+3)+(DISP_W*(y+2)*4)]=0xFF;

                    }   
                }   
            }   
}

void draw_boundingbox(ssd_ctx ctx , rga_transform_t dst)
{
    int i;

        for(i = 0 ; i<ctx.detections.count ; i++){
            draw_rect(dst , ctx.detections.objects[i].rect.left, ctx.detections.objects[i].rect.top,
                        ctx.detections.objects[i].rect.right, ctx.detections.objects[i].rect.bottom);
        }
}



int ssd_init(const char *model_name, const char *bbox_path, const char *label_path, ssd_ctx *ctx)
{
	int ret;

	ret = rknn_init_helper(model_name, &ctx->rknn);
	if (ret != 0) {
		fprintf(stderr, "%s : Failed to load model", __func__);
		return -1;
	}
	
	ret = loadLabelName(label_path, ctx->labels);
	if (ret != 0) {
		fprintf(stderr, "%s : Failed to load label", __func__);
		return -1;
	}
	
	ret = loadBoxPriors(bbox_path, ctx->box_priors);
	if (ret != 0) {
		fprintf(stderr, "%s : Failed to load priors", __func__);
		return -1;
	}
	
	return 0;
}

int ssd_run(ssd_ctx *ctx, uint8_t *img, int w, int h, ssize_t size)
{
	int ret;
	rknn_input inputs[1];
	rknn_output outputs[2];

	memset(inputs, 0x00, sizeof(inputs));
	inputs[0].index = 0;
	inputs[0].type = RKNN_TENSOR_UINT8;
	inputs[0].size = size;
	inputs[0].fmt = RKNN_TENSOR_NHWC;
	//inputs[0].buf = img;
	inputs[0].buf = IMAGE;


	//printf("input set start \n");
	ret = rknn_inputs_set(ctx->rknn, 1, inputs);
	if (ret < 0) {
		fprintf(stdout, "%s fail\n", __func__);
		return -1;
	}
	//printf("input set \n");
	ret = rknn_run(ctx->rknn, NULL);
	if (ret < 0) {
		fprintf(stdout, "%s fail\n", __func__);
		return -1;
	}

	//printf("rknn_run done\n");

	memset(outputs, 0x00, sizeof(outputs));
	outputs[0].want_float = 1;
	outputs[1].want_float = 1;

	ret = rknn_outputs_get(ctx->rknn, 2, outputs, NULL);
	if (ret < 0) {
		fprintf(stdout, "%s fail\n", __func__);
		return -1;
	}

	//printf("get output done\n");
	//SSD 후처리코드
	ssd_post((float *)outputs[0].buf, (float *)outputs[1].buf, w, h, ctx);
	rknn_outputs_release(ctx->rknn, 2, outputs);
	//printf("ssd post done\n");
	return 0;
}
/*
int main()
{
//ssd_init(const char *model_name, const char *bbox_path, const char *label_path, ssd_ctx *ctx);
//ssd_run(ssd_ctx *ctx, uint8_t *img, int w, int h, ssize_t size);	

	ssd_ctx ctx;
	uint8_t *img;

	ssd_init("model.rknn", "prior_box.txt", "labels.txt", &ctx);
	ssd_run(&ctx, img, 300, 300, 90000);

}
*/
