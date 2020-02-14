#include "rknn_helper.h"
		
static uint8_t *load_model(const char *filename, int *model_size)
{
	// Open Model File
	FILE *fp = fopen(filename, "rb");
	if (fp == NULL) {
		printf("Model %s fopen fail\n", filename);
		return NULL;
	}

	// Calculation Model File Size
	fseek(fp, 0, SEEK_END);
	const int model_len = ftell(fp);
	uint8_t *model = (uint8_t *)malloc(model_len);
	fseek(fp, 0, SEEK_SET);

	// Read Model into pointer
	if (model_len != fread(model, 1, model_len, fp)) {
		printf("Model %s fread fail\n", filename);
		free(model);
		return NULL;
	}

	*model_size = model_len;

	if (fp) {
		fclose(fp);
	}

	return model;
}

int rknn_init_helper(const char* filename, rknn_context *ctx)
{
	int model_len, ret;
	uint8_t *model;

	model = load_model(filename, &model_len);
	if (model == NULL) {
		printf("Failed to load model\n");
		return -1;
	}

	ret = rknn_init(ctx, model, model_len, 0);
	if (ret != RKNN_SUCC) {
		printf("Failed to initialize rknn\n");
		return -1;
	}

	free(model);
	return 0;
}

int rknn_deinit_helper(rknn_context ctx)
{
	// Destory and Deinitialize rknn_context
	int ret = rknn_destroy(ctx);
	if (ret != RKNN_SUCC) {
		RKNN_ERR("Failed to destroy rknn context\n");
		return -1;
	}

	return 0;
}

int rknn_run_helper(rknn_context ctx, void *data, uint32_t size, void *out)
{
	int ret;

	rknn_input input;
	rknn_output output;

	memset(&input, 0x00, sizeof(input));
	memset(&output, 0x00, sizeof(output));

	// Set rknn input data
	input.index = 0;
	input.type = RKNN_TENSOR_UINT8;
	input.size = size;
	input.fmt = RKNN_TENSOR_NHWC;
	input.buf = data;
	input.pass_through = 1;	

	ret = rknn_inputs_set(ctx, 1, &input);
	if (ret != RKNN_SUCC) {
		printf("Failed to set rknn inputs\n");
		return -1;
	}

	//	Run inference
	ret = rknn_run(ctx, NULL);
	if (ret != RKNN_SUCC) {
		RKNN_ERR("Failed to run rknn\n");
		return -1;
	}

	// Get RKNN Output
	output.want_float = 0;
	output.is_prealloc = 1;		//Use pre-allocated buffer
	output.buf = out;
	output.size = size;

	ret = rknn_outputs_get(ctx, 1, &output, NULL);
	if (ret != RKNN_SUCC) {
		RKNN_ERR("Failed to get rknn outputs\n");
		return -1;
	}

	return 0;
}


