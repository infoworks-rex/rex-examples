#!/usr/bin/env python
# coding: utf-8

import os,sys
import numpy as np
import tensorflow as tf
import cv2

from rknn.api import RKNN

OUT_DIR = "output"

PRESET = 256 


def main(folder="test"):
	folder = folder
	files = os.listdir(folder)
	
	for i in range(len(files)):
		img = cv2.imread("{}/{}".format(folder,files[i]))
		img = (img-127.5)/127.5
		h,w = img.shape[:2]
		print("w, h = ", w, h)
		input = cv2.resize(img, (PRESET, PRESET), interpolation=cv2.INTER_CUBIC)
		input = input.reshape(PRESET, PRESET, 3)
		input = np.array(input, dtype=np.float32)

		rknn = RKNN()
		print('--> Loading model')
	
		#rknn.config(channel_mean_value='0 0 0 255', reorder_channel='0 1 2')


		# Load TensorFlow Model
		print('--> Loading model')
		rknn.load_tensorflow(tf_pb='pretrained/SR_freeze.pb',
                     		inputs=['ESRGAN_g/Conv2D'],
                     		outputs=['output_image'],
                     		input_size_list=[[PRESET, PRESET, 3]])
		print('done')

		# Build Model
		print('--> Building model')
		rknn.build(do_quantization=False)
		print('done')

		# Export RKNN Model
		rknn.export_rknn('./sr_rknn.rknn')
	
		# Direct Load RKNN Model
		rknn.load_rknn('./sr_rknn.rknn')

		# init runtime environment
		print('--> Init runtime environment')
		ret = rknn.init_runtime()
		if ret != 0:
			print('Init runtime environment failed')
	

		# Inference
		print('--> Running model')

		output_image = rknn.inference(inputs=[input])
		print('complete')
		out = np.array(output_image, dtype = np.float64)
		print("output_image = ", out.shape)
		out = np.squeeze(out)

		Y_ = out.reshape(PRESET*4, PRESET*4, 3)
		Y_ = cv2.resize(Y_, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
		print("output shape is ",Y_.shape)

		#후처리 과정
	
		Y_ = (Y_ + 1)*127.5
		cv2.imwrite("{}/{}_yval.png".format(OUT_DIR, i), Y_)

		# Evaluate Perf on Simulator
		#rknn.eval_perf()

		# Release RKNN Context
		rknn.release()

if __name__ == '__main__':
    folder = "test"
    try:
        folder = sys.argv[1]
    except:
        pass
    main(folder)
