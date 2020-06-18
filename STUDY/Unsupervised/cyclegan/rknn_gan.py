#!/usr/bin/env python
# coding: utf-8


import numpy as np
import tensorflow as tf

from rknn.api import RKNN


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('input', 'input_sample.jpg', 'input image path (.jpg)')
tf.flags.DEFINE_string('output', 'output_sample.jpg', 'output image path (.jpg)')
tf.flags.DEFINE_integer('image_size', '256', 'image size, default: 256')

def convert2int(image):
	return tf.image.convert_image_dtype((image+1.0)/2.0, tf.uint8)

if __name__ == '__main__':

	rknn = RKNN()
	print('--> Loading model')
	
	#rknn.config(channel_mean_value='0 0 0 255', reorder_channel='0 1 2')


	# Load TensorFlow Model
	print('--> Loading model')
	rknn.load_tensorflow(tf_pb='./horse2zebra.pb',
                     inputs=['G_7/c7s1_32/Pad'],
                     outputs=['G_7/output/Tanh'],
                     input_size_list=[[256, 256, 3]])
	print('done')

	# Build Model
	print('--> Building model')
	rknn.build(do_quantization=False)
	print('done')

	# Export RKNN Model
	rknn.export_rknn('./gan_rknn.rknn')
	
	# Direct Load RKNN Model
	rknn.load_rknn('./gan_rknn.rknn')

	# init runtime environment
	print('--> Init runtime environment')
	ret = rknn.init_runtime()
	if ret != 0:
		print('Init runtime environment failed')
	

	# Inference
	print('--> Running model')

	#전처리 과정

	with tf.gfile.FastGFile(FLAGS.input, 'rb') as f:
		image_data = f.read()
		input_image = tf.image.decode_jpeg(image_data, channels=3)
		input_image = tf.image.resize_images(input_image, size=(FLAGS.image_size, FLAGS.image_size))
		input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
		input_image = (input_image/127.5) - 1.0
		input_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])
		input_image = tf.Session().run(input_image)

	output_image = rknn.inference(inputs=[input_image])
	output_image = np.array(output_image, dtype = np.float64)
	print("output_image = ", output_image.shape)

	#후처리 과정
	
	image = tf.map_fn(convert2int, output_image, dtype=tf.uint8)
	print("image = ", image.shape)
	image = tf.image.encode_jpeg(tf.squeeze(image))
	image = tf.Session().run(image)
	
	with open(FLAGS.output, 'wb') as f:
		f.write(image)

	# Evaluate Perf on Simulator
	#rknn.eval_perf()

	# Release RKNN Context
	rknn.release()

