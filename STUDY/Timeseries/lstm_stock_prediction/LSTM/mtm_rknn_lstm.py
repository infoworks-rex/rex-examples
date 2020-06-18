#!/usr/bin/env python
# coding: utf-8


import numpy as np
from rknn.api import RKNN
import datetime
import os
import csv
import pandas as pd


# Standardization
def data_standardization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()

# Min-Max scaling
def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)

def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()


input_data_column_cnt = 6
output_data_column_cnt = 6

seq_length = 28
rnn_cell_hidden_dim = 120
forget_bias = 1.0
num_stacked_layers = 7
keep_prob = 1.0

epoch_num = 1000
learning_rate = 0.01

predict_days = 30


if __name__ == '__main__':

	rknn = RKNN()
	print('--> Loading model')

	#rknn.config(channel_mean_value='0 0 0 255', reorder_channel='0 1 2')


	# Load TensorFlow Model
	print('--> Loading model')
	rknn.load_tensorflow(tf_pb='./freeze.pb',
                     inputs=['Placeholder'],
                     outputs=['fully_connected/Identity'],
                     input_size_list=[[28, 6]])
	print('done')

	# Build Model
	print('--> Building model')
	rknn.build(do_quantization=False)
	print('done')

	# Export RKNN Model
	rknn.export_rknn('./MTM_LSTM_RKNN.rknn')

	# Direct Load RKNN Model
	rknn.load_rknn('./MTM_LSTM_RKNN.rknn')

	stock_file_name = 'AAPL_5Y.csv'
	encoding = 'euc-kr'
	names = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
	
	# Read and Delete Axis
	raw_dataframe = pd.read_csv(stock_file_name, names=names, encoding=encoding)
	today = raw_dataframe.values[-1, 0]
	del raw_dataframe['Date']
	storage = raw_dataframe.values[1:].astype(np.float32)

	"""
	pandas 안쓰는 방법

	storage = np.genfromtxt(stock_file_name, encoding='euc-kr', delimiter=',')
	today = storage[-1,0]
	print("today = ", today)
	storage = np.genfromtxt(stock_file_name, delimiter=',', dtype=np.float32)
	storage = np.delete(storage, (0), axis=0)
	storage = np.delete(storage, (0), axis=1)
	print(storage)
	"""

	price = storage[:,:-1]
	norm_price = min_max_scaling(price)
	print("price.shape: ", price.shape)
	print("price[0]: ", price[0])
	print("norm_price[0]: ", norm_price[0])
	print("="*100)
	
	volume = storage[:,-1:]
	norm_volume = min_max_scaling(volume)
	print("volume.shape: ", volume.shape)
	print("volume[0]: ", volume[0])
	print("norm_volume[0]: ", norm_volume[0])
	print("="*100)
	
	x = np.concatenate((norm_price, norm_volume), axis=1)
	print("x.shape: ", x.shape)
	print("x[0]: ", x[0])
	print("x[-1]: ", x[-1])
	print("="*100)
	
	y = x
	print("y[0]: ",y[0])
	print("y[-1]: ",y[-1])
	
	
	recent_data = np.array([x[len(x)-seq_length : ]])
	time = datetime.datetime.strptime(today, '%Y-%m-%d')	

	recent_data = np.squeeze(recent_data, axis=0)

	print("recent_data.shape:", recent_data.shape)
	print("recent_data:", recent_data)

	# init runtime environment
	print('--> Init runtime environment')
	ret = rknn.init_runtime()
	if ret != 0:
		print('Init runtime environment failed')
	
	for i in range(predict_days):

		# Inference
		#print('--> Running model')
		test_predict = rknn.inference(inputs=[recent_data])
		test_predict = np.array(test_predict, dtype = np.float32)
		test_predict = np.squeeze(test_predict)
		#print("reselt = ", test_predict)
		#print(test_predict.shape)
		
		# Evaluate Perf on Simulator
		#rknn.eval_perf(inputs=[recent_data])

		test_predict1 = reverse_min_max_scaling(price,test_predict[0:5]) 
		test_predict2 = reverse_min_max_scaling(volume,test_predict[5])
		real_test_predict = np.append(test_predict1, test_predict2)    		  

		time = time + datetime.timedelta(days=1)
		str_time = time.strftime("%Y-%m-%d")

		print(str_time, "'s stock price = ", np.array(real_test_predict[0:5], dtype=np.float16))

		# 원하는 파일에 저장
		f = open("predict.csv", 'a', newline='')
		wr = csv.writer(f)
		wr.writerow([str_time, real_test_predict[0], real_test_predict[1], real_test_predict[2],
		                real_test_predict[3], real_test_predict[4], real_test_predict[5]])
		f.close()
		
		
		# 후처리
		test_predict = np.expand_dims(test_predict, axis=0)
		recent_data = np.concatenate((recent_data, test_predict), axis=0)
		recent_data = np.delete(recent_data, [0], axis=0)
		
		

	# Release RKNN Context
	rknn.release()

