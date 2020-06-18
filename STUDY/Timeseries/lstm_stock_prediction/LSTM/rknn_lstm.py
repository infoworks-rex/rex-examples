#!/usr/bin/env python
# coding: utf-8


import numpy as np
from rknn.api import RKNN



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
output_data_column_cnt = 1

seq_length = 28
rnn_cell_hidden_dim = 20
forget_bias = 1.0
num_stacked_layers = 1
keep_prob = 1.0

epoch_num = 1000
learning_rate = 0.01



if __name__ == '__main__':

	rknn = RKNN(verbose=True)
	print('--> Loading model')
	rknn.load_rknn('./RNN_RKNN.rknn')

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
	rknn.build(do_quantization=False, dataset='./AAPL_5Y_sq.csv')
	print('done')

	# Export RKNN Model
	rknn.export_rknn('./LSTM_RKNN.rknn')

	# Direct Load RKNN Model
	rknn.load_rknn('./LSTM_RKNN.rknn')

	stock_file_name = 'AAPL_5Y.csv'

	storage = np.genfromtxt(stock_file_name, delimiter=',', dtype=np.float32)
	storage = np.delete(storage, (0), axis=0)
	storage = np.delete(storage, (0), axis=1)
	print(storage)

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
	
	y = x[:, [-2]]
	print("y[0]: ",y[0])
	print("y[-1]: ",y[-1])
	
	
	recent_data = np.array([x[len(x)-seq_length : ]])
	
	recent_data = np.squeeze(recent_data, axis=0)

	print("recent_data.shape:", recent_data.shape)
	print("recent_data:", recent_data)

	# init runtime environment
	print('--> Init runtime environment')
	ret = rknn.init_runtime()
	if ret != 0:
		print('Init runtime environment failed')
	
	# Inference
	print('--> Running model')
	test_predict = rknn.inference(inputs=[recent_data])
	#test_predict = np.array(test_predict, dtype = np.float64)
	print('done')
	print('inference result: ', test_predict)


	# Evaluate Perf on Simulator
	#rknn.eval_perf(inputs=[recent_data])

	#test_predict = sess.run(hypothesis, feed_dict={X: recent_data})

	print("test_predict", test_predict)
	test_predict = reverse_min_max_scaling(price,test_predict)
	print("Tomorrow's stock price", test_predict[0])

	# Release RKNN Context
	rknn.release()

