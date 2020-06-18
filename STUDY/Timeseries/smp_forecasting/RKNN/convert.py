from rknn.api import RKNN

rknn = RKNN(verbose=True)

#load
ret = rknn.load_tensorflow(tf_pb='./lstm_tanh.pb',
			   inputs=['Placeholder'],
			   outputs=['fully_connected/Identity'],
			   input_size_list=[[ 28, 8]])
if ret !=0:
  print('Load failed!')
  exit(ret)
print('Load success!')

#build
ret = rknn.build(do_quantization=False)
if ret !=0:
	print('Build failed!')
	exit(ret)
print('Build success!')

#export
ret = rknn.export_rknn('./lstm.rknn') 
if ret != 0:
	print('Export failed!')
	exit(ret)
print('Saved model')

