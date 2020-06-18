from rknn.api import RKNN

rknn = RKNN(verbose=True)

print('--> Loading model')
ret = rknn.load_tensorflow(tf_pb = './decision.pb',
                           inputs = ['segment/conv2d/Conv2D'],
                           outputs = ['segment/Softmax'],
                           input_size_list =[[1280, 512, 1]])
if ret !=0:
  print('Load failed!')
  exit(ret)
print('done')

print('--> Building model')
ret = rknn.build(do_quantization=False, dataset='./dataset.txt')

if ret != 0:
  print('Build failed!')
  exit(ret)
print('done')

print('--> Export RKNN model')
ret = rknn.export_rknn('./segmentation.rknn')

if ret != 0:
  print('Export failed!')
  exit(ret)
print('done')
