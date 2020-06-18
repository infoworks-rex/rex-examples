#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from rknn.api import RKNN

# Create RKNN object
rknn = RKNN()

# Load TensorFlow Model
print('--> Loading model')
rknn.load_tensorflow(tf_pb='./freeze.pb',
                     inputs=['transpose'],
                     outputs=['fully_connected/Identity'],
                     input_size_list=[[28, 6]])
print('done')

# Build Model
print('--> Building model')
rknn.build(do_quantization=False)
print('done')

# Export RKNN Model
rknn.export_rknn('./RNN_RKNN.rknn')

# Direct Load RKNN Model
# rknn.load_rknn('./ssd_mobilenet_v1_coco.rknn')

rknn.release()

