#-*- coding:utf-8 -*-
from __future__ import print_function
import os
import cv2
import time
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import *
from model import *
from glob import glob
from PIL import Image
from rknn.api import RKNN
from skimage import color,filters

rknn = RKNN() #verbose=True
height = 0
width = 0
channels = 0

# Load RKNN Model
print('-> Load RKNN model 1')
ret1 = rknn.load_rknn('./1.rknn')
print('done')

###load eval data
eval_low_data = []
eval_img_name = []
eval_low_data_name = glob('./test/*')
eval_low_data_name.sort()
for idx in range(len(eval_low_data_name)):
    [_, name] = os.path.split(eval_low_data_name[idx])
    suffix = name[name.find('.') + 1:]
    name = name[:name.find('.')]
    eval_img_name.append(name)
    eval_low_im = load_images(eval_low_data_name[idx])
    eval_low_data.append(eval_low_im)
    print(eval_low_im.shape) # (400, 600, 3)
    height, width, channels = eval_low_im.shape # test 사진의 원본 크기 저장 >> 최종이미지 저장시 사용함
    input_low = eval_low_data[idx]
    input_low_eval = np.expand_dims(input_low, axis=0) # (1, 400, 600, 3)

sample_dir = './results/test/'
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)

# DecomNet 실행
print('-> Running model 1')
ret1 = rknn.init_runtime()
if ret1 != 0:
    print('Init runtime environment 1 failed')
    exit(ret1)
print('done')

print('-> Inference model 1')
decom_r_low, decom_i_low = rknn.inference(inputs=[input_low_eval]) # decom_r_low(3채널)
#print(type(decom_r_low)) #numpy.ndarray
print('=> 1 run success')

#image save (성공)
save_images(os.path.join(sample_dir, 'decom_r_low.png'), decom_r_low)
save_images(os.path.join(sample_dir, 'decom_i_low.png'), decom_i_low)

#------------------------------------------------------------------------------------------------------------
# Load RKNN Model
print('--> Load RKNN model 2')
ret2 = rknn.load_rknn('./2.rknn')
print('done')

# RestorationNet 실행
print('--> Running model 2')
ret2 = rknn.init_runtime()
if ret2 != 0:
    print('Init runtime environment 2 failed')
    exit(ret2)
print('done')

print('--> Inference model 2')
## load image
decom_r_low = load_images('./results/test/decom_r_low.png')
decom_i_low = load_images('./results/test/decom_i_low.png') # (400, 600)
restoration_r = rknn.inference(inputs=[decom_r_low, decom_i_low]) #restoration_r(3채널)
print('==> 2 run success')

#image save (성공)
save_images(os.path.join(sample_dir, 'restoration_r.png'), restoration_r)
restoration_r = load_images('./results/test/restoration_r.png') # (400, 600, 3)
#------------------------------------------------------------------------------------------------------------
#different exposure level을 얻기위해 ratio를 0-5.0 사이 값으로 바꿀 수 있음
ratio = 17.0 # 17(정수)로 쓰면 에러
i_low_data_ratio = np.ones([height, width])*(ratio) # (400, 600)
i_low_ratio_expand = np.expand_dims(i_low_data_ratio , axis =2) # (400, 600, 1)
#i_low_ratio_expand2 = np.expand_dims(i_low_ratio_expand, axis=0) # (1, 400, 600, 1)
i_low_ratio_expand = i_low_ratio_expand.astype(np.uint8)

# Load RKNN Model
print('---> Load RKNN model 3')
ret3 = rknn.load_rknn('./3.rknn')
print('done')

# AdjustmentNet 실행
print('---> Running model 3')
ret3 = rknn.init_runtime()
if ret3 != 0:
    print('Init runtime environment 3 failed')
    exit(ret3)
print('done')

print('---> Inference model 3')
#decom_i_low = load_images('./results/test/decom_i_low.png') # load image
adjust_i = rknn.inference(inputs=[decom_i_low, i_low_ratio_expand]) #adjust_i(1채널)
print('===> 3 run success')

#image save
save_images(os.path.join(sample_dir, 'adjust_i.png'), adjust_i)

#------------------------------------------------------------------------------------------------------------
# 어두운 부분 복구하기위한 operator
decom_r_sq = np.squeeze(decom_r_low)
r_gray = color.rgb2gray(decom_r_sq)
r_gray_gaussion = filters.gaussian(r_gray, 3)
low_i =  np.minimum((r_gray_gaussion*2)**0.5,1)
low_i_expand_0 = np.expand_dims(low_i, axis = 0)
low_i_expand_3 = np.expand_dims(low_i_expand_0, axis = 3)
result_denoise = restoration_r*low_i_expand_3 # result_denoise (1,400,600,3)
result_denoise = np.squeeze(result_denoise, axis=0) # (400,600,3)

adjust_i = load_images('./results/test/adjust_i.png') # adjust_i (400,600)
adjust_i = np.expand_dims(adjust_i, axis=2) # adjust_i (400, 600, 1) <class 'numpy.ndarray'>

fusion4 = result_denoise*adjust_i # fusion4 (400, 600, 3) <class 'numpy.ndarray'>

decom_i_low = np.expand_dims(decom_i_low, axis=2) # (400, 600, 1)

# over-exposure를 방지하기 위해 original input과 융합 //fusion = restoration_r*adjust_i
fusion2 = decom_i_low*input_low_eval + (1-decom_i_low)*fusion4
# 최종 출력 이미지 저장
save_images(os.path.join(sample_dir, '%s_kindle.png' % (name)), fusion2)

