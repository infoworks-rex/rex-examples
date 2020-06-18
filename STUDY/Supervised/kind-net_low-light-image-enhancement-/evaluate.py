# coding: utf-8
from __future__ import print_function
import os
import time
import random
from PIL import Image
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from glob import glob
from skimage import color,filters

tf.compat.v1.global_variables_initializer() #텐서플로 프로그램에서는 모든 변수를 초기화하기 위하여 global_variables_initializer()라는 특별한 함수를 명시적으로 호출해야 한다.
sess = tf.compat.v1.Session()

input_decom = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_decom') # placeholder는 데이터를 입력받는 비어있는 변수로, 여기서는 input값인 저조도 이미지를 받아온다. 
input_low_r = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_low_r')
input_low_i = tf.compat.v1.placeholder(tf.float32, [None, None, None, 1], name='input_low_i')
input_high_r = tf.compat.v1.placeholder(tf.float32, [None, None, None, 3], name='input_high_r')
input_high_i = tf.compat.v1.placeholder(tf.float32, [None, None, None, 1], name='input_high_i')
input_low_i_ratio = tf.compat.v1.placeholder(tf.float32, [None, None, None, 1], name='input_low_i_ratio')

[R_decom, I_decom] = DecomNet_simple(input_decom) #해당 함수는 model.py파일에 정의되어있다. 
decom_output_R = R_decom
decom_output_I = I_decom
output_r = Restoration_net(input_low_r, input_low_i)
output_i = Illumination_adjust_net(input_low_i, input_low_i_ratio)

# 학습된 가중치를 담을 변수를 선언하는 부분
var_Decom = [var for var in tf.compat.v1.trainable_variables() if 'DecomNet' in var.name] #
var_adjust = [var for var in tf.compat.v1.trainable_variables() if 'Illumination_adjust_net' in var.name]
var_restoration = [var for var in tf.compat.v1.trainable_variables() if 'Restoration_net' in var.name]

# 변수를 갱신하고 저장하는 새로운 변수saver_NNN을 만드는 부분
saver_Decom = tf.compat.v1.train.Saver(var_list = var_Decom) #tf.train.Saver()는 텐서플로우에서 모델과 파라미터를 저장하고(save)하고 불러올수(restore) 있게 만들어주는 API이다. (*)
saver_adjust = tf.compat.v1.train.Saver(var_list=var_adjust) #여기서는 파라미터를 저장한다. (*)
saver_restoration = tf.compat.v1.train.Saver(var_list=var_restoration)

#체크포인트 생성 경로를 지정하고, 각각의 모듈별로 체크포인트 생성이 안 될 경우 어디서 에러가 났는지 에러메세지 출력하는 부분
decom_checkpoint_dir ='./checkpoint/decom_net_train/' 
ckpt_pre=tf.compat.v1.train.get_checkpoint_state(decom_checkpoint_dir)
if ckpt_pre:
    print('loaded '+ckpt_pre.model_checkpoint_path)
    saver_Decom.restore(sess,ckpt_pre.model_checkpoint_path) #여기서는 파라미터를 불러온다. (*)
else:
    print('No decomnet checkpoint!')

checkpoint_dir_adjust = './checkpoint/illumination_adjust_net_train/'
ckpt_adjust=tf.compat.v1.train.get_checkpoint_state(checkpoint_dir_adjust)
if ckpt_adjust:
    print('loaded '+ckpt_adjust.model_checkpoint_path)
    saver_adjust.restore(sess,ckpt_adjust.model_checkpoint_path)
else:
    print("No adjust pre model!")

checkpoint_dir_restoration = './checkpoint/Restoration_net_train/'
ckpt=tf.compat.v1.train.get_checkpoint_state(checkpoint_dir_restoration)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver_restoration.restore(sess,ckpt.model_checkpoint_path)
else:
    print("No restoration pre model!")

###load eval data
eval_low_data = []
eval_img_name =[]
eval_low_data_name = glob('./test/*') #특정 파일만 모아서 출력하기. 여기서는 지정된 경로의 모든('*')파일을 불러오고
eval_low_data_name.sort() #정렬한다. 여기서는 파일이름이 모두 숫자라서 정렬 가능
for idx in range(len(eval_low_data_name)): #이미지 갯수(횟수)만큼 반복하는데
    [_, name] = os.path.split(eval_low_data_name[idx])
    suffix = name[name.find('.') + 1:] # "파일이름 = 파일인덱스+1" 이므로 인덱스에 1을더해서 파일이름과 name의 인덱스값을 맞춰준다. /suffix=접미사 
    name = name[:name.find('.')] #
    eval_img_name.append(name) #eval_img_name리스트에 name값을 하나씩 추가해준다
    eval_low_im = load_images(eval_low_data_name[idx]) 
    eval_low_data.append(eval_low_im)
    print(eval_low_im.shape) #출력 양식은 이미지 shape으로, 예를들면 "(400,600,3)"처럼 출력된다. (이미지 세로, 가로, 채널수)

# 네트워크 최종 결과물을 저장하는 경로 지정
sample_dir = './results/test/' #해당 경로 아래에 sample_dir이라는 디렉토리가 없으면 만들어준다. (맨밑줄코드에 또 나온다) 
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)

print("Start evaluating!") 
start_time = time.time()
for idx in range(len(eval_low_data)):
    print(idx+1) #사진이름(갯수)대로 출력하기위해 수정했습니다
    name = eval_img_name[idx]
    input_low = eval_low_data[idx]
    input_low_eval = np.expand_dims(input_low, axis=0) # 차원을 늘리는 함수, axis=0이면 행을 늘린다.
    h, w, _ = input_low.shape
### input_low_eval가 입력인 Decomposition-Net을 거친 output=> decom_r_low, decom_i_low
    decom_r_low, decom_i_low = sess.run([decom_output_R, decom_output_I], feed_dict={input_decom: input_low_eval}) # 세션을 생성할때 feed_dict의 키워드 형태로 텐서(데이터)를 placeholder와 맵핑할 수 있다. //참고 https://gdyoon.tistory.com/5
### Decom-Net 거친 결과값을 Restoration_net의 입력으로 넣음     
    restoration_r = sess.run(output_r, feed_dict={input_low_r: decom_r_low, input_low_i: decom_i_low})
### change the ratio to get different exposure level, the value can be 0-5.0
    ratio = 5.0 # 이 부분을 조절하여 조도를 조정할 수 있다 (0~5)
    i_low_data_ratio = np.ones([h, w])*(ratio) #모든 원소가 1인 (h*w)사이즈 행렬에 ratio값을 곱해서 필터를 만들어준다. => i_low_data_ratio
    i_low_ratio_expand = np.expand_dims(i_low_data_ratio , axis =2) #★행렬 형상 변환 
    i_low_ratio_expand2 = np.expand_dims(i_low_ratio_expand, axis=0)
    adjust_i = sess.run(output_i, feed_dict={input_low_i: decom_i_low, input_low_i_ratio: i_low_ratio_expand2})

    #The restoration result can find more details from very dark regions, however, it will restore the very dark regions
    #with gray colors, we use the following operator to alleviate this weakness.  
    decom_r_sq = np.squeeze(decom_r_low) #행렬 차원을 줄임
    r_gray = color.rgb2gray(decom_r_sq) # 휘도는 유지하고 색상과 채도 정보는 제거하여 RGB 이미지를 회색조(GrayScale)로 변환
    r_gray_gaussion = filters.gaussian(r_gray, 3) #가우시안필터 적용 => 노이즈 없애기 위함
    low_i =  np.minimum((r_gray_gaussion*2)**0.5,1)
    low_i_expand_0 = np.expand_dims(low_i, axis = 0) ##
    low_i_expand_3 = np.expand_dims(low_i_expand_0, axis = 3) #다시 3차원 행렬로 확장
    result_denoise = restoration_r*low_i_expand_3
    fusion4 = result_denoise*adjust_i
    
    #fusion = restoration_r*adjust_i
    # fusion with the original input to avoid over-exposure
    fusion2 = decom_i_low*input_low_eval + (1-decom_i_low)*fusion4
    #print(fusion2.shape)
    save_images(os.path.join(sample_dir, '%s_kindle.png' % (name)), fusion2)
    
