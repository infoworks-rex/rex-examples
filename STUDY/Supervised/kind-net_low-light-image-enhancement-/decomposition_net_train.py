# coding: utf-8
from __future__ import print_function
import os, time, random
import tensorflow as tf
from PIL import Image
import numpy as np
from utils import *
from model import *
from glob import glob

batch_size = 10 # 전체 데이터셋중에서 10개씩 나눠 실행 
patch_size = 48 # 패치사이즈 48*48

sess = tf.compat.v1.Session() #텐서플로우에서는 그래프를 생성하고 실행시켜줘야 한다. 생성한 그래프를 tf.Session()을 통해 run()할 수 있다. 

input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low') # input으로 쓰일 저조도&고조도 이미지 쌍을 placeholder에 담아준다.  
input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

[R_low, I_low] = DecomNet_simple(input_low) #해당 함수는 "model.py" 파일에 있다. 
[R_high, I_high] = DecomNet_simple(input_high) 

I_low_3 = tf.concat([I_low, I_low, I_low], axis=3) # axis 인덱스에 맞게 데이터를 표처럼 옆으로 concat(연결)한다. #axis=3 
I_high_3 = tf.concat([I_high, I_high, I_high], axis=3) #왜 동일한 이미지 3개를 연결하는지? 

#network output
output_R_low = R_low
output_R_high = R_high
output_I_low = I_low_3
output_I_high = I_high_3

# define loss

def mutual_i_loss(input_I_low, input_I_high): #low이미지와 high이미지의 차이를 계산하는 손실함수
    low_gradient_x = gradient(input_I_low, "x")
    high_gradient_x = gradient(input_I_high, "x")
    x_loss = (low_gradient_x + high_gradient_x)* tf.exp(-10*(low_gradient_x+high_gradient_x)) #tf.exp()는 지수를 계산해주는 함수
    low_gradient_y = gradient(input_I_low, "y")
    high_gradient_y = gradient(input_I_high, "y")
    y_loss = (low_gradient_y + high_gradient_y) * tf.exp(-10*(low_gradient_y+high_gradient_y))
    mutual_loss = tf.reduce_mean( x_loss + y_loss) 
    return mutual_loss

def mutual_i_input_loss(input_I_low, input_im): 
    input_gray = tf.image.rgb_to_grayscale(input_im) #연산량을 줄이기위해 채널 수를 3->1로 줄인다
    low_gradient_x = gradient(input_I_low, "x")
    input_gradient_x = gradient(input_gray, "x")
    x_loss = tf.abs(tf.div(low_gradient_x, tf.maximum(input_gradient_x, 0.01))) #abs()는 절댓값을 만들어주는 함수, div()는 차이를 계산하는 함수이다. 즉 가중치 차의값의 양수화를 해주는 것이다.
    low_gradient_y = gradient(input_I_low, "y")
    input_gradient_y = gradient(input_gray, "y")
    y_loss = tf.abs(tf.div(low_gradient_y, tf.maximum(input_gradient_y, 0.01))) #
    mut_loss = tf.reduce_mean(x_loss + y_loss) 
    return mut_loss

recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3 -  input_low))
recon_loss_high = tf.reduce_mean(tf.abs(R_high * I_high_3 - input_high))

equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_high))

i_mutual_loss = mutual_i_loss(I_low, I_high)

i_input_mutual_loss_high = mutual_i_input_loss(I_high, input_high)
i_input_mutual_loss_low = mutual_i_input_loss(I_low, input_low)

loss_Decom = 1*recon_loss_high + 1*recon_loss_low \
               + 0.01 * equal_R_loss + 0.2*i_mutual_loss \
             + 0.15* i_input_mutual_loss_high + 0.15* i_input_mutual_loss_low

###
lr = tf.placeholder(tf.float32, name='learning_rate')

optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')
var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]

train_op_Decom = optimizer.minimize(loss_Decom, var_list = var_Decom)
sess.run(tf.global_variables_initializer()) # 변수 전체를 초기화 해주는 함수 

saver_Decom = tf.train.Saver(var_list = var_Decom)
print("[*] Initialize model successfully...")

#load data
###train_data
train_low_data = []
train_high_data = []
train_low_data_names = glob('./LOLdataset/our485/low/*.png') #학습용 input 저조도 이미지 불러오고 (다음 라인에서)정렬
train_low_data_names.sort()
train_high_data_names = glob('./LOLdataset/our485/high/*.png') #학습용 input 고조도 이미지 불러오고 (다음 라인에서)정렬
train_high_data_names.sort()
assert len(train_low_data_names) == len(train_high_data_names) # 이미지 쌍이 입력된걸 확인한 후 학습함
print('[*] Number of training data: %d' % len(train_low_data_names))
for idx in range(len(train_low_data_names)): #입력이미지 개수만큼 반복한다. (입력이미지들 파일명이 모두 숫자이기때문에 파일명만큼 반복)
    low_im = load_images(train_low_data_names[idx])
    train_low_data.append(low_im)
    high_im = load_images(train_high_data_names[idx])
    train_high_data.append(high_im)
###eval_data
eval_low_data = []
eval_high_data = []
eval_low_data_name = glob('./LOLdataset/eval15/low/*.png') #평가용 input 저조도 이미지 불러오고 (다음 라인에서)정렬
eval_low_data_name.sort()
eval_high_data_name = glob('./LOLdataset/eval15/high/*.png*') #평가용 input 저조도 이미지 불러오고 (다음 라인에서)정렬
eval_high_data_name.sort()
for idx in range(len(eval_low_data_name)):
    eval_low_im = load_images(eval_low_data_name[idx])
    eval_low_data.append(eval_low_im)
    eval_high_im = load_images(eval_high_data_name[idx])
    eval_high_data.append(eval_high_im)


epoch = 2000
learning_rate = 0.0001

sample_dir = './Decom_net_train/'
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)

eval_every_epoch = 200
train_phase = 'decomposition'
numBatch = len(train_low_data) // int(batch_size)
train_op = train_op_Decom
train_loss = loss_Decom
saver = saver_Decom

checkpoint_dir = './checkpoint/decom_net_train/'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

start_step = 0
start_epoch = 0
iter_num = 0
print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

start_time = time.time()
image_id = 0
for epoch in range(start_epoch, epoch):
    for batch_id in range(start_step, numBatch):
        batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        for patch_id in range(batch_size):
            h, w, _ = train_low_data[image_id].shape
            x = random.randint(0, h - patch_size)
            y = random.randint(0, w - patch_size)
            rand_mode = random.randint(0, 7)
            batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
            batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
            image_id = (image_id + 1) % len(train_low_data)
            if image_id == 0:
                tmp = list(zip(train_low_data, train_high_data))
                random.shuffle(tmp)
                train_low_data, train_high_data  = zip(*tmp)

        _, loss = sess.run([train_op, train_loss], feed_dict={input_low: batch_input_low, \
                                                              input_high: batch_input_high, \
                                                              lr: learning_rate})
        print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
              % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
        iter_num += 1
    if (epoch + 1) % eval_every_epoch == 0:
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch + 1))
        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0) #차원 확장시켜줌
            result_1, result_2 = sess.run([output_R_low, output_I_low], feed_dict={input_low: input_low_eval})
            save_images(os.path.join(sample_dir, 'low_%d_%d.png' % ( idx + 1, epoch + 1)), result_1, result_2)
        for idx in range(len(eval_high_data)):
            input_high_eval = np.expand_dims(eval_high_data[idx], axis=0)
            result_11, result_22 = sess.run([output_R_high, output_I_high], feed_dict={input_high: input_high_eval})
            save_images(os.path.join(sample_dir, 'high_%d_%d.png' % ( idx + 1, epoch + 1)), result_11, result_22)
         
    saver.save(sess, checkpoint_dir + 'model.ckpt')

print("[*] Finish training for phase %s." % train_phase)
