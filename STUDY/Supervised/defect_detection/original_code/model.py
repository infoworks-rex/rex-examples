# -*- coding: utf8 -*-
import os
import numpy as np
import tensorflow as tf
from config import  *
from tensorflow.python.ops import embedding_ops
from tensorflow.python.layers.core import Dense
from tensorflow.contrib import rnn
from tensorflow.python.framework import graph_util
from tensorflow.python.ops import math_ops


class Model(object):
    def __init__(self,sess,param):
        self.step = 0
        self.__session = sess
        self.is_training=True
        self.__learn_rate = param["learn_rate"]
        self.__learn_rate=param["learn_rate"]
        self.__max_to_keep=param["max_to_keep"]
        self.__checkPoint_dir = param["checkPoint_dir"]
        self.__restore = param["b_restore"]
        self.__mode= param["mode"]
        self.is_training=True
        self.__batch_size = param["batch_size"]
        if  self.__mode is "savaPb" :
            self.__batch_size = 1

        # Building graph
        with self.__session.as_default():
            self.build_model()

        #파라미터 초기화, 또는 파라미터 읽기
        with self.__session.as_default():
            self.init_op.run()
            self.__saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.__max_to_keep)
            tf.io.write_graph(sess.graph, './','final1.pbtxt')
            
            logs_path = './logs'
            summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())


            # Loading last save if needed
            if self.__restore:
                ckpt = tf.train.latest_checkpoint(self.__checkPoint_dir)
                
                if ckpt:
                    self.step = int(ckpt.split('-')[1])
                    self.__saver.restore(self.__session, ckpt)
#                    print('Restoring from epoch:{}'.format( self.step))
#                    self.step+=1
#                    tf.io.write_graph(sess.graph, './','segment.pbtxt')

#            logs_path = './logs'
#            summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
#            print("Run the command line:\n" \
#                  "--> tensorboard --logdir=./logs " \
#                  "\nThen open http://0.0.0.0:6006/ into your web browser")


    def build_model(self):
        def SegmentNet(input, scope, is_training, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
              L1 = tf.layers.conv2d(input, 32, 5, padding='same')
              L1 = tf.layers.batch_normalization(L1, training=is_training) 
              L1 = tf.nn.relu(L1)

              L2 = tf.layers.conv2d(L1, 32, 5, padding='same')
              L2 = tf.layers.batch_normalization(L2, training=is_training)
              L2 = tf.nn.relu(L2)
              L2 = tf.layers.max_pooling2d(L2, 2, 2)

              L3 = tf.layers.conv2d(L2, 64, 5, padding='same')
              L3 = tf.layers.batch_normalization(L3, training=is_training)
              L3 = tf.nn.relu(L3)

              L4 = tf.layers.conv2d(L3, 64, 5, padding='same')
              L4 = tf.layers.batch_normalization(L4, training=is_training)
              L4 = tf.nn.relu(L4)

              L5 = tf.layers.conv2d(L4, 64, 5, padding='same')
              L5 = tf.layers.batch_normalization(L5, training=is_training)
              L5 = tf.nn.relu(L5)
              L5 = tf.layers.max_pooling2d(L5, 2, 2)

              L6 = tf.layers.conv2d(L5, 64, 5, padding='same')
              L6 = tf.layers.batch_normalization(L6, training=is_training)
              L6 = tf.nn.relu(L6)

              L7 = tf.layers.conv2d(L6, 64, 5, padding='same')
              L7 = tf.layers.batch_normalization(L7, training=is_training)
              L7 = tf.nn.relu(L7)

              L8 = tf.layers.conv2d(L7, 64, 5, padding='same')
              L8 = tf.layers.batch_normalization(L8, training=is_training)
              L8 = tf.nn.relu(L8)

              L9 = tf.layers.conv2d(L8, 64, 5, padding='same')
              L9 = tf.layers.batch_normalization(L9, training=is_training)
              L9 = tf.nn.relu(L9)
              L9 = tf.layers.max_pooling2d(L9, 2, 2)

              L10 = tf.layers.conv2d(L9, 1024, 15, padding='same')
              L10 = tf.layers.batch_normalization(L10, training=is_training)
              L10 = tf.nn.relu(L10)

              features = L10

              L11 = tf.layers.conv2d(L10, 1, 1, padding='same')
              net = tf.layers.batch_normalization(L11, training=is_training)

              logits_pixel=net

              net=tf.sigmoid(net, name=None)
              mask=net
              
              
              # features: Decision 네트워크로 전달할 1024채널
              # logits_pixel: features에 conv 1X1를 하여 만든 1채널
              # mask: logits_pixel에 sigmoid 함수 쓴 후 DecisionNet에 전달
              return features,logits_pixel,mask

        def DecisionNet(feature, mask, scope, is_training,num_classes=2, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
              net = tf.concat([feature, mask], axis=3)
              net = tf.layers.max_pooling2d(net, 2, 2)
              L1 = tf.layers.conv2d(net, 8, 5, padding='same')
              L1 = tf.layers.batch_normalization(L1, training=is_training)
              L1 = tf.nn.relu(L1)
              L1 = tf.layers.max_pooling2d(L1, 2, 2)

              L2 = tf.layers.conv2d(L1, 16, 5, padding='same') 
              L2 = tf.layers.batch_normalization(L2, training=is_training)
              L2 = tf.nn.relu(L2)
              L2 = tf.layers.max_pooling2d(L2, 2, 2)

              L3 = tf.layers.conv2d(L2, 32, 5, padding='same')
              L3 = tf.layers.batch_normalization(L3, training=is_training)
              net = tf.nn.relu(L3)

              vector1=math_ops.reduce_mean(net,[1,2],name='pool4', keepdims=True)
              vector2=math_ops.reduce_max(net,[1,2],name='pool5', keepdims=True)
              vector3=math_ops.reduce_mean(mask,[1,2],name='pool6', keepdims=True)
              vector4=math_ops.reduce_max(mask,[1,2],name='pool7', keepdims=True)
              vector=tf.concat([vector1,vector2,vector3,vector4],axis=3)
              vector=tf.squeeze(vector,axis=[1,2])
              logits=tf.contrib.layers.fully_connected(vector, num_classes,activation_fn=None)
              output=tf.argmax(logits, axis=1)
              # logits은 확률, output은 0 또는 1 (defect = false or true)
              return  logits,output

        Image = tf.placeholder(tf.float32, shape=(self.__batch_size, 1280, 512, 1), name='Image')
        PixelLabel=tf.placeholder(tf.float32, shape=(self.__batch_size, 160, 64, 1), name='PixelLabel')#IMAGE_SIZE[0]/8,IMAGE_SIZE[1]/8, 1), name='PixelLabel')
        Label = tf.placeholder(tf.int32, shape=(self.__batch_size), name='Label')

        # 모델 output
        features, logits_pixel, mask=SegmentNet(Image, 'segment',  self.is_training)
        logits_class,output_class=DecisionNet(features, mask, 'decision',  self.is_training)
        print(mask)
        print(mask.shape)
        # loss function
        logits_pixel=tf.reshape(logits_pixel,[self.__batch_size,-1]) #sigmoid 적용 전 
        PixelLabel_reshape=tf.reshape(PixelLabel,[self.__batch_size,-1]) #ground truth
        #loss_pixel은 pixel loss, loss_class는 defect 존재 여부 loss
        loss_pixel = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_pixel, labels=PixelLabel_reshape))
        loss_class = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_class,labels=Label))
        loss_total=loss_pixel+loss_class
        optimizer = tf.train.GradientDescentOptimizer(self.__learn_rate)

        train_var_list = [v for v in tf.trainable_variables() ]
        train_segment_var_list = [v for v in tf.trainable_variables() if 'segment' in v.name ]
        train_decision_var_list = [v for v in tf.trainable_variables() if 'decision' in v.name]

        optimize_segment = optimizer.minimize(loss_pixel,var_list=train_segment_var_list)
        optimize_decision = optimizer.minimize(loss_class, var_list=train_decision_var_list)
        optimize_total = optimizer.minimize(loss_total, var_list=train_var_list)

        init_op=tf.global_variables_initializer()
        self.Image=Image
        self.PixelLabel = PixelLabel
        self.Label = Label
        self.features = features
        self.mask = mask
        self.logits_class=logits_class
        self.output_class=output_class
        self.loss_pixel = loss_pixel
        self.loss_class = loss_class
        self.loss_total = loss_total
        self.optimize_segment = optimize_segment
        self.optimize_decision = optimize_decision
        self.optimize_total = optimize_total
        self.init_op=init_op

    def save(self):
        self.__saver.save(
            self.__session,
            os.path.join(self.__checkPoint_dir, 'ckp'),
            global_step=self.step
            
        )

