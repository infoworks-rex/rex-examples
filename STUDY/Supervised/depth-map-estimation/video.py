# -*- coding: utf8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CIDA_VISIBLE_DEVICES"]="1"

import argparse
import torch
import torch.nn.parallel

import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_demo as loaddata
import pdb

import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")
import cv2

import pandas as pd
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from demo_transform import *

#deep learning network 선택
def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model

def main():
    #model 선택
    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    #cuda를 이용한 데이터 병렬 처리
    model = torch.nn.DataParallel(model).cuda()
    #이미 학습된 model을 가져옴
    model.load_state_dict(torch.load('./pretrained_model/model_senet'))
    #dropout 및 배치 정규화를 평가 모드로 설정해야 일관성 있는 출력이 가능
    model.eval()
    
    #torch정규화에 사용되는 평균, 표준편차 값
    __imagenet_stats={'mean':[0.485, 0.456, 0.406], 'std':[0.229, 0.224, 0.225]}

    #동영상을 frame으로 가져오기
    capture=cv2.VideoCapture('data/demo/input.avi')
    #동영사이 열려있다면 반복
    while(capture.isOpened()):
        #capture.read()로부터 입력여부와 frame 받아 옴
        ret, frame = capture.read()

        #입력이 false라면 while문 벗어남
        if not ret:
            break
        
        #PIL로 바꾸기
        cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv_img)        
        
        #size 줄이기
        resize_frame=torchvision.transforms.Scale([320,240])(pil_img)
        
        #이미지 crop하기
        crop_frame=torchvision.transforms.CenterCrop((228,304))(resize_frame) #h,w
        
        #이미지 텐서로 바꾸기
        to_tensor=torchvision.transforms.ToTensor()(crop_frame)
        
        #이미지 정규화하기
        normalized_img=torchvision.transforms.Normalize(__imagenet_stats['mean'], __imagenet_stats['std'])(to_tensor)
        
        #torch 배열 맞추기 위해 추가
        normalized_img = torch.unsqueeze(normalized_img, 0)
        
        #DataLoader는 torchvision의 함수
        nyu2_loader = DataLoader(normalized_img, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        
        for i, image in enumerate(nyu2_loader):       
            #tensor 연산에 자동 미분 제공, 기본 CUDA장치에 할당
            image2 = torch.autograd.Variable(normalized_img, volatile=True).cuda()
            out = model(image2)

            #tensor의 shape를 변경한 후 텐서를 numpy배열로 변환
            np_array = out.view(out.size(2),out.size(3)).data.cpu().numpy()

            #0~255의 값으로 바꾸어 줌
            cv2.normalize(np_array, np_array, 0, 255, cv2.NORM_MINMAX)
            
            #numpy 배열을 PIL 이미지로변환
            img = Image.fromarray(np.uint8(np_array), 'L')
                        
            #PIL 이미지를 numpy 배열로 변환
            pix = np.array(img)

            #openCV shape에 맞게 배열 변환 (그레이 버전이라 channel 1개 추가함)
            opencv_array = np.expand_dims(pix,2)
            
            #화면에 출력
            cv2.imshow('gray', opencv_array)
            if cv2.waitKey(1)>0: break

    capture.release()


if __name__ == '__main__':
    main()
