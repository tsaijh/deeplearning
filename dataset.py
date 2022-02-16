# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 11:05:49 2021

@author: chulab
"""
import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm


#讀圖
print('reading image')
data_raw=[]
data_mask=[]

datapath = input('INSERT THE DATA PATH :\n')

rawpath = datapath+'/'+'raw'
rawfilelist = os.listdir(rawpath)
for rawfile in tqdm(rawfilelist):
    img = cv2.imread(rawpath+'/'+rawfile,2)
    data_raw.append(img)

maskpath = datapath+'/'+'mask'
maskfilelist = os.listdir(maskpath)
for maskfile in tqdm(maskfilelist):
    img = cv2.imread(maskpath+'/'+maskfile,2)
    data_mask.append(img)


#切成適合模型訓練大小
print('cut')

size = 256
ilength = data_raw[0].shape[0]
jlength = data_raw[0].shape[1]

train_raw_list=[] 
train_mask_list=[]

for x in tqdm(range(len(data_raw))):
    for i in range(0,ilength-size,128):
        for j in range(0,jlength-size,128):
            train_mask_data = np.zeros((size,size),dtype='uint8')
            train_raw_data = np.zeros((size,size),dtype='uint16')
        
            train_mask_data = data_mask[x][i:i+size,j:j+size]
            train_raw_data = data_raw[x][i:i+size,j:j+size]
            
            train_mask_data = cv2.resize(train_mask_data,(128,128))
            train_raw_data = cv2.resize(train_raw_data,(128,128))
            train_raw_list.append(train_raw_data)
            train_mask_list.append(train_mask_data)


#存圖
print('saving image')

os.makedirs(datapath+'/'+'train')
savepath = datapath+'/'+'train'
os.makedirs(savepath+'/'+'train_raw')
os.makedirs(savepath+'/'+'train_mask')
saverawpath = savepath+'/'+'train_raw'
savemaskpath = savepath+'/'+'train_mask'
for l in tqdm(range(len(train_raw_list))):
    Image.fromarray(train_raw_list[l]).save(saverawpath+'/'+str(l)+'.tif')
    Image.fromarray(train_mask_list[l]).save(savemaskpath+'/'+str(l)+'.tif')