# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 12:38:51 2021

@author: chulab
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Lambda, Input, Conv2D, Conv2DTranspose, BatchNormalization, MaxPooling2D, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#資料前處理
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1

datapath = input('INSERT THE DATA PATH :\n')
trainrawpath = datapath+'/'+'train'+'/'+'train_raw'

image_ids = next(os.walk(trainrawpath))[2]
LENGTH = len(image_ids)

X = np.zeros((LENGTH, IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS))
Y = np.zeros((LENGTH, IMG_HEIGHT, IMG_WIDTH,1), dtype=np.bool)


rawtrainfilelist = os.listdir(trainrawpath)
number = 0
for rawtrainfile in tqdm(rawtrainfilelist):
    img = cv2.imread(trainrawpath+'/'+rawtrainfile,2)
    img = cv2.resize(img, (128,128))
    img = np.expand_dims(img, axis=2)
    X[number] = img
    number += 1

trainmaskpath = datapath+'/'+'train'+'/'+'train_mask'
masktrainfilelist = os.listdir(trainmaskpath)
number = 0
for masktrainfile in tqdm(masktrainfilelist):
    img = cv2.imread(trainmaskpath+'/'+masktrainfile,2)
    img = cv2.resize(img, (128,128))
    img = np.expand_dims(img, axis=2)
    Y[number] = img
    number += 1


x_train=X
y_train=Y


# 創建 U-Net 模型架構
inputs = Input((IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS))
s = Lambda(lambda x: x / 1) (inputs)

c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (s)
c1 = BatchNormalization()(c1)
c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c1)
c1 = BatchNormalization()(c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p1)
c2 = BatchNormalization()(c2)
c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c2)
c2 = BatchNormalization()(c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p2)
c3 = BatchNormalization()(c3)
c3 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c3)
c3 = BatchNormalization()(c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p3)
c4 = BatchNormalization()(c4)
c4 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c4)
c4 = BatchNormalization()(c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (p4)
c5 = BatchNormalization()(c5)
c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c5)
c5 = BatchNormalization()(c5)

u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u6)
c6 = BatchNormalization()(c6)
c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c6)
c6 = BatchNormalization()(c6)

u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u7)
c7 = BatchNormalization()(c7)
c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c7)
c7 = BatchNormalization()(c7)

u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u8)
c8 = BatchNormalization()(c8)
c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c8)
c8 = BatchNormalization()(c8)

u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (u9)
c9 = BatchNormalization()(c9)
c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (c9)
c9 = BatchNormalization()(c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)


model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


#訓練
earlystopper = EarlyStopping(patience=10, verbose=1)
checkpointer = ModelCheckpoint(datapath+'/'+'model.h5', verbose=1, save_best_only=True)
results = model.fit(x_train, y_train, validation_split=0.2, batch_size=10, epochs=60, callbacks=[earlystopper, checkpointer]) 


#評估函式定義
def iou_pred(target,prediction):
    intersection = np.logical_and(target,prediction)
    union = np.logical_or(target,prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score



def dice_coef(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    both_sum = (np.logical_not(y_true == 0)).sum()+(np.logical_not(y_pred == 0)).sum()
    return 2. * intersection.sum() / both_sum


#載入模型並測試
model = load_model(datapath+'/'+'model.h5')
model.summary

test_raw_path = datapath+'/'+'test'+'/'+'raw'
test_mask_path = datapath+'/'+'test'+'/'+'mask'

rawtestfilelist = os.listdir(test_raw_path)
masktestfilelist = os.listdir(test_mask_path)

n = 0
for rawtest in rawtestfilelist:
    rawtest = cv2.resize(rawtest, (128,128))
    predict = model.predict(rawtest, verbose=1)
    prediction = (predict > 0.5).astype(np.uint8)
    prediction2 = np.logical_not(prediction == 0)
    mask_test = cv2.resize(cv2.imread(test_mask_path+'/'+masktestfilelist[n]), (128,128))
    print('iou_pred : '+iou_pred(prediction2,mask_test))
    print('dice_coef : '+dice_coef(prediction2,mask_test))
    n += 1
