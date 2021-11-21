# -*- coding: utf-8 -*-

"""
use CNN for regression
"""

import f_load_data
import f_parameters
import r_ols
import f_preprocess
from keras.callbacks import EarlyStopping
import filter
import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
from keras.callbacks import EarlyStopping
from imutils import paths
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
import tkinter
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import random
import sys
import time
import f_model_analysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve
import pandas
import itertools

sys.dont_write_bytecode = True

matplotlib.use('TkAgg')

def data_normalization(data):
    data_nor = (data - np.mean(data))/np.std(data)
    return data_nor


def get_activation():
    if f_parameters.NN_NUM_CLASS == 2:
        return "sigmoid"
    else:
        return "softmax"


def cal_auc(pred_proba, y_test):
    fpr, tpr, _ = roc_curve(y_test, pred_proba)
    roc_auc = roc_auc_score(y_test, pred_proba)
    return roc_auc


def CNN(x_train, y_train, x_test, y_test):
    y_train, y_test = to_categorical(y_train, num_classes=f_parameters.NN_NUM_CLASS), \
                      to_categorical(y_test, num_classes=f_parameters.NN_NUM_CLASS)
    width, height = np.shape(x_train)[1], np.shape(x_train)[2]
    print("width =", width, "  height =", height)
    x_train, x_test = tf.expand_dims(x_train, 3), tf.expand_dims(x_test, 3)
    print(np.shape(x_train))
    print(np.shape(x_test))
    print(np.shape(y_train))
    print(np.shape(y_test))
    TrainCNN(x_train, y_train, width, height, x_test, y_test)


def TrainCNN(x_train, y_train, width, height, x_test, y_test):
    epoch_i = f_parameters.EPOCH_R_CNN  # epoch number
    # 2. 定义模型结构
    # 迭代次数：第一次设置为30，后为了优化训练效果更改为100，后改为50
    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=(2, 2), padding='same',
                     input_shape=(width, height, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(f_parameters.NN_NUM_CLASS, activation=get_activation()))
    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
    # counts = np.bincount(y_train[:, 0])
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=50, mode='max')
    print("x_train.shape ->", np.shape(x_train))
    print("y_train.shape ->", np.shape(y_train))
    count = 0
    for i in range(y_train.shape[0]):
        if np.argmax(y_train[i]) == 1:
            count += 1
    class_weigh = {0: (y_train.shape[0] - count) / y_train.shape[0], 1: count / y_train.shape[0]}
    print("epoch ->", epoch_i)
    history = model.fit(x_train, y_train, batch_size=8, epochs=epoch_i, verbose=1,
                        # callbacks=[early_stopping],
                        validation_split=0.1,
                        # validation_data=(x_test, y_test),
                        class_weight=class_weigh,
                        shuffle=True)
    print(model.summary())

    # 4. 训练
    # 绘制训练 & 验证的准确率值
    if epoch_i > 5:
        repair = 5
    else:
        repair = epoch_i
    # for i in range(repair):
    #     history.history['accuracy'][i] = min(0.0, history.history['accuracy'][i])
    #     history.history['val_accuracy'][i] = min(0.0, history.history['val_accuracy'][i])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # 绘制训练 & 验证的损失值
    for i in range(repair):
        history.history['loss'][i] = min(1.5, history.history['loss'][i])
        history.history['val_loss'][i] = min(1.5, history.history['val_loss'][i])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    data_size = len(y_train)
    print("train length ->", data_size)
    print("X_test size ->", len(x_test))
    print("Y_test size ->", len(y_test))
    score = model.evaluate(x_test, y_test)
    print('acc ->', score[1])
    # saving the model
    save_dir = f_parameters.MODEL_SAVE
    model_name = "model_r_cnn_" + str(epoch_i) + "_" + str(score[1]) + ".h5"

    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Saved trained model at %s ' % model_path)
    print("train model done!")


if __name__ == '__main__':
    path = f_parameters.DATA_PATH
    test = False
    x_train, y_train, x_test, y_test = \
        f_load_data.f_load_data_predict(path, test_mode=False)
    x_train, y_train = f_preprocess.reshape_width_height(x_train, y_train)
    x_test, y_test = f_preprocess.reshape_width_height(x_test, y_test)
    CNN(x_train, y_train, x_test, y_test)
