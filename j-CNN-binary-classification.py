# -*- coding: utf-8 -*-

import load_data
import parameters
import liner_regression_ols
import preprocess
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
import model_analysis

sys.dont_write_bytecode = True

matplotlib.use('TkAgg')


def vertify_model(test, test_img, expect, total=10):
    print("test.shape ->", np.shape(test))
    print("test_img.shape ->", np.shape(test_img))
    print("expect.shape ->", np.shape(expect))
    # sort by last modified time
    model_lists = os.listdir(parameters.MODEL_SAVE)
    model_lists = sorted(model_lists,
                         key=lambda files: os.path.getmtime(os.path.join(parameters.MODEL_SAVE, files)),
                         reverse=False)
    model_path_vertify = ""
    for modelLists in os.listdir(parameters.MODEL_SAVE):
        model_path_vertify = os.path.join(parameters.MODEL_SAVE, modelLists)
        print(model_path_vertify)

    if model_path_vertify == "":  # if the pwd is NULL
        print("No model saved!")
        exit()

    model = load_model(model_path_vertify)
    print("model loaded!")

    ones = []
    for i in range(len(expect)):
        if expect[i][1] == 1:
            ones.append(i)
    print("ones.length ->", len(ones))

    select_test = []
    select_test_img = []
    select_expect = []
    final_expect = []
    one, zero = 0, 0
    size = len(expect)
    for i in range(total):
        if one > zero:
            index = random.randrange(size)
            while np.argmax(expect[index]) == 1:
                index = random.randrange(size)
            select_test.append(test[index])
            select_expect.append(expect[index])
            select_test_img.append(test_img[index])
        elif zero > one:
            index = random.randrange(size)
            while np.argmax(expect[index]) == 0:
                index = random.randrange(size)
            select_test.append(test[index])
            select_expect.append(expect[index])
            select_test_img.append(test_img[index])
        else:
            index = random.randrange(size)
            select_test.append(test[index])
            select_expect.append(expect[index])
            select_test_img.append(test_img[index])
        if np.argmax(select_expect[i]) == 0:
            zero += 1
        else:
            one += 1
        final_expect.append(np.argmax(select_expect[i]))
    print("expect.argmax ->", final_expect)

    count_vertify = 0
    for i in range(len(select_test)):
        data_piece = select_test[i]
        data_piece_img = select_test_img[i]
        data_piece = np.expand_dims(data_piece, 0)  # 扩展至四维
        print(np.shape(data_piece))
        output = model.predict(data_piece)
        print("output ->", output)
        print("count ->", i, "   \t-> ", output.argmax())
        plt.figure("Test Image")
        for_show = []
        for k in range(np.shape(select_test_img)[1]):
            for_show_unit = []
            for j in range(np.shape(select_test_img)[2]):
                for_show_unit.append(data_piece_img[k][j][0])
            for_show.append(for_show_unit)
        plt.imshow(for_show)
        plt.axis("off")  # 关掉坐标轴为 off
        count_vertify += 1
        title = "CNN -> " + str(output.argmax()) + " | Exp -> " + str(final_expect[i])
        plt.title(title)  # 图像题目
        plt.show()


def CNN(X_train, Y_train, X_test, Y_test, mode=5, big=False, exp=False, ver=False):
    X_train, Y_train = preprocess.un_balance(X_train, Y_train)
    # X_train, X_test = preprocess.data_normalization(X_train), preprocess.data_normalization(X_test)
    X_test_img = preprocess.data_normalization(X_test)
    if exp:
        if big:
            X_train, X_test, X_test_img = preprocess.high_dimension_big_exp(X_train), \
                                          preprocess.high_dimension_big_exp(X_test), \
                                          preprocess.high_dimension_big_exp(X_test_img)
        else:
            X_train, X_test, X_test_img= preprocess.high_dimension_exp(X_train), \
                                         preprocess.high_dimension_exp(X_test),\
                                         preprocess.high_dimension_exp(X_test_img)
    else:
        if big:
            X_train, X_test, X_test_img = preprocess.high_dimension_big(X_train), \
                                          preprocess.high_dimension_big(X_test), \
                                          preprocess.high_dimension_big(X_test_img)
        else:
            X_train, X_test, X_test_img= preprocess.high_dimension(X_train), \
                                         preprocess.high_dimension(X_test),\
                                         preprocess.high_dimension(X_test_img)
    width, height = np.shape(X_train)[1], np.shape(X_train)[2]
    print("width =", width, "  height =", height)

    Y_test_list, Y_train_list = np.array(Y_test), np.array(Y_train)
    Y_test_list, Y_train_list = to_categorical(Y_test_list, num_classes=3), to_categorical(Y_train_list, num_classes=3)
    init_time = time.time()
    TrainCnnModel(np.array(X_train), Y_train_list, width, height, np.array(X_test), Y_test_list, big=big, exp=exp)
    print("CNN done, time ->", time.time() - init_time)
    model_analysis.Model_List_1_time[mode] = time.time() - init_time
    if ver:
        print("ver ->", ver)
        vertify_model(np.array(X_test), np.array(X_test_img), Y_test_list, total=10)
    else:
        print("ver ->", ver)


def TrainCnnModel(x_train, y_train, width, height, x_test, y_test, big=False, exp=False):
    i = parameters.EPOCH_NUM  # epoch number
    # 2. 定义模型结构
    # 迭代次数：第一次设置为30，后为了优化训练效果更改为100，后改为50
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(2, 2), padding='same',
                     input_shape=(width, height, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=25, mode='max')
    print("x_train.shape ->", np.shape(x_train))
    print("y_train.shape ->", np.shape(y_train))
    history = model.fit(x_train, y_train, batch_size=32, epochs=i, verbose=1,
                        callbacks=[early_stopping], validation_data=(x_test, y_test), shuffle=True)

    # 4. 训练
    # 绘制训练 & 验证的准确率值
    history.history['accuracy'][0] = min(1.0, history.history['accuracy'][0])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # 绘制训练 & 验证的损失值
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
    save_dir = parameters.MODEL_SAVE
    if exp:
        if big:
            model_name = "model_cnn_big_exp_" + str(i) + "_" + str(score[1]) + ".h5"
        else:
            model_name = "model_cnn_exp_" + str(i) + "_" + str(score[1]) + ".h5"
    else:
        if big:
            model_name = "model_cnn_big_" + str(i) + "_" + str(score[1]) + ".h5"
        else:
            model_name = "model_cnn_" + str(i) + "_" + str(score[1]) + ".h5"
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Saved trained model at %s ' % model_path)
    print("train model done!")


if __name__ == '__main__':
    path = parameters.DATA_PATH
    test = False
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path,
                                                                                                       test_mode=test)
    # CNN(merge_feature, merge_target, end_off_feature, end_off_target, big=False, exp=False, ver=False)
    CNN(merge_feature, merge_target, end_off_feature, end_off_target, mode=5, big=False, exp=True, ver=True)
    # CNN(merge_feature, merge_target, end_off_feature, end_off_target, big=True, exp=False, ver=False)
    # CNN(merge_feature, merge_target, end_off_feature, end_off_target, big=True, exp=True, ver=False)
