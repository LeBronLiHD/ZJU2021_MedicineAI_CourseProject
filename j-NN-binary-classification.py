# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import parameters
import load_data
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
import os
from sklearn.model_selection import train_test_split
import preprocess
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import random
import time
import model_analysis

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
matplotlib.use('agg')


def high_dimension_exp(data):
    print("before reshaping ->", np.shape(data))
    print(type(data))
    data_expand = []
    for i in range(np.shape(data)[0]):
        data_unit = []
        for j in range(parameters.COLUMNS):
            data_row = []
            for k in range(parameters.COLUMNS):
                if k == 0 or j == 0 or k == parameters.COLUMNS - 1 or j == parameters.COLUMNS - 1:
                    data_row.append(0)
                    continue
                if (j - 1) * (parameters.COLUMNS - 2) + (k - 1) < data.shape[1]:
                    data_row.append(data[i][(j - 1) * (parameters.COLUMNS - 2) + (k - 1)])
                else:
                    data_row.append(0)
            data_unit.append(data_row)
        data_expand.append(data_unit)
    data_expand = tf.expand_dims(data_expand, 3)
    print("after reshaping ->", np.shape(data_expand))
    return data_expand


def vertify_model(test, expect, model, total=10):
    print("test.shape ->", np.shape(test))
    print("expect.shape ->", np.shape(expect))
    test_img = high_dimension_exp(test)
    print("test_img.shape ->", np.shape(test_img))

    ones = []
    for i in range(len(expect)):
        if expect[i] == 1:
            ones.append(i)
    print("ones.length ->", len(ones))

    select_test = []
    select_test_img = []
    select_expect = []
    final_expect = []
    size = len(expect)
    one, zero = 0, 0
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
        final_expect.append(select_expect[i])
    print("expect.argmax ->", final_expect)

    count_vertify = 0
    for i in range(len(select_test)):
        data_piece = select_test[i]
        data_piece_img = select_test_img[i]
        print(np.shape(data_piece))
        output = model.predict(data_piece)
        print("output ->", output)
        print("count ->", i, "   \t-> ", output.argmax())
        plt.figure()
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



def NN(X_train, Y_train, X_test, Y_test, mode=4):
    init_time = time.time()
    X_train, Y_train = preprocess.un_balance(X_train, Y_train)
    # X_train, X_test = preprocess.data_normalization(X_train), preprocess.data_normalization(X_test)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    Y_train, Y_test = to_categorical(Y_train, num_classes=3), to_categorical(Y_test, num_classes=3)
    print("X_train shape ->", np.shape(X_train))
    print("Y_train shape ->", np.shape(Y_train))
    print("X_test shape ->", np.shape(X_test))
    print("Y_test shape ->", np.shape(Y_test))
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    print(type(X_train))
    print(type(X_test))

    # building a linear stack of layers with the sequential model
    model = Sequential()
    model.add(Dropout(0.2))
    model.add(Dense(512, input_shape=(62,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # training the model and saving metrics in history
    epoch_number = parameters.EPOCH_NUM
    history = model.fit(X_train, Y_train,
                        batch_size=32, epochs=epoch_number,
                        verbose=1,
                        validation_data=(X_test, Y_test),
                        shuffle=True)

    # plotting the metrics
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

    data_size = len(Y_train)
    print("train length ->", data_size)
    print("X_test size ->", len(X_test))
    print("Y_test size ->", len(Y_test))
    score = model.evaluate(X_test, Y_test)
    print('acc ->', score[1])
    # saving the model
    save_dir = parameters.MODEL_SAVE
    model_name = "model_simple_" + str(epoch_number) + "_" + str(score[1]) + ".h5"
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Saved trained model at %s ' % model_path)
    print("NN done. time ->", time.time() - init_time)
    model_analysis.Model_List_1_time[mode] = time.time() - init_time
    vertify_model(X_test, Y_test, model)


if __name__ == '__main__':
    path = parameters.DATA_PATH
    test = False
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path,
                                                                                                       test_mode=test)
    NN(merge_feature, merge_target, end_off_feature, end_off_target, mode=4)
