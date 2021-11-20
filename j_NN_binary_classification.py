# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import f_parameters
import f_load_data
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import os
from sklearn.model_selection import train_test_split
import f_preprocess
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import random
import time
import f_model_analysis
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
matplotlib.use('TkAgg')


def data_normalization(data):
    data_nor = (data - np.mean(data))/np.std(data)
    return data_nor


def cal_auc(pred_proba, y_test):
    fpr, tpr, _ = roc_curve(y_test, pred_proba)
    roc_auc = roc_auc_score(y_test, pred_proba)
    return roc_auc


def high_dimension_exp(data):
    print("before reshaping ->", np.shape(data))
    print(type(data))
    data_expand = []
    for i in range(np.shape(data)[0]):
        data_unit = []
        for j in range(f_parameters.COLUMNS):
            data_row = []
            for k in range(f_parameters.COLUMNS):
                if k == 0 or j == 0 or k == f_parameters.COLUMNS - 1 or j == f_parameters.COLUMNS - 1:
                    data_row.append(0)
                    continue
                if (j - 1) * (f_parameters.COLUMNS - 2) + (k - 1) < data.shape[1]:
                    data_row.append(data[i][(j - 1) * (f_parameters.COLUMNS - 2) + (k - 1)])
                else:
                    data_row.append(0)
            data_unit.append(data_row)
        data_expand.append(data_unit)
    data_expand = tf.expand_dims(data_expand, 3)
    print("after reshaping ->", np.shape(data_expand))
    return data_expand


def vertify_model(test, expect, total=10, mode=4, nor_img=True):
    print("test.shape ->", np.shape(test))
    print("expect.shape ->", np.shape(expect))
    if nor_img:
        test_img = data_normalization(test)
        test_img = high_dimension_exp(test_img)
    else:
        test_img = high_dimension_exp(test)
    print("test_img.shape ->", np.shape(test_img))
    model_lists = os.listdir(f_parameters.MODEL_SAVE)
    model_lists = sorted(model_lists,
                         key=lambda files: os.path.getmtime(os.path.join(f_parameters.MODEL_SAVE, files)),
                         reverse=False)
    model_path_vertify = ""
    for modelLists in os.listdir(f_parameters.MODEL_SAVE):
        model_path_vertify = os.path.join(f_parameters.MODEL_SAVE, modelLists)
        print(model_path_vertify)

    if model_path_vertify == "":  # if the pwd is NULL
        print("No model saved!")
        exit()

    print("model_select ->", model_path_vertify)
    model = load_model(model_path_vertify)
    print("model loaded!")
    ones = []
    for i in range(len(expect)):
        if np.argmax(expect[i]) == 1:
            ones.append(i)
    print("ones.length ->", len(ones))

    right = 0
    right_0_1 = [0, 0]
    error_0_1 = [0, 0]
    impossible_r_e = [0, 0]
    test_pred = []
    test_test = []
    size = len(test)
    count = len(ones)
    for i in range(size):
        data_ana_piece = test[i]
        data_ana_piece = np.expand_dims(data_ana_piece, 0)
        output = model.predict(data_ana_piece)
        test_pred.append(output.argmax())
        test_test.append(np.argmax(expect[i]))
        if i % 200 == 0:
            print("model analysis ... i =", i)
        if output.argmax() == np.argmax(expect[i]):
            right += 1
            if np.argmax(expect[i]) == 0:
                right_0_1[0] += 1
            elif np.argmax(expect[i]) == 1:
                right_0_1[1] += 1
            else:
                impossible_r_e[0] += 1
        else:
            if np.argmax(expect[i]) == 0:
                error_0_1[0] += 1
            elif np.argmax(expect[i]) == 1:
                error_0_1[1] += 1
            else:
                impossible_r_e[1] += 1
        continue

    auc = cal_auc(test_pred, test_test)
    print("auc ->", auc)
    f_model_analysis.Model_List_1_auc[mode] = auc
    print("right =", right)
    print("fault =", size - right)
    print("overall right ratio =", right / size)
    print("0 right ratio =", right_0_1[0] / (size - count))
    print("1 right ratio =", right_0_1[1] / count)
    print("right_0_1 ->", right_0_1)
    print("error_0_1 ->", error_0_1)
    print("impossible_r_e ->", impossible_r_e)
    f_model_analysis.Model_List_1_right_0[mode] = right_0_1[0] / (size - count)
    f_model_analysis.Model_List_1_right_1[mode] = right_0_1[1] / count
    f_model_analysis.Model_List_1_right_all[mode] = right / size

    select_test = []
    select_test_img = []
    select_expect = []
    final_expect = []
    one, zero = 0, 0
    for i in range(total):
        if one > zero:
            index = random.randrange(size)
            while expect[index][0] == 1:
                index = random.randrange(size)
            select_test.append(test[index])
            select_expect.append(0)
            select_test_img.append(test_img[index])
        elif zero > one:
            index = random.randrange(size)
            while expect[index][0] == 0:
                index = random.randrange(size)
            select_test.append(test[index])
            select_expect.append(1)
            select_test_img.append(test_img[index])
        else:
            index = random.randrange(size)
            select_test.append(test[index])
            select_expect.append(expect[index][1])
            select_test_img.append(test_img[index])
        if select_expect[i] == 0:
            zero += 1
        else:
            one += 1
        final_expect.append(select_expect[i])
    print("expect.argmax ->", final_expect)

    count_vertify = 0
    for i in range(len(select_test)):
        data_piece = select_test[i]
        data_piece_img = select_test_img[i]
        data_piece = np.expand_dims(data_piece, 0)
        # print(np.shape(data_piece))
        output = model.predict(data_piece)
        print("output ->", output)
        print("count ->", i, "   \t-> ", output.argmax())
        plt.figure()
        for_show = []
        for k in range(np.shape(data_piece_img)[0]):
            for_show_unit = []
            for j in range(np.shape(data_piece_img)[1]):
                for_show_unit.append(data_piece_img[k][j][0])
            for_show.append(for_show_unit)
        plt.imshow(for_show)
        plt.axis("off")  # 关掉坐标轴为 off
        count_vertify += 1
        title = "CNN -> " + str(output.argmax()) + " | Exp -> " + str(final_expect[i])
        plt.title(title)  # 图像题目
        plt.show()


def NN(X_train, Y_train, X_t_test, Y_t_test, data, imbalance=False, mode=4):
    if imbalance:
        X_train, Y_train = f_preprocess.un_balance(X_train, Y_train, ratio="minority")
        X_test, Y_test = f_preprocess.un_balance(X_t_test, Y_t_test, ratio="minority")
    else:
        X_test, Y_test = X_t_test, Y_t_test
    # X_train, X_test = f_preprocess.data_normalization(X_train), f_preprocess.data_normalization(X_test)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    Y_train, Y_test = to_categorical(Y_train, num_classes=f_parameters.NN_NUM_CLASS), \
                      to_categorical(Y_test, num_classes=f_parameters.NN_NUM_CLASS)
    print("X_train shape ->", np.shape(X_train))
    print("Y_train shape ->", np.shape(Y_train))
    print("X_test shape ->", np.shape(X_test))
    print("Y_test shape ->", np.shape(Y_test))
    X_train = X_train.astype('float64')
    X_test = X_test.astype('float64')
    print(type(X_train))
    print(type(X_test))
    TrainNNModel(X_train, Y_train, X_test, Y_test, imbalance, mode=4)
    Y_t_test = to_categorical(Y_t_test, num_classes=f_parameters.NN_NUM_CLASS)
    vertify_model(np.array(X_t_test), np.array(Y_t_test), nor_img=True)


def get_activation():
    if f_parameters.NN_NUM_CLASS == 2:
        return "sigmoid"
    else:
        return "softmax"


def TrainNNModel(X_train, Y_train, X_test, Y_test, imbalance=False, mode=4):
    init_time = time.time()
    # building a linear stack of layers with the sequential model
    model = Sequential()
    model.add(Dropout(0.2))
    model.add(Dense(512, input_shape=(62,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(f_parameters.NN_NUM_CLASS))
    model.add(Activation(get_activation()))

    # compiling the sequential model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # training the model and saving metrics in history
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4, mode='min')
    epoch_number = f_parameters.EPOCH_NN_NUM
    count = 0
    for i in range(Y_train.shape[0]):
        if np.argmax(Y_train[i]) == 1:
            count += 1
    class_weigh = {0: (Y_train.shape[0] - count)/Y_train.shape[0], 1: count/Y_train.shape[0]}
    print(class_weigh)
    history = model.fit(X_train, Y_train,
                        batch_size=32, epochs=epoch_number,
                        verbose=1,
                        validation_split=0.1,
                        # validation_data=(X_test, Y_test),
                        # callbacks=[early_stopping],
                        class_weight=class_weigh,
                        shuffle=True)
    print(model.summary())

    # plotting the metrics
    history.history['accuracy'][0] = min(0.0, history.history['accuracy'][0])
    history.history['val_accuracy'][0] = min(0.0, history.history['val_accuracy'][0])
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # 绘制训练 & 验证的损失值
    history.history['loss'][0] = min(1.0, history.history['loss'][0])
    history.history['val_loss'][0] = min(1.0, history.history['val_loss'][0])
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
    save_dir = f_parameters.MODEL_SAVE
    model_name = "model_simple_" + str(epoch_number) + "_" + str(score[1]) + ".h5"
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Saved trained model at %s ' % model_path)
    print("NN done. time ->", time.time() - init_time)
    f_model_analysis.Model_List_1_time[mode] = time.time() - init_time


if __name__ == '__main__':
    path = f_parameters.DATA_PATH
    test = False
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = \
        f_load_data.f_load_data(path, test_mode=test)
    NN(merge_feature, merge_target, end_off_feature, end_off_target, end_off, imbalance=True, mode=4)
