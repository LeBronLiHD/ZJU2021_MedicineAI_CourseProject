# -*- coding: utf-8 -*-

import numpy as np
import f_parameters
import f_preprocess
import f_load_data
from keras.callbacks import EarlyStopping
from imutils import paths
from keras.models import Sequential, load_model
from tensorflow.keras import regularizers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
import os
import matplotlib
import matplotlib.pyplot as plt
import f_model_analysis
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.keras.utils import to_categorical
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras import Input
from keras import Model


def data_normalization(data):
    data_nor = (data - np.mean(data))/np.std(data)
    return data_nor


def cal_auc(pred_proba, y_test):
    fpr, tpr, _ = roc_curve(y_test, pred_proba)
    roc_auc = roc_auc_score(y_test, pred_proba)
    return roc_auc


def TrainAEModel(x_train, y_train, x_test, y_test):
    input_img = Input(shape=(f_parameters.DIM_NUM,))
    encoded = Dense(32, activation='relu')(input_img)
    # encoded = Dense(256, activation='relu')(encoded)
    # encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    encoded = Dense(8, activation='relu')(encoded)

    decoded = Dense(16, activation='relu')(encoded)
    # decoded = Dense(128, activation='relu')(decoded)
    # decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(32, activation='relu')(decoded)
    decoded = Dense(f_parameters.DIM_NUM, activation='sigmoid')(decoded)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
    epoch_i = f_parameters.EPOCH_AE

    print("epoch ->", epoch_i)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, mode='min')
    history = autoencoder.fit(x_train, x_train, batch_size=8, epochs=epoch_i, verbose=1,
                              callbacks=[early_stopping],
                              # validation_split=0.1,
                              validation_data=(x_test, x_test),
                              shuffle=True)
    print(autoencoder.summary())

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

    score = autoencoder.evaluate(x_test, x_test)
    print('model accuracy ->', score)
    # saving the model
    save_dir = f_parameters.MODEL_SAVE
    model_name = "model_ae_" + str(epoch_i) + "_" + str(score) + ".h5"
    model_path = os.path.join(save_dir, model_name)
    autoencoder.save(model_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Saved trained model at %s ' % model_path)
    print("train model done!")


def AE(X_train, Y_train, X_test, Y_test, x_pd_train, x_pd_test, im_balance=True, train=True, ver=True):
    if im_balance:
        X_train, Y_train = f_preprocess.un_balance(X_train, Y_train, ratio="minority")
        X_test, Y_test = f_preprocess.un_balance(X_test, Y_test, ratio="minority")
    X_test = np.array(X_test)
    X_train = np.array(X_train)
    X_train, X_test = data_normalization(X_train), data_normalization(X_test)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    print("X_train type ->", type(X_train))
    print("X_test type ->", type(X_test))
    print("X_train shape ->", X_train.shape)
    print("X_test shape ->", X_test.shape)
    if train:
        TrainAEModel(X_train, Y_train, X_test, Y_test)


if __name__ == '__main__':
    path = f_parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = \
        f_load_data.f_load_data(path, test_mode=False)
    AE(merge_feature, merge_target, end_off_feature, end_off_target, merge_feature, end_off_feature,
       train=True, ver=True)
