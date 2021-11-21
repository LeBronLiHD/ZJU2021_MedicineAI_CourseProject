# -*- coding: utf-8 -*-

"""
generate more '1' data in Convolutional auto-encoder
"""

import numpy as np
import f_parameters
import f_preprocess
import f_load_data
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense
import os
import matplotlib
import matplotlib.pyplot as plt
import f_model_analysis
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras import Input
from keras import Model


def data_normalization(data):
    data_nor = (data - np.mean(data))/np.std(data)
    return data_nor


def cal_auc(pred_proba, y_test):
    fpr, tpr, _ = roc_curve(y_test, pred_proba)
    roc_auc = roc_auc_score(y_test, pred_proba)
    return roc_auc


def TrainCAEModel(x_train, y_train, x_test, y_test):
    width = np.shape(x_train)[1]
    height = np.shape(x_train)[2]

    input_img = Input(shape=(width, height, 1))

    x = Conv2D(16, (2, 2), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (2, 2), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, (2, 2), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (2, 2), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (2, 2), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    epoch_i = f_parameters.EPOCH_AE

    print("epoch ->", epoch_i)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, mode='min')
    history = autoencoder.fit(x_train, x_train, batch_size=64, epochs=epoch_i, verbose=1,
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
    print('model accuracy ->', score[1])
    # saving the model
    save_dir = f_parameters.MODEL_SAVE
    model_name = "model_cae_" + str(epoch_i) + "_" + str(score[1]) + ".h5"
    model_path = os.path.join(save_dir, model_name)
    autoencoder.save(model_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Saved trained model at %s ' % model_path)
    print("train model done!")


def CAE(X_train, Y_train, X_test, Y_test, im_balance=True, train=True, ver=True):
    X_test = np.array(X_test)
    X_train = np.array(X_train)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    print("X_train type ->", type(X_train))
    print("X_test type ->", type(X_test))
    print("X_train shape ->", X_train.shape)
    print("X_test shape ->", X_test.shape)
    if train:
        TrainCAEModel(X_train, Y_train, X_test, Y_test)


if __name__ == '__main__':
    path = f_parameters.DATA_PATH
    test = False
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = \
        f_load_data.f_load_data(path, test_mode=False)
    x_train, y_train, x_test, y_test = \
        merge_feature, merge_target, end_off_feature, merge_feature
    x_train, x_test = f_preprocess.high_dimension_big(x_train, edge=False), \
                      f_preprocess.high_dimension_big(x_test, edge=False)
    CAE(x_train, y_train, x_test, y_test, train=True, ver=True)
