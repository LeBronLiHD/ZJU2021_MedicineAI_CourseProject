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


def vertify_model(test, expect, x_pd_train, x_pd_test):
    print("test.shape ->", np.shape(test))
    print("expect.shape ->", np.shape(expect))
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
    print("model auc ->", auc)
    print("right =", right)
    print("fault =", size - right)
    print("overall right ratio =", right / size)
    print("0 right ratio =", right_0_1[0] / (size - count))
    print("1 right ratio =", right_0_1[1] / count)
    print("right_0_1 ->", right_0_1)
    print("error_0_1 ->", error_0_1)
    print("impossible_r_e ->", impossible_r_e)

    X_pred = model.predict(np.array(x_pd_test))
    X_pred = pd.DataFrame(X_pred,
                          columns=x_pd_test.columns)
    X_pred.index = x_pd_test.index

    threshod = 0.3
    scored = pd.DataFrame(index=x_pd_test.index)
    scored['Loss_mae'] = np.mean(np.abs(X_pred - x_pd_test), axis=1)
    scored['Threshold'] = threshod
    scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
    scored.head()

    X_pred_train = model.predict(np.array(x_pd_train))
    X_pred_train = pd.DataFrame(X_pred_train,
                                columns=x_pd_train.columns)
    X_pred_train.index = x_pd_train.index

    scored_train = pd.DataFrame(index=x_pd_train.index)
    scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train - x_pd_train), axis=1)
    scored_train['Threshold'] = threshod
    scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
    scored = pd.concat([scored_train, scored])

    scored.plot(logy=True, figsize=(10, 6), ylim=[1e-2, 1e2], color=['blue', 'red'])


def TrainAEModel(x_train, y_train, x_test, y_test, x_pd_train):
    input_img = Input(shape=(f_parameters.DIM_NUM,))
    encoded = Dense(512, activation='relu')(input_img)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = Dense(128, activation='relu')(encoded)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(f_parameters.DIM_NUM, activation='sigmoid')(decoded)
    autoencoder = Model(input_img, input_img, decoded)
    autoencoder.compile(loss='binary_crossentropy', optimizer='adam')
    epoch_i = f_parameters.EPOCH_AE
    count = 0
    for i in range(y_train.shape[0]):
        if np.argmax(y_train[i]) == 1:
            count += 1
    class_weigh = {0: (y_train.shape[0] - count) / y_train.shape[0], 1: count / y_train.shape[0]}
    print(class_weigh)
    print("epoch ->", epoch_i)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, mode='min')
    history = autoencoder.fit(x_train, x_train, batch_size=64, epochs=epoch_i, verbose=1,
                              callbacks=[early_stopping],
                              # validation_split=0.1,
                              validation_data=(x_test, x_test),
                              class_weight=class_weigh,
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

    X_pred = autoencoder.predict(x_train)
    X_pred = pd.DataFrame(X_pred,
                          columns=x_pd_train.columns)
    X_pred.index = x_pd_train.index

    scored = pd.DataFrame(index=x_pd_train.index)
    scored['Loss_mae'] = np.mean(np.abs(X_pred - x_pd_train), axis=1)
    plt.figure()
    sns.distplot(scored['Loss_mae'],
                 bins=10,
                 kde=True,
                 color='blue')
    plt.xlim([0.0, .5])
    plt.show()

    score = autoencoder.evaluate(x_test, y_test)
    print('model accuracy ->', score[1])
    # saving the model
    save_dir = f_parameters.MODEL_SAVE
    model_name = "model_ae_" + str(epoch_i) + "_" + str(score[1]) + ".h5"
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
    if train:
        TrainAEModel(X_train, Y_train, X_test, Y_test, x_pd_train)
    if ver:
        vertify_model(X_test, Y_test, x_pd_train, x_pd_test)


if __name__ == '__main__':
    path = f_parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = \
        f_load_data.f_load_data(path, test_mode=False)
    AE(merge_feature, merge_target, end_off_feature, end_off_target, merge_feature, end_off_feature)
