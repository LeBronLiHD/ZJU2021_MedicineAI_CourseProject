# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import parameters
import load_data
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import os
from sklearn.model_selection import train_test_split
import preprocess
from tensorflow.keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
matplotlib.use('agg')


def NN(data, feature, target):
    X_train, Y_train = feature, target
    X_train, Y_train = preprocess.un_balance(X_train, Y_train)
    _, X_test, _, Y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=1)
    X_train, X_test = preprocess.data_normalization(X_train), preprocess.data_normalization(X_test)
    feature = np.array(feature)
    target = np.array(target)
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    Y_train, Y_test = to_categorical(Y_train, num_classes=3), to_categorical(Y_test, num_classes=3)
    print("feature.shape ->", np.shape(feature))
    print("target.shape ->", np.shape(target))
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
    epoch_number = 100
    history = model.fit(X_train, Y_train,
                        batch_size=16, epochs=epoch_number,
                        verbose=1,
                        validation_data=0.1)

    # plotting the metrics
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.tight_layout()

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
    print("train model done!")


if __name__ == '__main__':
    path = parameters.DATA_PATH
    test = True
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path,
                                                                                                       test_mode=test)
    if test == False:
        data, data_feature, data_target = load_data.merge_data(end_off, merge,
                                                               end_off_feature, merge_feature,
                                                               end_off_target, merge_target)
    else:
        data, data_feature, data_target = end_off, end_off_feature, end_off_target
    NN(data, data_feature, data_target)
