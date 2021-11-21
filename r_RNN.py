# -*- coding: utf-8 -*-

"""
RNN for regression
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import f_parameters
import f_load_data
from keras import losses
import f_preprocess
import os
import time
import f_model_analysis
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras import layers
from keras import models


# Build the RNN model
def build_model(units, high_middle_size, input_dim, output_size, middle_size, allow_cudnn_kernel=True):
    # CuDNN is only available at the layer level, and not at the cell level.
    # This means `LSTM(units)` will use the CuDNN kernel,
    # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
    if allow_cudnn_kernel:
        # The LSTM layer with default options uses CuDNN.
        lstm_layer = layers.LSTM(units, input_shape=(None, input_dim))
    else:
        # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
        lstm_layer = layers.RNN(
            layers.LSTMCell(units), input_shape=(None, input_dim),
        )
    model = models.Sequential(
        [
            lstm_layer,
            layers.Dense(middle_size),
            layers.Dense(middle_size),
            layers.Dense(high_middle_size),
            layers.BatchNormalization(),
            layers.Dense(high_middle_size),
            layers.Dense(middle_size),
            layers.Dense(middle_size),
            layers.Dense(output_size)
        ]
    )
    return model


def RNN(x_train, y_train, x_test, y_test):
    # TODO: f_preprocess
    middle_size = 1024
    high_middle_size = middle_size * 2
    batch_size = 256
    input_dim = 61
    units = 1024
    gru_units = units * 2
    output_size = 2
    model = build_model(units, high_middle_size, input_dim, output_size, middle_size, allow_cudnn_kernel=False)
    model.compile(
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="sgd",
        metrics=["accuracy"],
    )
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        batch_size=batch_size, epochs=f_parameters.EPOCH_RNN,
        verbose=1, shuffle=True
    )
    print(model.summary())

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    history.history['val_loss'][0] = min(1.0, history.history['val_loss'][0])
    history.history['loss'][0] = min(1.0, history.history['loss'][0])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    score = model.evaluate(x_test, y_test)
    print('model accuracy ->', score[1])
    # saving the model
    save_dir = f_parameters.MODEL_SAVE
    model_name = "model_rnn_" + str(f_parameters.EPOCH_RNN) + "_" + str(score[1]) + ".h5"
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('saved trained model at %s ' % model_path)


def simpleRNNModel():
    init_time = time.time()
    # build rnn model
    model = Sequential()
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
    model.add(layers.Embedding(input_dim=f_parameters.RNN_INPUT_DIM, output_dim=64))
    model.add(layers.LSTM(256))
    model.add(layers.GRU(512, return_sequences=True))
    model.add(layers.LSTM(512))
    model.add(layers.GRU(512, return_sequences=True))
    model.add(layers.LSTM(256))
    model.add(layers.Dense(10))
    return model


if __name__ == '__main__':
    path = f_parameters.DATA_PATH
    test = False
    x_train, y_train, x_test, y_test = \
        f_load_data.f_load_data_predict(path, test_mode=False, normal=False)
    x_train, y_train = f_preprocess.reshape_width_height(x_train, y_train)
    x_test, y_test = f_preprocess.reshape_width_height(x_test, y_test)
    RNN(x_train, y_train, x_test, y_test)

