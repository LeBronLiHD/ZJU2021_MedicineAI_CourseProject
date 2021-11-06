# -*- coding: utf-8 -*-

"""
RNN for regression
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import f_parameters
import f_load_data
from sklearn.metrics import roc_auc_score, roc_curve
from keras.callbacks import EarlyStopping
import os
from sklearn.model_selection import train_test_split
import f_preprocess
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import random
import time
import f_model_analysis
from keras.models import Sequential, load_model
from keras.utils import np_utils
from keras import layers
from keras import models


# Build the RNN model
def build_model(units, input_dim, output_size, allow_cudnn_kernel=True):
    # CuDNN is only available at the layer level, and not at the cell level.
    # This means `LSTM(units)` will use the CuDNN kernel,
    # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
    if allow_cudnn_kernel:
        # The LSTM layer with default options uses CuDNN.
        lstm_layer = layers.LSTM(units, input_shape=(None, input_dim))
    else:
        # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
        lstm_layer = layers.RNN(
            layers.LSTMCell(units), input_shape=(None, input_dim)
        )
    model = models.Sequential(
        [
            lstm_layer,
            layers.BatchNormalization(),
            layers.Dense(output_size),
        ]
    )
    return model


def RNN(X_train, Y_train, X_test, Y_test, imbalance=True, nor=False):
    # TODO: f_preprocess
    batch_size = 64
    # Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
    # Each input sequence will be of size (28, 28) (height is treated like time).
    input_dim = 28
    units = 64
    output_size = 10  # labels are from 0 to 9
    TrainRNNModel(X_train, Y_train, X_test, Y_test)


def TrainRNNModel(X_train, Y_train, X_test, Y_test):
    init_time = time.time()
    # build rnn model
    model = Sequential()
    # Add an Embedding layer expecting input vocab of size 1000, and
    # output embedding dimension of size 64.
    model.add(layers.Embedding(input_dim=f_parameters.RNN_INPUT_DIM, output_dim=64))
    # The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
    model.add(layers.GRU(256, return_sequences=True))
    # Add a LSTM layer with 128 internal units.
    model.add(layers.LSTM(128))
    # Add a Dense layer with 10 units.
    model.add(layers.Dense(10))


if __name__ == '__main__':
    path = f_parameters.DATA_PATH
    test = False
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = \
        f_load_data.f_load_data_predict(path, test_mode=False)

