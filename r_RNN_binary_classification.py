# -*- coding: utf-8 -*-

"""
RNN for binary classification
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
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras import layers


def RNN(X_train, Y_train, X_test, Y_test, imbalance=True, nor=False):
    # TODO: f_preprocess
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

