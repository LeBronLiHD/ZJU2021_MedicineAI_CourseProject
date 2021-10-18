# -*- coding: utf-8 -*-

"""
preprocess of data
1. data cleaning
    1. missing value
        1. delete the piece of data
        2. interpolation of value
            1. replace
            2. nearest neighbor imputation
            3. regression method
            4. spline interpolation
    2. outliers
        1. simple statistical analysis
            1. observe the maximum and minimum values and determine whether it is reasonable
            2. three delta in normal distribution
            3. box plot analysis
                1. upper quartile and lower quartile
                2. overcome the problem that delta in distribution is under the influence of outliers
    3. duplicated data
        1. analysis first, and remove it if the duplicated data makes no sense
    3. inconsistent data
2. data transformation
    1. square, square root, exponent, logarithm, etc.
    2. normalization
        1. maximum and minimum normalization
        2. zero mean normalization
    3. discretization of continuous data
    4. attribute structure, like BMI
"""
import numpy
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import load_data
import parameters
from scipy.interpolate import lagrange
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NearMiss
import tensorflow as tf
import matplotlib.pyplot as plt
import math


def high_dimension_big_exp(data, edge=False):
    print("before reshaping ->", data.shape)
    print(type(data))
    data_expand = []
    for i in range(data.shape[0]):
        data_unit = []
        for j in range(parameters.COLUMNS_BIG):
            data_row = []
            for k in range(parameters.COLUMNS_BIG):
                if k == 0 or j == 0 or k == parameters.COLUMNS_BIG - 1 or j == parameters.COLUMNS_BIG - 1:
                    data_row.append(0)
                    continue
                if ((j - 1) * (parameters.COLUMNS_BIG - 2) + (k - 1)) // parameters.REPEAT < data.shape[1]:
                    data_row.append(data[data.columns[((j - 1) * (parameters.COLUMNS_BIG - 2) + (k - 1)) // parameters.REPEAT]].iat[i])
                else:
                    data_row.append(0)
            data_unit.append(data_row)
        data_expand.append(data_unit)
    data_expand = tf.expand_dims(data_expand, 3)
    print("after reshaping ->", numpy.shape(data_expand))
    return data_expand


def high_dimension_exp(data):
    print("before reshaping ->", data.shape)
    print(type(data))
    data_expand = []
    for i in range(data.shape[0]):
        data_unit = []
        for j in range(parameters.COLUMNS):
            data_row = []
            for k in range(parameters.COLUMNS):
                if k == 0 or j == 0 or k == parameters.COLUMNS - 1 or j == parameters.COLUMNS - 1:
                    data_row.append(0)
                    continue
                if (j - 1) * (parameters.COLUMNS - 2) + (k - 1) < data.shape[1]:
                    data_row.append(data[data.columns[(j - 1) * (parameters.COLUMNS - 2) + (k - 1)]].iat[i])
                else:
                    data_row.append(0)
            data_unit.append(data_row)
        data_expand.append(data_unit)
    data_expand = tf.expand_dims(data_expand, 3)
    print("after reshaping ->", numpy.shape(data_expand))
    return data_expand


def high_dimension_big(data, edge=False):
    print("before reshaping ->", data.shape)
    print(type(data))
    data_expand = []
    for i in range(data.shape[0]):
        data_unit = []
        for j in range(parameters.COLUMNS_BIG - 2):
            data_row = []
            for k in range(parameters.COLUMNS_BIG - 2):
                if (j * (parameters.COLUMNS_BIG - 2) + k) // parameters.REPEAT < data.shape[1]:
                    data_row.append(data[data.columns[(j * (parameters.COLUMNS_BIG - 2) + k) // parameters.REPEAT]].iat[i])
                else:
                    data_row.append(0)
            data_unit.append(data_row)
        data_expand.append(data_unit)
    data_expand = tf.expand_dims(data_expand, 3)
    print("after reshaping ->", numpy.shape(data_expand))
    return data_expand


def high_dimension(data):
    print("before reshaping ->", data.shape)
    print(type(data))
    data_expand = []
    for i in range(data.shape[0]):
        data_unit = []
        for j in range(parameters.COLUMNS - 2):
            data_row = []
            for k in range(parameters.COLUMNS - 2):
                if j * (parameters.COLUMNS - 2) + k < data.shape[1]:
                    data_row.append(data[data.columns[j * (parameters.COLUMNS - 2) + k]].iat[i])
                else:
                    data_row.append(0)
            data_unit.append(data_row)
        data_expand.append(data_unit)
    data_expand = tf.expand_dims(data_expand, 3)
    print("after reshaping ->", numpy.shape(data_expand))
    return data_expand


def transfer_to_image(data):
    for_show = []
    for k in range(np.shape(data)[1]):
        for_show_unit = []
        for j in range(np.shape(data)[2]):
            for_show_unit.append(data[k][j][0])
        for_show.append(for_show_unit)
    return for_show


def test_high_dimension(feature, total=5):
    print("feature.shape ->", feature.shape)
    ratio = total/feature.shape[0]
    feature = feature.sample(frac=ratio).reset_index(drop=True)
    print("feature_sample.shape ->", feature.shape)
    print("test high_dimension()")
    one = high_dimension_exp(feature)
    print("test high_dimension_big()")
    two = high_dimension_exp(feature)
    print("test high_dimension_exp()")
    three = high_dimension_exp(feature)
    print("test high_dimension_big_exp()")
    four = high_dimension_exp(feature)
    for i in range(total):
        plt.figure()
        plt.subplot(2, 2, 1)
        for_show = transfer_to_image(one[i])
        plt.imshow(for_show)
        plt.title("none")
        plt.subplot(2, 2, 2)
        for_show = transfer_to_image(two[i])
        plt.imshow(for_show)
        plt.title("exp")
        plt.subplot(2, 2, 3)
        for_show = transfer_to_image(three[i])
        plt.imshow(for_show)
        plt.title("big")
        plt.subplot(2, 2, 4)
        for_show = transfer_to_image(four[i])
        plt.imshow(for_show)
        plt.title("big_exp")
        plt.show()


def un_balance(X_train, Y_train, ratio="auto", mode=1, ensemble=False):
    if ensemble == False:
        if mode == 1:
            model = SMOTE(random_state=60, sampling_strategy=ratio, k_neighbors=8, n_jobs=None)
        elif mode == 2:
            model = SVMSMOTE(random_state=60, sampling_strategy=ratio, k_neighbors=8, m_neighbors=16, n_jobs=None)
        else:
            model = RandomOverSampler(sampling_strategy=ratio, random_state=60)
    else:
        model = NearMiss(sampling_strategy=ratio, version=3, n_neighbors=8, n_neighbors_ver3=3, n_jobs=None)
    X, Y = model.fit_resample(X_train, Y_train)
    size = len(Y)
    count = 0
    for i in range(size):
        if Y.at[i, Y.columns[Y.shape[1] - 1]] == 1:
            count += 1
    print("after im-balance ->", count, size - count, count / size)
    return X, Y


def ployinterp_column(data_piece, index, cycle=parameters.INTER_CYCLE):
    values = data_piece[list(range(index - cycle, index)) + list(range(index + 1, index + cycle + 1))]  # extract value
    values = values[values.notnull()]
    return lagrange(values.index, list(values))(index)


def data_cleaning(data):
    print("data cleaning...")
    # first, for missing value
    print("before cleaning, data.shape ->", data.shape)
    print("missing value of data ->")
    print(data[data.isnull().values==True])
    size = len(data)
    if data[data.isnull().values==True].empty:
        print("no nan data.")
    else:
        for feature in data.columns:
            for i in range(size):
                if data[feature].isnull()[i]:
                    data.at[i, feature] = ployinterp_column(data[feature], i)
    # second, for outliers
    print("after missing value process, data.shape ->", data.shape)
    toolarge, toosmall = 0, 0
    for feature in data.columns:
        upper_quartile = data[feature].quantile(0.75)
        lower_quartile = data[feature].quantile(0.25)
        for i in range(size):
            value = data.at[i, feature]
            if value >= 1.5 * (upper_quartile - lower_quartile) + upper_quartile:
                data.at[i, feature] = 1.5 * (upper_quartile - lower_quartile) + upper_quartile
                toolarge += 1
            if value <= lower_quartile - 1.5 * (upper_quartile - lower_quartile):
                data.at[i, feature] = lower_quartile - 1.5 * (upper_quartile - lower_quartile)
                toosmall += 1
    print("toolarge =", toolarge, "\t\ttoosmall =", toosmall)
    # third, duplicated data
    print("after outliers process, data.shape ->", data.shape)
    data.drop_duplicates(subset=None, keep="first", inplace=False, ignore_index=True)
    print("after duplicated value process, data.shape ->", data.shape)
    print("data cleaning done.")
    return data


def data_normalization_maxmin(data, have_target=False):
    print("data normalization maxmin...")
    answer = []
    for i in range(data.shape[0]):
        answer.append(data.at[i, data.columns[data.shape[1] - 1]])
    data_nor = (data - data.min())/(data.max() - data.min())
    for i in range(data_nor.shape[0]):
        data_nor.at[i, data.columns[data_nor.shape[1] - 1]] = answer[i]
    print("data normalization maxmin done.")
    return data_nor


def data_normalization(data, have_target=False):
    print("data normalization...")
    answer = []
    if have_target:
        for i in range(data.shape[0]):
            answer.append(data.at[i, data.columns[data.shape[1] - 1]])
    data_nor = (data - data.mean())/data.std()
    if have_target:
        for i in range(data_nor.shape[0]):
            data_nor.at[i, data.columns[data_nor.shape[1] - 1]] = answer[i]
    print("data normalization done.")
    return data_nor


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path,
                                                                                                       test_mode=True)
    test_high_dimension(end_off_feature)
    end_off_clean = data_cleaning(end_off)
