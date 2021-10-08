# -*- coding: utf-8 -*-

"""
preprocess od data
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

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import load_data
import parameters


def data_cleaning(data):
    print("data cleaning...")
    # first, for missing value
    # data = data.dropna()
    # second, for outliers
    print("data cleaning done.")


def data_normalization_maxmin(data):
    print("data normalization maxmin...")
    answer = []
    for i in range(data.shape[0]):
        answer.append(data.at[i, data.columns[data.shape[1] - 1]])
    data_nor = (data - data.min())/(data.max() - data.min())
    for i in range(data_nor.shape[0]):
        data_nor.at[i, data.columns[data_nor.shape[1] - 1]] = answer[i]
    print("data normalization maxmin done.")
    return data_nor


def data_normalization(data):
    print("data normalization...")
    answer = []
    for i in range(data.shape[0]):
        answer.append(data.at[i, data.columns[data.shape[1] - 1]])
    data_nor = (data - data.mean())/data.std()
    for i in range(data_nor.shape[0]):
        data_nor.at[i, data.columns[data_nor.shape[1] - 1]] = answer[i]
    print("data normalization done.")
    return data_nor


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path)
