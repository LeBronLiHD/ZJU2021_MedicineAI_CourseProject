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

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import load_data
import parameters
from scipy.interpolate import lagrange


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
    end_off_clean = data_cleaning(end_off)
