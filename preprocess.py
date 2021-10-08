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
import load_data
import parameters


def data_cleaning(data):
    print("data cleaning...")
    print("data cleaning done.")


def data_transformation(data):
    print("data transformation...")
    print("data transformation done.")


def data_out_of_order(data):
    # out-of-order process
    print("data out of order...")
    print("data out of order done.")


def data_sample(data, ratio):
    # shrink the amount of data
    # new scale = original scale * ratio
    # 1. out-of-order process
    # 2. extract data piece in a particular cycle
    print("data sample...")
    print("data sample done.")


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path)
