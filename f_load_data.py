# -*- coding: utf-8 -*-

"""
load excel files
"""

import os
import f_parameters
import pandas
import openpyxl
import warnings


def merge_data(end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target):
    print("merge data ...")
    all_label_same = 0
    for i in range(end_off.shape[1]):
        if end_off.columns[i] != merge.columns[i]:
            all_label_same += 1
    print("all_label_same =", all_label_same)
    data = pandas.concat([end_off, merge], ignore_index=True, keys=None)
    feature = pandas.concat([end_off_feature, merge_feature], ignore_index=True, keys=None)
    target = pandas.concat([end_off_target, merge_target], ignore_index=True, keys=None)
    print("data.shape ->", data.shape)
    print("feature.shape ->", feature.shape)
    print("target.shape ->", target.shape)
    print("merge data done.")
    return data, feature, target


def f_load_data_predict(path, test_mode=True):
    print("load data for predict...")
    print("test_mode =", test_mode)
    files = []
    for file in os.listdir(path):
        print(file)
        files.append(os.path.join(path, file))
    divide_groups_test = []  # int
    divide_groups_train = []
    # groups_test = []  # Dataframe
    groups_test_feature = []  # Dataframe
    groups_test_target = []  # Dataframe
    # groups_train = []  # Dataframe
    groups_train_feature = []  # Dataframe
    groups_train_target = []  # Dataframe
    basic_info_end_off = [0, 1]
    basic_info_merge = [0, 1, 2, 3]
    basic_data_end_off = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1", usecols=basic_info_end_off)
    basic_data_merge = pandas.read_excel(files[1], na_values="NaN", sheet_name="Sheet1", usecols=basic_info_merge)
    size_end_off = len(basic_data_end_off)
    size_merge = len(basic_data_merge)

    divide_groups_test.append(0)
    for i in range(1, size_end_off):
        former = basic_data_end_off.at[i - 1, basic_data_end_off.columns[basic_data_end_off.shape[1] - 1]]
        if basic_data_end_off.at[i, basic_data_end_off.columns[basic_data_end_off.shape[1] - 1]] < former:
            divide_groups_test.append(i)
    divide_groups_test.append(size_end_off)
    # divide_groups.append(0 + size_end_off)

    divide_groups_train.append(0)
    for i in range(1, size_merge):
        former = basic_data_merge.at[i - 1, basic_data_merge.columns[basic_data_merge.shape[1] - 1]]
        if basic_data_merge.at[i, basic_data_merge.columns[basic_data_merge.shape[1] - 1]] < former:
            divide_groups_train.append(i)
    divide_groups_train.append(size_merge)

    # print("count_groups =", len(divide_groups_train))
    # count_groups = len(divide_groups)
    print("groups for train ->", len(divide_groups_train) - 1)
    print("groups for test ->", len(divide_groups_test) - 1)
    count_groups_test, count_groups_train = len(divide_groups_test) - 1, len(divide_groups_train) - 1
    # divide_groups.append(len(basic_data_merge) + len(basic_data_end_off))

    end_col = [i for i in range(2, f_parameters.END_OFF_COL - 1)]
    merge_col = [i for i in range(4, f_parameters.MERGE_COL - 1)]

    end_off_feature = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1", usecols=end_col)
    end_off_target = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1",
                                       usecols=[f_parameters.END_OFF_COL])
    # end_off = pandas.merge(end_off_feature, end_off_target, left_index=True, right_index=True)
    merge_feature = pandas.read_excel(files[1], na_values="NaN", sheet_name="Sheet1", usecols=merge_col)
    merge_target = pandas.read_excel(files[1], na_values="NaN", sheet_name="Sheet1", usecols=[f_parameters.MERGE_COL])
    # merge = pandas.merge(merge_feature, merge_target, left_index=True, right_index=True)

    print("end_off_feature.shape ->", end_off_feature.shape)
    print("merge_feature.shape ->", merge_feature.shape)
    print("end_off_target.shape ->", end_off_target.shape)
    print("merge_target.shape ->", merge_target.shape)

    for i in range(count_groups_test):
        for j in range(divide_groups_test[i], divide_groups_test[i + 1]):
            groups_test_feature_piece = []
            for k in range(f_parameters.RNN_INPUT_DIM):
                groups_test_feature_piece.append(end_off_feature.at[j, end_off_feature.columns[k]])
            groups_test_feature.append(groups_test_feature_piece)
        # groups_test_feature.append(end_off_feature[divide_groups_test[i]:divide_groups_test[i + 1]])
        groups_test_target.append(end_off_target.at[divide_groups_test[i + 1] - 1, end_off_target.columns[0]])
        # groups_test.append(end_off[divide_groups_test[i]:divide_groups_test[i + 1]])
    for i in range(count_groups_train):
        for j in range(divide_groups_train[i], divide_groups_train[i + 1]):
            groups_train_feature_piece = []
            for k in range(f_parameters.RNN_INPUT_DIM):
                groups_train_feature_piece.append(merge_feature.at[j, merge_feature.columns[k]])
            groups_train_feature.append(groups_train_feature_piece)
        # groups_train_feature.append(merge_feature[divide_groups_train[i]:divide_groups_train[i + 1]])
        groups_train_target.append(merge_target.at[divide_groups_train[i + 1] - 1, merge_target.columns[0]])
        # groups_train.append(merge[divide_groups_train[i]:divide_groups_train[i + 1]])
    print("load data for predict done.")
    if test_mode:
        return groups_test_feature, groups_test_target, None, None
    else:
        return groups_train_feature, groups_train_target, groups_test_feature, groups_test_target


def f_load_data(path, test_mode=True):
    print("load data ...")
    print("test_mode =", test_mode)
    print("data path ->", path)
    files = []
    for file in os.listdir(path):
        print(file)
        files.append(os.path.join(path, file))
    end_col = [i for i in range(2, f_parameters.END_OFF_COL)]
    merge_col = [i for i in range(4, f_parameters.MERGE_COL)]
    if test_mode:
        end_off_feature = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1", usecols=end_col)
        merge_feature = None  # pandas.read_excel(files[1], na_values="NaN", sheet_name="Sheet1", usecols=merge_col)
        end_off_target = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1",
                                           usecols=[f_parameters.END_OFF_COL])
        merge_target = None  # pandas.read_excel(files[1], na_values="NaN", sheet_name="Sheet1",
                             # usecols=[f_parameters.MERGE_COL])
        end_off = pandas.merge(end_off_feature, end_off_target, left_index=True, right_index=True)
        merge = None  # pandas.merge(merge_feature, merge_target, left_index=True, right_index=True)
    else:
        end_off_feature = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1", usecols=end_col)
        merge_feature = pandas.read_excel(files[1], na_values="NaN", sheet_name="Sheet1", usecols=merge_col)
        end_off_target = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1",
                                           usecols=[f_parameters.END_OFF_COL])
        merge_target = pandas.read_excel(files[1], na_values="NaN", sheet_name="Sheet1", usecols=[f_parameters.MERGE_COL])
        end_off = pandas.merge(end_off_feature, end_off_target, left_index=True, right_index=True)
        merge = pandas.merge(merge_feature, merge_target, left_index=True, right_index=True)
    if test_mode:
        print("end_off.axes ->", end_off.axes)
        # print("merge.axes ->", merge.axes)
        print("end_off_feature.shape ->", end_off_feature.shape)
        # print("merge_feature.shape ->", merge_feature.shape)
        print("end_off_target.shape ->", end_off_target.shape)
        # print("merge_target.shape ->", merge_target.shape)
    else:
        print("end_off.shape ->", end_off.shape)
        print("merge.shape ->", merge.shape)
        print("end_off_feature.shape ->", end_off_feature.shape)
        print("merge_feature.shape ->", merge_feature.shape)
        print("end_off_target.shape ->", end_off_target.shape)
        print("merge_target.shape ->", merge_target.shape)
    print("load data done.")
    return end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target


if __name__ == '__main__':
    path = f_parameters.DATA_PATH
    mode = False  # True for judge and False for predict
    if mode:
        end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = f_load_data(path, test_mode=False)
    else:
        f_load_data_predict(path, test_mode=False)
