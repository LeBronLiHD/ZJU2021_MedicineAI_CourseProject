# -*- coding: utf-8 -*-

"""
load excel files
"""

import os
import parameters
import pandas
import openpyxl


def merge_data_judge(end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target):
    print("merge data for analysis and judge...")
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
    print("merge data for analysis and judge done.")
    return data, feature, target


def load_data_predict(path, test_mode=True):
    print("load data for predict...")
    files = []
    for file in os.listdir(path):
        print(file)
        files.append(os.path.join(path, file))
    divide_groups = []  # int
    groups = []  # Dataframe
    groups_feature = []  # Dataframe
    groups_target = []  # Dataframe
    basic_info_end_off = [0, 1]
    basic_info_merge = [0, 1, 2, 3]
    basic_data_end_off = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1", usecols=basic_info_end_off)
    basic_data_merge = pandas.read_excel(files[1], na_values="NaN", sheet_name="Sheet1", usecols=basic_info_merge)
    size_end_off = len(basic_data_end_off)
    size_merge = len(basic_data_merge)
    for i in range(size_end_off):
        if basic_data_end_off.at[i, basic_data_end_off.columns[basic_data_end_off.shape[1] - 1]] == 0:
            divide_groups.append(i)
    count_groups_end_off = len(divide_groups)
    for i in range(size_merge):
        if basic_data_merge.at[i, basic_data_merge.columns[basic_data_merge.shape[1] - 1]] == 0:
            divide_groups.append(i + size_end_off)
    print("count_groups =", len(divide_groups))
    count_groups = len(divide_groups)
    print("count_groups =", len(divide_groups))
    divide_groups.append(len(basic_data_merge) + len(basic_data_end_off))
    if test_mode:
        end_col = [i for i in range(2, parameters.END_OFF_COL)]
        end_off_feature = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1", usecols=end_col)
        end_off_target = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1",
                                           usecols=[parameters.END_OFF_COL])
        end_off = pandas.merge(end_off_feature, end_off_target, left_index=True, right_index=True)
        print("end_off.shape ->", end_off.shape)
        print("end_off_feature.shape ->", end_off_feature.shape)
        print("end_off_target.shape ->", end_off_target.shape)
        for i in range(count_groups_end_off):
            groups_feature.append(end_off_feature[divide_groups[i]:divide_groups[i + 1]])
            groups_target.append(end_off_target[divide_groups[i]:divide_groups[i + 1]])
            groups.append(end_off[divide_groups[i]:divide_groups[i + 1]])
        print("groups.shape ->", groups[0].shape)
        print("groups_feature.shape ->", groups_feature[0].shape)
        print("groups_target.shape ->", groups_target[0].shape)
    else:
        end_col = [i for i in range(2, parameters.END_OFF_COL)]
        merge_col = [i for i in range(4, parameters.MERGE_COL)]
        end_off_feature = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1", usecols=end_col)
        end_off_target = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1",
                                           usecols=[parameters.END_OFF_COL])
        end_off = pandas.merge(end_off_feature, end_off_target, left_index=True, right_index=True)
        merge_feature = pandas.read_excel(files[1], na_values="NaN", sheet_name="Sheet1", usecols=merge_col)
        merge_target = pandas.read_excel(files[1], na_values="NaN", sheet_name="Sheet1", usecols=[parameters.MERGE_COL])
        merge = pandas.merge(merge_feature, merge_target, left_index=True, right_index=True)
        data, feature, target = merge_data_judge(end_off, merge,
                                                 end_off_feature, merge_feature,
                                                 end_off_target, merge_target)
        print("end_off.shape ->", end_off.shape)
        print("merge.shape ->", merge.shape)
        print("end_off_feature.shape ->", end_off_feature.shape)
        print("merge_feature.shape ->", merge_feature.shape)
        print("end_off_target.shape ->", end_off_target.shape)
        print("merge_target.shape ->", merge_target.shape)
        for i in range(count_groups):
            groups_feature.append(feature[divide_groups[i]:divide_groups[i + 1]])
            groups_target.append(target[divide_groups[i]:divide_groups[i + 1]])
            groups.append(data[divide_groups[i]:divide_groups[i + 1]])
        print("groups.shape ->", groups[0].shape)
        print("groups_feature.shape ->", groups_feature[0].shape)
        print("groups_target.shape ->", groups_target[0].shape)
    print("count_groups =", len(divide_groups))
    print("load data for predict done.")
    return groups, groups_feature, groups_target


def load_data(path, test_mode=True):
    print("load data...")
    print("data path ->", path)
    files = []
    for file in os.listdir(path):
        print(file)
        files.append(os.path.join(path, file))
    end_col = [i for i in range(2, parameters.END_OFF_COL)]
    merge_col = [i for i in range(4, parameters.MERGE_COL)]
    if test_mode:
        end_off_feature = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1", usecols=end_col)
        merge_feature = None  # pandas.read_excel(files[1], na_values="NaN", sheet_name="Sheet1", usecols=merge_col)
        end_off_target = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1",
                                           usecols=[parameters.END_OFF_COL])
        merge_target = None  # pandas.read_excel(files[1], na_values="NaN", sheet_name="Sheet1",
                             # usecols=[parameters.MERGE_COL])
        end_off = pandas.merge(end_off_feature, end_off_target, left_index=True, right_index=True)
        merge = None  # pandas.merge(merge_feature, merge_target, left_index=True, right_index=True)
    else:
        end_off_feature = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1", usecols=end_col)
        merge_feature = pandas.read_excel(files[1], na_values="NaN", sheet_name="Sheet1", usecols=merge_col)
        end_off_target = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1",
                                           usecols=[parameters.END_OFF_COL])
        merge_target = pandas.read_excel(files[1], na_values="NaN", sheet_name="Sheet1", usecols=[parameters.MERGE_COL])
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
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data(path, test_mode=False)
    groups, groups_feature, groups_target = merge_data_judge(end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target)
