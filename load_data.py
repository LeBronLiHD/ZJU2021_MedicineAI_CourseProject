# -*- coding: utf-8 -*-

"""
load excel files
"""

import os
import parameters
import pandas
import openpyxl


def load_data(path):
    print("load data...")
    print("data path ->", path)
    files = []
    for file in os.listdir(path):
        print(file)
        files.append(os.path.join(path, file))
    end_col = [i for i in range(2, parameters.END_OFF_COL)]
    merge_col = [i for i in range(4, parameters.MERGE_COL)]
    end_off_feature = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1", usecols=end_col)
    merge_feature = pandas.read_excel(files[1], na_values="NaN", sheet_name="Sheet1", usecols=merge_col)
    end_off_target = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1", usecols=[parameters.END_OFF_COL])
    merge_target = pandas.read_excel(files[1], na_values="NaN", sheet_name="Sheet1", usecols=[parameters.MERGE_COL])
    end_off = pandas.merge(end_off_feature, end_off_target, left_index=True, right_index=True)
    merge = pandas.merge(merge_feature, merge_target, left_index=True, right_index=True)
    print("end_off.axes ->", end_off.axes)
    print("merge.axes ->", merge.axes)
    print("end_off_feature.shape ->", end_off_feature.shape)
    print("merge_feature.shape ->", merge_feature.shape)
    print("end_off_target.shape ->", end_off_target.shape)
    print("merge_target.shape ->", merge_target.shape)
    print("load data done.")
    return end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data(path)
