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
    data_end_all = pandas.read_excel(files[0], na_values="NaN", sheet_name="Sheet1")
    data_merge = pandas.read_excel(files[1], na_values="NaN", sheet_name="Sheet1")
    print("data_end_all.axes ->", data_end_all.axes)
    print("data_merge.axes ->", data_merge.axes)
    print("load data done.")
    return data_end_all, data_merge


if __name__ == '__main__':
    path = parameters.DATA_PATH
    data_end_all, data_merge = load_data(path)
