# -*- coding: utf-8 -*-

"""
detect the distribution of all features in different diseases

also, for end_all_1230_ch only
"""

import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import load_data
import parameters
import preprocess
from sklearn.utils import shuffle
import single_feature_distribution


def heatmap(data, important):
    data = data.sample(frac=parameters.SAMPLE_RATIO).reset_index(drop=True)
    data = preprocess.data_normalization(data, have_target=True)
    print("data.shape ->", data.shape)
    important[0].append(data.shape[1] - 1)
    select_col = []
    for i in range(len(important[0])):
        select_col.append(data.columns[important[0][i]])
    data_selected = pandas.DataFrame(data, columns=select_col)
    print("data_selected.shape ->", data_selected.shape)
    print("data_selected.columns ->", data_selected.columns)
    size = len(data_selected.columns)
    plt.subplots(figsize=(size, size))
    sns.heatmap(data_selected.corr(), annot=True, vmax=1, square=True,
                yticklabels=data_selected.columns.values.tolist(),
                xticklabels=data_selected.columns.values.tolist(), cmap="RdBu")
    plt.title("heatmap")
    plt.show()


def multi_feature(data, important):
    print("multi-feature distribution...")
    # preprocessing
    data = data.sample(frac=parameters.SAMPLE_RATIO).reset_index(drop=True)
    data = preprocess.data_normalization(data, have_target=True)
    print("data.shape ->", data.shape)
    important[0].append(data.shape[1] - 1)
    select_col = []
    for i in range(len(important[0])):
        select_col.append(data.columns[important[0][i]])
    data_selected = pandas.DataFrame(data, columns=select_col)
    print("data_selected.shape ->", data_selected.shape)
    print("data_selected.columns ->", data_selected.columns)
    size = len(data_selected.columns)
    # Andrews Curves involve using attributes of samples as coefficients for Fourier series and then plotting these
    pd.plotting.andrews_curves(data_selected, data_selected.columns[size - 1], color=["green", "red"])
    plt.title("andrews_curves")
    plt.show()
    # Parallel coordinates plots each feature on a separate column &
    # then draws lines connecting the features for each data sample
    pd.plotting.parallel_coordinates(data_selected, data_selected.columns[size - 1], color=["green", "red"])
    plt.title("parallel_coordinates")
    plt.show()
    # radviz  puts each feature as a point on a 2D plane, and then simulates
    # having each sample attached to those points through a spring weighted by the relative value for that feature
    pd.plotting.radviz(data_selected, data_selected.columns[size - 1], color=["green", "red"])
    plt.title("radviz")
    plt.show()
    print("multi-feature distribution done.")


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path,
                                                                                                       test_mode=True)
    # end_off, merge, end_off_feature, merge_feature = \
    #     preprocess.data_cleaning(end_off), preprocess.data_cleaning(merge), \
    #     preprocess.data_cleaning(end_off_feature), preprocess.data_cleaning(merge_feature)
    important, important_h = single_feature_distribution.single_feature(end_off, end_off_feature, end_off_target, False)
    heatmap(end_off, important_h)
    multi_feature(end_off, important)
