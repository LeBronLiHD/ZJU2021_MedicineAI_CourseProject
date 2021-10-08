# -*- coding: utf-8 -*-

"""
detect the distribution of various features in different diseases

since merge_reform_age_1230 is more in regression while end_all_1230_ch is more in distribution,
feature distribution is only for end_all_1230_ch
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import load_data
import parameters
import math
import statistics
import heapq

import preprocess


def get_correlation(data_1, col_1, data_2, col_2):
    rows = data_1.shape[0]  # data_1.shape[0] == data_2.shape[0]
    num_1 = []
    num_2 = []
    for i in range(rows):
        num_1.append(data_1.at[i, data_1.columns[col_1]])
        num_2.append(data_2.at[i, data_2.columns[col_2]])
    cov_up = 0
    mean_1 = statistics.mean(num_1)
    mean_2 = statistics.mean(num_2)
    for i in range(rows):
        cov_up += (num_1[i] - mean_1) * (num_2[i] - mean_2)
    cov = cov_up/(rows - 1)
    return abs(cov/(statistics.pstdev(num_1) * statistics.pstdev(num_2)))


def single_feature(data, feature, target, show_image):
    print("single feature distribution...")
    correlation = [[], []]
    # X = [(i + 1) for i in range(parameters.END_OFF_COL - 1)]
    data_new = data.sample(frac=parameters.SAMPLE_RATIO_SINGLE).reset_index(drop=True)
    data_new = preprocess.data_normalization(data_new)
    print("data_new.shape ->", data_new.shape)
    for i in range(0, parameters.END_OFF_COL - 3):
        print("Y ->", target.columns[0], "\t\t\tX ->", feature.columns[i])
        correlation[0].append(i)
        correlation[1].append(get_correlation(data_new, parameters.END_OFF_COL - 3, data_new, i))
        if show_image:
            # look at an individual feature in Seaborn through a boxplot
            sns.boxplot(x=target.columns[0], y=feature.columns[i], data=data)
            plt.show()
            # kdeplot looking at single feature relations
            sns.FacetGrid(data, hue=target.columns[0], height=6).map(sns.kdeplot, feature.columns[i]).add_legend()
            plt.show()
            # Denser regions of the data are fatter, and sparser thiner in a violin plot
            sns.violinplot(x=target.columns[0], y=feature.columns[i], data=data, size=6)
            plt.show()
    plt.figure()
    plt.plot(correlation[0], correlation[1], ".b")
    standard = max(correlation[1]) * parameters.EXP_CORR_RATE
    plt.plot(correlation[0], [standard] * len(correlation[0]), "-r")
    plt.xlabel("sequence number")
    plt.ylabel("correlation")
    plt.title("correlation of features")
    plt.show()
    important = [[], []]
    for i in range(0, parameters.END_OFF_COL - 3):
        if correlation[1][i] >= standard:
            important[0].append(correlation[0][i])
            important[1].append(correlation[1][i])
    print("single feature distribution done.")
    print("number of important features =", len(important[0]))
    most_influence_index = []
    most_influence_feature = []
    return_important = [[], []]
    for i in range(len(important[0])):
        return_important[0].append(important[0][i])
        return_important[1].append(important[1][i])
    for i in range(len(important[0])):
        index = important[1].index(max(important[1]))
        most_influence_index.append(important[0][index])
        most_influence_feature.append(data_new.columns[important[0][index]])
        important[1].remove(important[1][index])
        important[0].remove(important[0][index])
    print("important ->", important)
    print("most_influence_index ->", most_influence_index)
    print("most_influence_feature ->", most_influence_feature)
    return return_important


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path)
    single_feature(end_off, end_off_feature, end_off_target, False)
