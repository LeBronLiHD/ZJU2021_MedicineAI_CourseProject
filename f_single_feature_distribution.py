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
import f_load_data
import f_parameters
import math
import statistics
import heapq

import f_preprocess


def get_n_largest(list, n):
    ans, idx = 0, 0
    if n >= len(list):
        n = len(list) - 1
        return np.min(list)
    list_copy = []
    for i in range(len(list)):
        list_copy.append(list[i])
    for i in range(n):
        ans, idx = np.max(list_copy), np.argmax(list_copy)
        list_copy.pop(idx)
    return ans


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
    # X = [(i + 1) for i in range(f_parameters.END_OFF_COL - 1)]
    data_new = data.sample(frac=f_parameters.SAMPLE_RATIO_SINGLE).reset_index(drop=True)
    data_new = f_preprocess.data_normalization(data_new, have_target=True)
    print("data_new.shape ->", data_new.shape)
    print("data_new & target ->", data_new.columns[data_new.shape[1] - 1], target.columns[0])
    for i in range(0, f_parameters.END_OFF_COL - 3):
        print("Y ->", target.columns[0], "\t\t\tX ->", feature.columns[i])
        correlation[0].append(i)
        correlation[1].append(get_correlation(data_new, f_parameters.END_OFF_COL - 3, data_new, i))
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
    standard = get_n_largest(correlation[1], f_parameters.EXP_CORR_TOP)
    standard_h = get_n_largest(correlation[1], f_parameters.EXP_CORR_TOP_H)
    plt.plot(correlation[0], [standard] * len(correlation[0]), "-r")
    plt.xlabel("sequence number")
    plt.ylabel("correlation")
    plt.title("correlation of features")
    plt.show()
    important = [[], []]
    important_h = [[], []]
    for i in range(0, f_parameters.END_OFF_COL - 3):
        if correlation[1][i] >= standard:
            important[0].append(correlation[0][i])
            important[1].append(correlation[1][i])
            important_h[0].append(correlation[0][i])
            important_h[1].append(correlation[1][i])
        elif correlation[1][i] >= standard_h:
            important_h[0].append(correlation[0][i])
            important_h[1].append(correlation[1][i])
        else:
            continue
    print("single feature distribution done.")
    # plt.subplots(figsize=(data_new.shape[1], data_new.shape[1]))
    # sns.heatmap(data_new.corr(), annot=True, vmax=1, square=True,
    #             yticklabels=data_new.columns.values.tolist(),
    #             xticklabels=data_new.columns.values.tolist(), cmap="RdBu")
    # plt.title("heatmap")
    # plt.show()
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
    print("important ->", return_important[0])
    print("important_h ->", important_h[0])
    print("most_influence_index ->", most_influence_index)
    print("most_influence_feature ->", most_influence_feature)
    return return_important, important_h


if __name__ == '__main__':
    path = f_parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = f_load_data.f_load_data(path,
                                                                                                       test_mode=True)
    single_feature(end_off, end_off_feature, end_off_target, False)
