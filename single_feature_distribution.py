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


def single_feature(data, feature, target):
    print("single feature distribution...")
    for i in range(parameters.END_OFF_DIS):
        print("X ->", target.columns[0], "\nY ->", feature.columns[i])
        # look at an individual feature in Seaborn through a boxplot
        sns.boxplot(x=target.columns[0], y=feature.columns[i], data=data)
        # kdeplot looking at single feature relations
        sns.FacetGrid(data, hue=target.columns[0], height=6).map(sns.kdeplot, feature.columns[i]).add_legend()
        # Denser regions of the data are fatter, and sparser thiner in a violin plot
        sns.violinplot(x=target.columns[0], y=feature.columns[i], data=data, size=6)
        plt.show()
    print("single feature distribution done.")


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path)
    single_feature(end_off, end_off_feature, end_off_target)
