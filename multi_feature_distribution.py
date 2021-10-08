# -*- coding: utf-8 -*-

"""
detect the distribution of all features in different diseases

also, for end_all_1230_ch only
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import load_data
import parameters


def single_feature(data, feature, target):
    print("multi-feature distribution...")
    # preprocessing
    # Parallel coordinates plots each feature on a separate column &
    # then draws lines connecting the features for each data sample
    pd.plotting.andrews_curves(data, target.columns[0])
    # radviz  puts each feature as a point on a 2D plane, and then simulates
    # having each sample attached to those points through a spring weighted by the relative value for that feature
    pd.plotting.parallel_coordinates(data, target.columns[0])
    plt.show()
    print("multi-feature distribution done.")


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path)
    single_feature(end_off, end_off_feature, end_off_target)
