# -*- coding: utf-8 -*-

"""
use decision tree to train model
Decision Tree is a white box type of ML algorithm. It shares internal decision-making logic,
which is not available in the black box type of algorithms such as Neural Network
"""

import parameters
import load_data
import preprocess
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pandas
import single_feature_distribution


def decision_tree(data, feature, target):
    print("feature ->", feature.columns)
    print("target ->", target.columns)
    feature = preprocess.data_normalization(feature)
    X_train, X_test, Y_train, Y_test = train_test_split(feature, target, test_size=0.35, random_state=1)


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path)
    decision_tree(end_off, end_off_feature, end_off_target)
