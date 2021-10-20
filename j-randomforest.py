# -*- coding: utf-8 -*-

"""
use random forest to train model
"""

import liner_regression_ols
import parameters
import load_data
import preprocess
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import time
import model_analysis


def random_forest(end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target):
    end_off, end_off_feature, merge, merge_feature = \
        preprocess.data_normalization(end_off, have_target=True), \
        preprocess.data_normalization(end_off_feature, have_target=False), \
        preprocess.data_normalization(merge, have_target=True), \
        preprocess.data_normalization(merge_feature, have_target=False)
    model = train_random_forest(merge_feature, merge_target, 2)
    model_analysis.model_analysis(end_off_feature, end_off_target, model, end_off, 2, normal=True)


def train_random_forest(X_train, Y_train, mode):
    print("random_forest training...")
    init_time = time.time()
    forest = RandomForestClassifier(n_estimators=1000, criterion="gini", max_depth=100, max_features=None, n_jobs=6,
                                    verbose=1)
    X_train, Y_train= preprocess.un_balance(X_train, Y_train, ratio="minority")
    forest.fit(np.array(X_train), np.array(Y_train.values.ravel()))
    print("random_forest done, time ->", time.time() - init_time)
    model_analysis.Model_List_1_time[mode] = time.time() - init_time
    return forest


if __name__ == '__main__':
    path = parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = load_data.load_data(path,
                                                                                                       test_mode=False)
    random_forest(end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target)
