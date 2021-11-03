# -*- coding: utf-8 -*-

"""
use random forest to train model
"""

import r_ols
import f_parameters
import f_load_data
import f_preprocess
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import time
import f_model_analysis


def random_forest(end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target):
    end_off, end_off_feature, merge, merge_feature = \
        f_preprocess.data_normalization(end_off, have_target=True), \
        f_preprocess.data_normalization(end_off_feature, have_target=False), \
        f_preprocess.data_normalization(merge, have_target=True), \
        f_preprocess.data_normalization(merge_feature, have_target=False)
    model, pca_model = train_random_forest(merge_feature, merge_target, 2, pca=True, n_com=15)
    f_model_analysis.f_model_analysis(end_off_feature, end_off_target, model, end_off, 2, normal=True, pca=True, pca_model=pca_model)


def train_random_forest(X_train, Y_train, mode, pca=False, n_com=f_parameters.N_COMPONENTS):
    print("random_forest training...")
    init_time = time.time()
    forest = RandomForestClassifier(n_estimators=200, criterion="gini", max_depth=30, n_jobs=6, verbose=1)
    X_train, Y_train= f_preprocess.un_balance(X_train, Y_train, ratio="minority")
    PCA_model = None
    if pca:
        PCA_model = PCA(n_components=n_com)
        X_train = PCA_model.fit_transform(X_train.iloc[:, :-1])
    forest.fit(np.array(X_train), np.array(Y_train.values.ravel()))
    print("random_forest done, time ->", time.time() - init_time)
    f_model_analysis.Model_List_1_time[mode] = time.time() - init_time
    return forest, PCA_model


if __name__ == '__main__':
    path = f_parameters.DATA_PATH
    end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target = f_load_data.f_load_data(path,
                                                                                                       test_mode=False)
    random_forest(end_off, merge, end_off_feature, merge_feature, end_off_target, merge_target)
